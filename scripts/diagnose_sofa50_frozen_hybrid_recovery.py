#!/usr/bin/env python3
from __future__ import annotations

"""Read-only fusion of frozen Sofa50 Arm-B Laplacians and Arm-E vertices."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import (
    ARM_B,
    ARM_E,
    GROUPS,
    MILD,
    SPECTRAL_BANDS,
    SPECTRAL_PROTOCOL,
    STRONG,
    VARIANTS,
    _payload,
    _starts,
    _variant,
    spectral_band_components,
)
from mlr.data import Mesh
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_H = "Hybrid_B_laplacian_E_anchor"
SWEEP_LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
PCG_TOLERANCE = 1e-4
PCG_MAXIMUM_ITERATIONS = 2048
PRIMARY_FIELDS = (
    "refined_chamfer",
    "same_index_recovered_vertex_rms",
    "p2s_p95",
    "fscore",
    "normal_consistency",
    "normalized_flip_rate",
)
ANALYSIS_GROUPS = {
    **GROUPS,
    "subdivided": set(VARIANTS) - {"A1", "A2"},
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_rows(payload: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in payload["rows"] if row["split"] == split]


def _archived_predictions(report: Path, arm: str, split: str) -> np.ndarray:
    path = report / "shards" / f"{arm}_prediction_arrays.npz"
    return np.load(path)[f"{split}_prediction"].astype(np.float64)


def _inputs(args: argparse.Namespace, split: str):
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
    b_payload = _payload(args.arm_b_report.resolve(), ARM_B)
    e_payload = _payload(args.arm_e_report.resolve(), ARM_E)
    b_rows, e_rows = _split_rows(b_payload, split), _split_rows(e_payload, split)
    expected = list(dataset.sample_ids)
    if [row["sample_id"] for row in b_rows] != expected:
        raise RuntimeError(f"{split}: frozen Arm B IDs/order differ from manifest")
    if [row["sample_id"] for row in e_rows] != expected:
        raise RuntimeError(f"{split}: frozen Arm E IDs/order differ from manifest")
    b_array = _archived_predictions(args.arm_b_report.resolve(), ARM_B, split)
    e_array = _archived_predictions(args.arm_e_report.resolve(), ARM_E, split)
    return (
        dataset,
        b_payload,
        e_payload,
        b_rows,
        e_rows,
        b_array,
        e_array,
        _starts(b_rows, b_array),
        _starts(e_rows, e_array),
    )


def _pcg(
    b_prediction: np.ndarray,
    direct_vertices: np.ndarray,
    static: Mapping[str, Any],
    regularization: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = torch.as_tensor(b_prediction, dtype=torch.float64, device=device)
    anchor = torch.as_tensor(direct_vertices, dtype=torch.float64, device=device)
    edge_index = torch.as_tensor(static["edge_index"], dtype=torch.long, device=device)
    degree = torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        recovered, audit = recovery_forward_audit(
            prediction,
            anchor,
            edge_index,
            degree,
            regularization=float(regularization),
            maximum_iterations=PCG_MAXIMUM_ITERATIONS,
            tolerance=PCG_TOLERANCE,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    vertices = recovered.detach().cpu().numpy()
    return vertices, {
        "pcg_iterations": audit.iterations,
        "pcg_converged": audit.converged,
        "pcg_relative_residual": audit.relative_residual,
        "pcg_runtime_seconds": runtime,
        "pcg_dtype": "float64",
        "pcg_tolerance": PCG_TOLERANCE,
        "pcg_maximum_iterations": PCG_MAXIMUM_ITERATIONS,
    }


def _row(
    split: str,
    arm: str,
    sample_id: str,
    index: int,
    vertices: np.ndarray,
    clean: Mesh,
    initial: Mesh,
    metric: Mapping[str, Any],
    solver: Mapping[str, Any] | None = None,
    regularization: float | None = None,
) -> dict[str, Any]:
    initial_metric = _geometry_row(split, sample_id, "initial", initial, clean, initial)
    initial_cd = float(initial_metric["chamfer"])
    refined_cd = float(metric["chamfer"])
    result = {
        "split": split,
        "arm": arm,
        "sample_id": sample_id,
        "index": index,
        "variant": _variant(sample_id),
        "vertices": initial.num_vertices,
        "faces": initial.num_faces,
        "initial_chamfer": initial_cd,
        "refined_chamfer": refined_cd,
        "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
        "eta": (initial_cd - refined_cd) / initial_cd,
        "p2s": float(metric["p2s"]),
        "p2s_p95": float(metric["p2s_p95"]),
        "fscore": float(metric["fscore"]),
        "normal_consistency": float(metric["normal_consistency"]),
        "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
        "normalized_flip_rate": int(metric["introduced_flipped_faces"]) / initial.num_faces,
        "new_degenerate_faces": int(metric["new_degenerate_faces"]),
        "same_index_recovered_vertex_rms": float(
            np.sqrt(np.mean(np.sum((vertices - clean.vertices) ** 2, axis=1)))
        ),
        "improved": refined_cd < initial_cd,
        "worsened": refined_cd > initial_cd,
        "lambda": regularization,
    }
    if solver:
        result.update(solver)
    return result


def _archived_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chamfer": row["refined_chamfer"],
        "p2s": row["p2s"],
        "p2s_p95": row["p2s_p95"],
        "fscore": row["fscore"],
        "normal_consistency": row["normal_consistency"],
        "introduced_flipped_faces": row["introduced_flipped_faces"],
        "new_degenerate_faces": row["new_degenerate_faces"],
    }


def _selected_indices(length: int, count: int, index: int) -> list[int]:
    return [item for item in range(length) if item % count == index]


def sweep_shard(args: argparse.Namespace) -> None:
    split = "validation"
    target = args.output_dir / "shards" / f"sweep_{args.shard_index:02d}.json"
    if target.is_file() and not args.force:
        print(f"resume: {target}")
        return
    dataset, b_payload, e_payload, b_rows, e_rows, b_array, e_array, b_starts, e_starts = _inputs(args, split)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    lsmr_checks: list[dict[str, Any]] = []
    indices = _selected_indices(len(dataset), args.shard_count, args.shard_index)
    for progress, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(np.asarray(static["vertices"], dtype=np.float64), np.asarray(static["faces"], dtype=np.int64)).ensure_normals()
        clean = _clean_mesh(static)
        count = initial.num_vertices
        b_prediction = b_array[b_starts[index] : b_starts[index] + count]
        e_displacement = e_array[e_starts[index] : e_starts[index] + count]
        direct = initial.vertices + e_displacement
        for regularization in SWEEP_LAMBDAS:
            hybrid, solver = _pcg(b_prediction, direct, static, regularization, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{sample_id} lambda={regularization}: PCG failed: {solver}")
            mesh = Mesh(hybrid, initial.faces.copy()).ensure_normals()
            metric = _geometry_row(split, sample_id, ARM_H, mesh, clean, initial)
            item = _row(split, ARM_H, sample_id, index, hybrid, clean, initial, metric, solver, regularization)
            item["hybrid_to_e_vertex_rms"] = float(np.sqrt(np.mean(np.sum((hybrid - direct) ** 2, axis=1))))
            rows.append(item)
        if progress == 1:
            lap, lap_data = uniform_sparse_laplacian(initial.faces, count)
            components, labels = component_labels(lap_data)
            for regularization in (1e-4, 1e-2, 3.0):
                hybrid, _ = _pcg(b_prediction, direct, static, regularization, device)
                reference, audit = regularized_sparse_solve(
                    lap, b_prediction, direct, labels, components, regularization,
                    atol=1e-12, btol=1e-12, maxiter=100000,
                )
                lsmr_checks.append({
                    "sample_id": sample_id,
                    "lambda": regularization,
                    "lsmr_all_converged": audit["all_converged"],
                    "pcg_vs_lsmr_vertex_rms": float(np.sqrt(np.mean(np.sum((hybrid - reference) ** 2, axis=1)))),
                    "pcg_vs_lsmr_max_coordinate": float(np.max(np.abs(hybrid - reference))),
                })
        print(f"sweep {progress}/{len(indices)} {sample_id}", flush=True)
    _write_json(target, {
        "read_only": True,
        "gt_used_for_prediction_or_recovery": False,
        "models_retrained": False,
        "split": split,
        "arm_b_checkpoint": b_payload["checkpoint"],
        "arm_b_checkpoint_sha256": b_payload["checkpoint_sha256"],
        "arm_e_checkpoint": e_payload["checkpoint"],
        "arm_e_checkpoint_sha256": e_payload["checkpoint_sha256"],
        "same_input_contract": True,
        "anchor_change_only": True,
        "rows": rows,
        "lsmr_checks": lsmr_checks,
    })


def merge_sweep(args: argparse.Namespace) -> None:
    payloads = [_read(args.output_dir / "shards" / f"sweep_{i:02d}.json") for i in range(args.shard_count)]
    rows = [row for payload in payloads for row in payload["rows"]]
    if len(rows) != 50 * len(SWEEP_LAMBDAS):
        raise RuntimeError(f"Expected {50 * len(SWEEP_LAMBDAS)} validation rows, got {len(rows)}")
    aggregate = []
    for regularization in SWEEP_LAMBDAS:
        selected = [row for row in rows if float(row["lambda"]) == regularization]
        aggregate.append({
            "lambda": regularization,
            "samples": len(selected),
            "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
            "relative_chamfer_gain": float(np.mean([row["relative_chamfer_gain"] for row in selected])),
            "eta": float(np.mean([row["eta"] for row in selected])),
            "same_index_recovered_vertex_rms": float(np.mean([row["same_index_recovered_vertex_rms"] for row in selected])),
            "p2s": float(np.mean([row["p2s"] for row in selected])),
            "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
            "fscore": float(np.mean([row["fscore"] for row in selected])),
            "normal_consistency": float(np.mean([row["normal_consistency"] for row in selected])),
            "normalized_flip_rate": sum(row["introduced_flipped_faces"] for row in selected) / sum(row["faces"] for row in selected),
            "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in selected)),
            "improved": int(sum(row["improved"] for row in selected)),
            "worsened": int(sum(row["worsened"] for row in selected)),
            "pcg_iterations_mean": float(np.mean([row["pcg_iterations"] for row in selected])),
            "pcg_iterations_max": int(max(row["pcg_iterations"] for row in selected)),
            "pcg_relative_residual_max": float(max(row["pcg_relative_residual"] for row in selected)),
            "hybrid_to_e_vertex_rms": float(np.mean([row["hybrid_to_e_vertex_rms"] for row in selected])),
        })
    selected_cd = min(aggregate, key=lambda row: row["refined_chamfer"])["lambda"]
    selected_vrms = min(aggregate, key=lambda row: row["same_index_recovered_vertex_rms"])["lambda"]
    selected_p95 = min(aggregate, key=lambda row: row["p2s_p95"])["lambda"]
    audit = bool(
        all(payload["read_only"] and not payload["gt_used_for_prediction_or_recovery"] and payload["same_input_contract"] and payload["anchor_change_only"] for payload in payloads)
        and all(row["pcg_converged"] for row in rows)
        and len({payload["arm_b_checkpoint_sha256"] for payload in payloads}) == 1
        and len({payload["arm_e_checkpoint_sha256"] for payload in payloads}) == 1
    )
    summary = {
        "contract_audit": audit,
        "selection_split": "validation",
        "selection_metric": "mean refined Chamfer",
        "lambda_hybrid_best": selected_cd,
        "lambda_best_vertex_rms_diagnostic": selected_vrms,
        "lambda_best_p2s_p95_diagnostic": selected_p95,
        "pcg_dtype": "float64",
        "pcg_tolerance": PCG_TOLERANCE,
        "pcg_maximum_iterations": PCG_MAXIMUM_ITERATIONS,
        "arm_b_checkpoint": payloads[0]["arm_b_checkpoint"],
        "arm_b_checkpoint_sha256": payloads[0]["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint": payloads[0]["arm_e_checkpoint"],
        "arm_e_checkpoint_sha256": payloads[0]["arm_e_checkpoint_sha256"],
        "aggregate": aggregate,
        "lsmr_checks": [row for payload in payloads for row in payload["lsmr_checks"]],
    }
    _write_json(args.output_dir / "sweep_summary.json", summary)
    _write_csv(args.output_dir / "validation_lambda_sweep.csv", aggregate)
    _write_csv(args.output_dir / "validation_lambda_sweep_per_sample.csv", rows)
    print(json.dumps({"contract_audit": audit, "lambda_hybrid_best": selected_cd, "diagnostic_vrms": selected_vrms, "diagnostic_p95": selected_p95}, indent=2))


def _component_metrics(
    split: str,
    sample_id: str,
    labels: np.ndarray,
    displacements: Mapping[str, np.ndarray],
    gt: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    for arm, displacement in displacements.items():
        centered_error_parts = []
        for component in range(int(labels.max()) + 1):
            mask = labels == component
            gt_mean = gt[mask].mean(axis=0)
            pred_mean = displacement[mask].mean(axis=0)
            error = float(np.linalg.norm(pred_mean - gt_mean))
            component_rows.append({
                "split": split, "sample_id": sample_id, "arm": arm,
                "component": component, "component_vertices": int(mask.sum()),
                "component_translation_error": error,
            })
            centered_gt = gt[mask] - gt_mean
            centered_pred = displacement[mask] - pred_mean
            centered_error_parts.append(centered_pred - centered_gt)
        error_values = np.concatenate(centered_error_parts, axis=0)
        centered_rows.append({
            "split": split, "sample_id": sample_id, "arm": arm,
            "centered_vertex_rms": float(np.sqrt(np.mean(np.sum(error_values**2, axis=1)))),
        })
    return component_rows, centered_rows


def _spectral_row(
    split: str,
    sample_id: str,
    faces: np.ndarray,
    signals: Mapping[str, np.ndarray],
    order: int,
) -> list[dict[str, Any]]:
    names = list(signals)
    stacked = np.concatenate([signals[name] for name in names], axis=1)
    filtered, _ = spectral_band_components(stacked, faces, order=order)
    rows = []
    for signal_index, name in enumerate(names):
        column = slice(3 * signal_index, 3 * signal_index + 3)
        values = stacked[:, column]
        total = float(np.square(values).sum())
        item: dict[str, Any] = {
            "split": split, "sample_id": sample_id, "signal": name,
            "vertices": len(values), "total_energy": total,
        }
        for band in SPECTRAL_BANDS:
            energy = max(0.0, float(np.einsum("ij,ij->", values, filtered[band][:, column])))
            item[f"{band}_energy"] = energy
            item[f"{band}_fraction"] = energy / max(total, 1e-30)
        rows.append(item)
    return rows


def selected_shard(args: argparse.Namespace) -> None:
    if args.split not in {"validation", "test"}:
        raise ValueError("selected phase requires validation or test split")
    sweep = _read(args.output_dir / "sweep_summary.json")
    regularization = float(sweep["lambda_hybrid_best"])
    target = args.output_dir / "shards" / f"selected_{args.split}_{args.shard_index:02d}.json"
    if target.is_file() and not args.force:
        print(f"resume: {target}")
        return
    dataset, b_payload, e_payload, b_rows, e_rows, b_array, e_array, b_starts, e_starts = _inputs(args, args.split)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    components_all: list[dict[str, Any]] = []
    centered_all: list[dict[str, Any]] = []
    spectral_all: list[dict[str, Any]] = []
    indices = _selected_indices(len(dataset), args.shard_count, args.shard_index)
    for progress, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(np.asarray(static["vertices"], dtype=np.float64), np.asarray(static["faces"], dtype=np.int64)).ensure_normals()
        clean = _clean_mesh(static)
        count = initial.num_vertices
        b_prediction = b_array[b_starts[index] : b_starts[index] + count]
        e_displacement = e_array[e_starts[index] : e_starts[index] + count]
        direct = initial.vertices + e_displacement
        hybrid, solver = _pcg(b_prediction, direct, static, regularization, device)
        if not solver["pcg_converged"]:
            raise RuntimeError(f"{sample_id}: selected hybrid PCG failed")
        lap, lap_data = uniform_sparse_laplacian(initial.faces, count)
        component_count, labels = component_labels(lap_data)
        b_vertices, b_solve = regularized_sparse_solve(
            lap, b_prediction, initial.vertices, labels, component_count, 1e-2,
            atol=1e-12, btol=1e-12, maxiter=100000,
        )
        if not b_solve["all_converged"]:
            raise RuntimeError(f"{sample_id}: frozen B recovery reproduction failed")
        methods = {
            ARM_B: b_vertices,
            ARM_E: direct,
            ARM_H: hybrid,
        }
        initial_metric = _geometry_row(args.split, sample_id, "initial", initial, clean, initial)
        rows.append(_row(args.split, "initial", sample_id, index, initial.vertices, clean, initial, initial_metric))
        for arm, vertices in methods.items():
            if arm == ARM_B:
                metric = _archived_metric(b_rows[index])
                if not np.isclose(float(metric["chamfer"]), float(b_rows[index]["refined_chamfer"])):
                    raise RuntimeError("Invalid frozen B metric")
                solve_audit = {"recovery_reproduced_lsmr": True}
                arm_lambda = 1e-2
            elif arm == ARM_E:
                metric = _archived_metric(e_rows[index])
                solve_audit = {"direct_vertex_prediction": True}
                arm_lambda = None
            else:
                metric = _geometry_row(args.split, sample_id, arm, Mesh(vertices, initial.faces.copy()).ensure_normals(), clean, initial)
                solve_audit = solver
                arm_lambda = regularization
            item = _row(args.split, arm, sample_id, index, vertices, clean, initial, metric, solve_audit, arm_lambda)
            archived = b_rows[index] if arm == ARM_B else e_rows[index] if arm == ARM_E else None
            if archived is not None and not np.isclose(
                item["same_index_recovered_vertex_rms"], float(archived["same_index_recovered_vertex_rms"]), atol=2e-9, rtol=0
            ):
                raise RuntimeError(f"{sample_id}: {arm} frozen output reproduction failed")
            rows.append(item)
        gt_disp = clean.vertices - initial.vertices
        disp = {arm: vertices - initial.vertices for arm, vertices in methods.items()}
        component_rows, centered_rows = _component_metrics(args.split, sample_id, labels, disp, gt_disp)
        components_all.extend(component_rows)
        centered_all.extend(centered_rows)
        signals = {
            "gt_displacement": gt_disp,
            "b_error": disp[ARM_B] - gt_disp,
            "e_error": disp[ARM_E] - gt_disp,
            "hybrid_error": disp[ARM_H] - gt_disp,
            "hybrid_minus_b": disp[ARM_H] - disp[ARM_B],
            "hybrid_minus_e": disp[ARM_H] - disp[ARM_E],
        }
        spectral_all.extend(_spectral_row(args.split, sample_id, initial.faces, signals, args.chebyshev_order))
        print(f"selected {args.split} {progress}/{len(indices)} {sample_id}", flush=True)
    _write_json(target, {
        "read_only": True,
        "gt_used_for_prediction_or_recovery": False,
        "models_retrained": False,
        "lambda_selected_from_validation": regularization,
        "metric_protocol": METRIC_PROTOCOL,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "spectral_order": args.chebyshev_order,
        "arm_b_checkpoint_sha256": b_payload["checkpoint_sha256"],
        "arm_e_checkpoint_sha256": e_payload["checkpoint_sha256"],
        "rows": rows,
        "component_rows": components_all,
        "centered_rows": centered_all,
        "spectral_rows": spectral_all,
    })


def _aggregate(rows: Sequence[Mapping[str, Any]], split: str, arm: str, group: str = "all") -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and row["arm"] == arm]
    if not selected:
        raise RuntimeError(f"No rows for {split}/{arm}/{group}")
    return {
        "split": split,
        "group": group,
        "arm": arm,
        "samples": len(selected),
        "initial_chamfer": float(np.mean([row["initial_chamfer"] for row in selected])),
        "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
        "relative_chamfer_gain": float(np.mean([row["relative_chamfer_gain"] for row in selected])),
        "eta": float(np.mean([row["eta"] for row in selected])),
        "p2s": float(np.mean([row["p2s"] for row in selected])),
        "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
        "fscore": float(np.mean([row["fscore"] for row in selected])),
        "normal_consistency": float(np.mean([row["normal_consistency"] for row in selected])),
        "introduced_flipped_faces": int(sum(row["introduced_flipped_faces"] for row in selected)),
        "normalized_flip_rate": sum(row["introduced_flipped_faces"] for row in selected) / sum(row["faces"] for row in selected),
        "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in selected)),
        "same_index_recovered_vertex_rms": float(np.mean([row["same_index_recovered_vertex_rms"] for row in selected])),
        "improved": int(sum(row["improved"] for row in selected)),
        "worsened": int(sum(row["worsened"] for row in selected)),
    }


def _paired(rows: Sequence[Mapping[str, Any]], split: str, baseline: str, group: str = "all") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {row["sample_id"]: row for row in rows if row["split"] == split and row["arm"] == baseline}
    hybrid = {row["sample_id"]: row for row in rows if row["split"] == split and row["arm"] == ARM_H}
    if base.keys() != hybrid.keys():
        raise RuntimeError(f"{split}/{group}: paired IDs differ for {baseline}")
    keys = sorted(base)
    lower = ("refined_chamfer", "same_index_recovered_vertex_rms", "p2s_p95", "normalized_flip_rate")
    higher = ("fscore", "normal_consistency")
    wins = {
        "split": split,
        "group": group,
        "comparison": f"{ARM_H}_vs_{baseline}",
        "samples": len(keys),
    }
    for field in lower:
        wins[f"hybrid_better_{field}"] = int(sum(float(hybrid[key][field]) < float(base[key][field]) for key in keys))
    for field in higher:
        wins[f"hybrid_better_{field}"] = int(sum(float(hybrid[key][field]) > float(base[key][field]) for key in keys))
    rng = np.random.default_rng(7)
    statistics = []
    for field in ("refined_chamfer", "same_index_recovered_vertex_rms", "p2s_p95", "normal_consistency"):
        difference = np.asarray([float(hybrid[key][field]) - float(base[key][field]) for key in keys])
        selections = rng.integers(0, len(keys), size=(10000, len(keys)))
        bootstrap = difference[selections].mean(axis=1)
        statistics.append({
            "split": split,
            "group": group,
            "baseline": baseline,
            "quantity": f"{field}_hybrid_minus_{baseline}",
            "samples": len(keys),
            "mean_paired_difference": float(difference.mean()),
            "median_paired_difference": float(np.median(difference)),
            "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        })
    return wins, statistics


def merge_selected(args: argparse.Namespace) -> None:
    payloads = []
    for split in ("validation", "test"):
        payloads.extend(
            _read(args.output_dir / "shards" / f"selected_{split}_{index:02d}.json")
            for index in range(args.shard_count)
        )
    rows = [row for payload in payloads for row in payload["rows"]]
    component_rows = [row for payload in payloads for row in payload["component_rows"]]
    centered_rows = [row for payload in payloads for row in payload["centered_rows"]]
    spectral_rows = [row for payload in payloads for row in payload["spectral_rows"]]
    if len(rows) != 2 * 50 * 4:
        raise RuntimeError(f"Expected 400 matched rows, found {len(rows)}")
    aggregate = [_aggregate(rows, split, arm) for split in ("validation", "test") for arm in ("initial", ARM_B, ARM_E, ARM_H)]
    paired_wins: list[dict[str, Any]] = []
    paired_statistics: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for baseline in (ARM_B, ARM_E):
            wins, statistics = _paired(rows, split, baseline)
            paired_wins.append(wins)
            paired_statistics.extend(statistics)
    recipe_aggregate: list[dict[str, Any]] = []
    recipe_wins: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for group, variants in [(variant, {variant}) for variant in VARIANTS] + list(ANALYSIS_GROUPS.items()):
            selected = [row for row in rows if row["split"] == split and row["variant"] in variants]
            for arm in (ARM_B, ARM_E, ARM_H):
                recipe_aggregate.append(_aggregate(selected, split, arm, group))
            for baseline in (ARM_B, ARM_E):
                wins, _ = _paired(selected, split, baseline, group)
                recipe_wins.append(wins)
    spectral_aggregate = []
    for split in ("validation", "test"):
        for signal in ("gt_displacement", "b_error", "e_error", "hybrid_error", "hybrid_minus_b", "hybrid_minus_e"):
            selected = [row for row in spectral_rows if row["split"] == split and row["signal"] == signal]
            total = sum(row["total_energy"] for row in selected)
            item = {
                "split": split, "signal": signal, "samples": len(selected),
                "vertices": int(sum(row["vertices"] for row in selected)),
                "total_energy": total,
                "mean_energy_per_vertex": total / sum(row["vertices"] for row in selected),
            }
            for band in SPECTRAL_BANDS:
                energy = sum(row[f"{band}_energy"] for row in selected)
                item[f"{band}_energy"] = energy
                item[f"{band}_fraction"] = energy / max(total, 1e-30)
            spectral_aggregate.append(item)
    component_aggregate = []
    for split in ("validation", "test"):
        for arm in (ARM_B, ARM_E, ARM_H):
            selected = [row["component_translation_error"] for row in component_rows if row["split"] == split and row["arm"] == arm]
            centered = [row["centered_vertex_rms"] for row in centered_rows if row["split"] == split and row["arm"] == arm]
            values = np.asarray(selected)
            component_aggregate.append({
                "split": split, "arm": arm, "components": len(values),
                "component_translation_error_mean": float(values.mean()),
                "component_translation_error_rms": float(np.sqrt(np.mean(values**2))),
                "component_translation_error_median": float(np.median(values)),
                "component_translation_error_p95": float(np.quantile(values, 0.95)),
                "centered_vertex_rms_mean": float(np.mean(centered)),
            })
    sweep = _read(args.output_dir / "sweep_summary.json")
    audit = bool(
        sweep["contract_audit"]
        and all(payload["read_only"] and not payload["gt_used_for_prediction_or_recovery"] for payload in payloads)
        and all(row.get("pcg_converged", True) for row in rows)
        and len({payload["arm_b_checkpoint_sha256"] for payload in payloads}) == 1
        and len({payload["arm_e_checkpoint_sha256"] for payload in payloads}) == 1
        and {payload["arm_b_checkpoint_sha256"] for payload in payloads}
        == {sweep["arm_b_checkpoint_sha256"]}
        and {payload["arm_e_checkpoint_sha256"] for payload in payloads}
        == {sweep["arm_e_checkpoint_sha256"]}
    )
    summary = {
        "contract_audit": audit,
        "read_only": True,
        "models_retrained": False,
        "gt_used_for_prediction_or_recovery": False,
        "same_archived_input_manifest_and_sample_order": True,
        "only_recovery_anchor_target_changed": True,
        "arm_b_checkpoint": sweep["arm_b_checkpoint"],
        "arm_b_checkpoint_sha256": sweep["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint": sweep["arm_e_checkpoint"],
        "arm_e_checkpoint_sha256": sweep["arm_e_checkpoint_sha256"],
        "lambda_hybrid_best": sweep["lambda_hybrid_best"],
        "lambda_selection_split": "validation",
        "lambda_selection_metric": "mean refined Chamfer",
        "metric_protocol": METRIC_PROTOCOL,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "aggregate": aggregate,
        "paired_wins": paired_wins,
        "paired_statistics": paired_statistics,
        "recipe_aggregate": recipe_aggregate,
        "recipe_paired_wins": recipe_wins,
        "spectral_aggregate": spectral_aggregate,
        "component_aggregate": component_aggregate,
    }
    _write_json(args.output_dir / "matched_summary.json", summary)
    _write_csv(args.output_dir / "matched_aggregate.csv", aggregate)
    _write_csv(args.output_dir / "matched_per_sample.csv", rows)
    _write_csv(args.output_dir / "paired_wins.csv", paired_wins)
    _write_csv(args.output_dir / "paired_bootstrap.csv", paired_statistics)
    _write_csv(args.output_dir / "recipe_aggregate.csv", recipe_aggregate)
    _write_csv(args.output_dir / "recipe_paired_wins.csv", recipe_wins)
    _write_csv(args.output_dir / "spectral_per_sample.csv", spectral_rows)
    _write_csv(args.output_dir / "spectral_aggregate.csv", spectral_aggregate)
    _write_csv(args.output_dir / "component_translation_per_component.csv", component_rows)
    _write_csv(args.output_dir / "centered_deformation_per_sample.csv", centered_rows)
    _write_csv(args.output_dir / "component_aggregate.csv", component_aggregate)
    print(json.dumps({"contract_audit": audit, "lambda": sweep["lambda_hybrid_best"], "rows": len(rows)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("sweep", "merge-sweep", "selected", "merge-selected"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index")
    if args.phase == "sweep":
        sweep_shard(args)
    elif args.phase == "merge-sweep":
        merge_sweep(args)
    elif args.phase == "selected":
        selected_shard(args)
    else:
        merge_selected(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

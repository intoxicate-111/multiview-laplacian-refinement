#!/usr/bin/env python3
from __future__ import annotations

"""Read-only matched-domain mechanism analysis for frozen B/E versus joint Hybrid."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats as scipy_stats

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import (
    _component_metrics,
    _row,
    _spectral_row,
)
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import (
    SPECTRAL_BANDS,
    SPECTRAL_PROTOCOL,
    _starts,
    spectral_band_components,
)
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


METHODS = (
    "Pretrained_B",
    "Pretrained_E",
    "Frozen_BE",
    "Joint_Lap",
    "Joint_Direct",
    "Joint_Hybrid",
)
PAIRS = {
    "B_E": ("Pretrained_B", "Pretrained_E"),
    "Joint_Lap_Direct": ("Joint_Lap", "Joint_Direct"),
}
LAMBDA_B = 1e-2
LAMBDA_H = 3e-2
PCG_TOLERANCE = 1e-8
PCG_MAXIMUM_ITERATIONS = 2048
FROZEN_BE_TOLERANCE = 1e-4
EPSILON = 1e-30
EXPECTED_JOINT_SHA256 = "9af46b5c3203415aa06c3967fe2f5d36bd1cab389f036c481e147e874e5dab62"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive(report: Path, arm: str, split: str) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    payload = _read(report / "shards" / f"{arm}.json")
    rows = [dict(row) for row in payload["rows"] if row["split"] == split]
    array = np.load(report / "shards" / f"{arm}_prediction_arrays.npz")[
        f"{split}_prediction"
    ].astype(np.float64)
    return rows, array, _starts(rows, array)


def _pcg(
    delta: np.ndarray,
    anchor: np.ndarray,
    static: Mapping[str, Any],
    device: torch.device,
    *,
    tolerance: float = PCG_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    with torch.no_grad():
        recovered, audit = recovery_forward_audit(
            torch.as_tensor(delta, dtype=torch.float64, device=device),
            torch.as_tensor(anchor, dtype=torch.float64, device=device),
            torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
            torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device),
            regularization=LAMBDA_H,
            maximum_iterations=PCG_MAXIMUM_ITERATIONS,
            tolerance=tolerance,
        )
    if not audit.converged:
        raise RuntimeError(f"PCG failed: {audit}")
    return recovered.detach().cpu().numpy(), {
        "pcg_iterations": int(audit.iterations),
        "pcg_relative_residual": float(audit.relative_residual),
        "pcg_converged": bool(audit.converged),
        "pcg_tolerance": tolerance,
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.reshape(-1), right.reshape(-1)
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPSILON))


def _latent_row(
    split: str,
    sample_id: str,
    pair: str,
    delta: np.ndarray,
    mapped_direct: np.ndarray,
) -> dict[str, Any]:
    difference = delta - mapped_direct
    delta_norm = float(np.linalg.norm(delta))
    direct_norm = float(np.linalg.norm(mapped_direct))
    return {
        "split": split,
        "sample_id": sample_id,
        "pair": pair,
        "redundancy_rms": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_discrepancy": float(np.linalg.norm(difference) / max(delta_norm, EPSILON)),
        "cosine": _cosine(delta, mapped_direct),
        "norm_ratio": delta_norm / max(direct_norm, EPSILON),
    }


def _lap_semantic_row(
    split: str,
    sample_id: str,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    error = np.linalg.norm(prediction - target, axis=1)
    magnitude = np.linalg.norm(target, axis=1)
    order = np.argsort(magnitude, kind="stable")
    top10 = order[-max(1, math.ceil(0.10 * len(order))) :]
    top1 = order[-max(1, math.ceil(0.01 * len(order))) :]
    return {
        "split": split,
        "sample_id": sample_id,
        "method": method,
        "raw_epe": float(error.mean()),
        "raw_rms": float(np.sqrt(np.mean(np.square(error)))),
        "raw_cosine": _cosine(prediction, target),
        "top10_epe": float(error[top10].mean()),
        "top1_epe": float(error[top1].mean()),
    }


def _position_semantic_row(
    split: str,
    sample_id: str,
    method: str,
    vertices: np.ndarray,
    clean: np.ndarray,
) -> dict[str, Any]:
    error = np.linalg.norm(vertices - clean, axis=1)
    return {
        "split": split,
        "sample_id": sample_id,
        "method": method,
        "vertex_rms": float(np.sqrt(np.mean(np.square(error)))),
        "vertex_error_mean": float(error.mean()),
        "vertex_error_p95": float(np.quantile(error, 0.95)),
    }


def _energy_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_energy = np.sum(np.square(left), axis=1)
    right_energy = np.sum(np.square(right), axis=1)
    denominator = min(float(left_energy.sum()), float(right_energy.sum()))
    return float(np.minimum(left_energy, right_energy).sum() / max(denominator, EPSILON))


def _error_pair_rows(
    split: str,
    sample_id: str,
    faces: np.ndarray,
    errors: Mapping[str, np.ndarray],
    order: int,
) -> list[dict[str, Any]]:
    names = list(errors)
    stacked = np.concatenate([errors[name] for name in names], axis=1)
    filtered, _ = spectral_band_components(stacked, faces, order=order)
    columns = {name: slice(3 * index, 3 * index + 3) for index, name in enumerate(names)}
    rows: list[dict[str, Any]] = []
    for pair, (left_name, right_name) in PAIRS.items():
        left, right = errors[left_name], errors[right_name]
        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "pair": pair,
                "band": "global",
                "error_cosine": _cosine(left, right),
                "energy_overlap": _energy_overlap(left, right),
            }
        )
        for band in SPECTRAL_BANDS:
            left_band = filtered[band][:, columns[left_name]]
            right_band = filtered[band][:, columns[right_name]]
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "pair": pair,
                    "band": band,
                    "error_cosine": _cosine(left_band, right_band),
                    "energy_overlap": _energy_overlap(left_band, right_band),
                }
            )
    return rows


def _component_complementarity(
    component_rows: Sequence[Mapping[str, Any]], split: str, sample_id: str, left: str, right: str
) -> float:
    a = {
        int(row["component"]): float(row["component_translation_error"])
        for row in component_rows
        if row["split"] == split and row["sample_id"] == sample_id and row["arm"] == left
    }
    b = {
        int(row["component"]): float(row["component_translation_error"])
        for row in component_rows
        if row["split"] == split and row["sample_id"] == sample_id and row["arm"] == right
    }
    if a.keys() != b.keys():
        raise RuntimeError("Component IDs differ between paired methods.")
    return float(np.mean([abs(a[key] - b[key]) for key in a]))


def shard(args: argparse.Namespace) -> None:
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    b_rows, b_array, b_starts = _archive(args.arm_b_report.resolve(), "B_lap_plus_refine", args.split)
    e_rows, e_array, e_starts = _archive(args.arm_e_report.resolve(), "E_direct_vertex_residual", args.split)
    expected = list(dataset.sample_ids)
    if [row["sample_id"] for row in b_rows] != expected or [row["sample_id"] for row in e_rows] != expected:
        raise RuntimeError("B/E archive IDs do not match the manifest.")
    device = torch.device(args.device)
    joint_run = args.joint_run.resolve()
    config_payload = _read(joint_run / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    joint_checkpoint = joint_run / "checkpoint_best.pt"
    joint_sha256 = _sha256(joint_checkpoint)
    if joint_sha256 != EXPECTED_JOINT_SHA256:
        raise RuntimeError(f"Joint checkpoint SHA mismatch: {joint_sha256}")
    model = _build_model(config, None, False).to(device)
    load_checkpoint(joint_checkpoint, model, map_location=device)
    model.eval()
    if not model.hybrid_direct_head_enabled:
        raise RuntimeError("Expected a shared-backbone joint hybrid model.")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    geometry_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    lap_semantic_rows: list[dict[str, Any]] = []
    position_semantic_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    error_pair_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    indices = [index for index in range(len(dataset)) if index % args.shard_count == args.shard_index]
    for progress, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        count = len(vertices)
        delta_b = b_array[b_starts[index] : b_starts[index] + count]
        disp_e = e_array[e_starts[index] : e_starts[index] + count]
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            joint_output = model(conditioned)
        joint_direct = joint_output.direct_vertex_displacement_prediction
        if joint_direct is None:
            raise RuntimeError("Joint model did not return a direct displacement.")
        delta_joint = joint_output.predicted_laplacian.detach().double().cpu().numpy()
        disp_joint = joint_direct.detach().double().cpu().numpy()
        if delta_joint.shape != delta_b.shape or disp_joint.shape != disp_e.shape:
            raise RuntimeError(f"{sample_id}: latent shape mismatch")
        lap, lap_data = uniform_sparse_laplacian(faces, count)
        component_count, labels = component_labels(lap_data)
        v_b, b_audit = regularized_sparse_solve(
            lap, delta_b, vertices, labels, component_count, LAMBDA_B,
            atol=1e-12, btol=1e-12, maxiter=100000,
        )
        v_jlap, jlap_audit = regularized_sparse_solve(
            lap, delta_joint, vertices, labels, component_count, LAMBDA_B,
            atol=1e-12, btol=1e-12, maxiter=100000,
        )
        if not b_audit["all_converged"] or not jlap_audit["all_converged"]:
            raise RuntimeError(f"{sample_id}: standalone recovery failed")
        v_e = vertices + disp_e
        v_jdirect = vertices + disp_joint
        v_frozen, frozen_audit = _pcg(
            delta_b, v_e, static, device, tolerance=FROZEN_BE_TOLERANCE
        )
        v_jhybrid, joint_audit = _pcg(delta_joint, v_jdirect, static, device)
        methods = {
            "Pretrained_B": v_b,
            "Pretrained_E": v_e,
            "Frozen_BE": v_frozen,
            "Joint_Lap": v_jlap,
            "Joint_Direct": v_jdirect,
            "Joint_Hybrid": v_jhybrid,
        }
        initial_metric = _geometry_row(args.split, sample_id, "initial", initial, clean, initial)
        geometry_rows.append(_row(args.split, "Initial", sample_id, index, vertices, clean, initial, initial_metric))
        for method, method_vertices in methods.items():
            metric = _geometry_row(
                args.split, sample_id, method,
                Mesh(method_vertices, faces.copy()).ensure_normals(), clean, initial,
            )
            solve = frozen_audit if method == "Frozen_BE" else joint_audit if method == "Joint_Hybrid" else None
            regularization = LAMBDA_B if method in {"Pretrained_B", "Joint_Lap"} else LAMBDA_H if method in {"Frozen_BE", "Joint_Hybrid"} else None
            geometry_rows.append(
                _row(args.split, method, sample_id, index, method_vertices, clean, initial, metric, solve, regularization)
            )
        delta_gt = lap @ clean.vertices
        mapped_e = lap @ v_e
        mapped_joint = lap @ v_jdirect
        latent_rows.extend(
            (
                _latent_row(args.split, sample_id, "B_E", delta_b, mapped_e),
                _latent_row(args.split, sample_id, "Joint_Lap_Direct", delta_joint, mapped_joint),
            )
        )
        lap_semantic_rows.extend(
            (
                _lap_semantic_row(args.split, sample_id, "Pretrained_B", delta_b, delta_gt),
                _lap_semantic_row(args.split, sample_id, "Joint_Lap", delta_joint, delta_gt),
            )
        )
        position_semantic_rows.extend(
            (
                _position_semantic_row(args.split, sample_id, "Pretrained_E", v_e, clean.vertices),
                _position_semantic_row(args.split, sample_id, "Joint_Direct", v_jdirect, clean.vertices),
            )
        )
        gt_displacement = clean.vertices - vertices
        displacements = {method: value - vertices for method, value in methods.items()}
        sample_components, sample_centered = _component_metrics(
            args.split, sample_id, labels, displacements, gt_displacement
        )
        component_rows.extend(sample_components)
        centered_rows.extend(sample_centered)
        errors = {method: value - clean.vertices for method, value in methods.items()}
        spectral_rows.extend(
            _spectral_row(args.split, sample_id, faces, {f"{method}_error": error for method, error in errors.items()}, args.chebyshev_order)
        )
        sample_error_pairs = _error_pair_rows(args.split, sample_id, faces, errors, args.chebyshev_order)
        error_pair_rows.extend(sample_error_pairs)
        geometry_map = {
            row["arm"]: row
            for row in geometry_rows
            if row["split"] == args.split and row["sample_id"] == sample_id
        }
        error_pair_map = {(row["pair"], row["band"]): row for row in sample_error_pairs}
        latent_map = {
            row["pair"]: row
            for row in latent_rows
            if row["split"] == args.split and row["sample_id"] == sample_id
        }
        for pair, (left, right) in PAIRS.items():
            hybrid = "Frozen_BE" if pair == "B_E" else "Joint_Hybrid"
            fusion_rows.append(
                {
                    "split": args.split,
                    "sample_id": sample_id,
                    "pair": pair,
                    "fusion_gain": min(float(geometry_map[left]["refined_chamfer"]), float(geometry_map[right]["refined_chamfer"])) - float(geometry_map[hybrid]["refined_chamfer"]),
                    "global_error_cosine": float(error_pair_map[(pair, "global")]["error_cosine"]),
                    "global_energy_overlap": float(error_pair_map[(pair, "global")]["energy_overlap"]),
                    "relative_redundancy": float(latent_map[pair]["relative_discrepancy"]),
                    "component_translation_complementarity": _component_complementarity(sample_components, args.split, sample_id, left, right),
                }
            )
        print(f"{args.split} shard={args.shard_index} {progress}/{len(indices)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target = args.output_dir / "shards" / f"mechanism_{args.split}_{args.shard_index:02d}.json"
    _write_json(
        target,
        {
            "read_only": True,
            "split": args.split,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "metric_protocol": METRIC_PROTOCOL,
            "spectral_protocol": SPECTRAL_PROTOCOL,
            "joint_checkpoint": str(joint_checkpoint),
            "joint_checkpoint_sha256": joint_sha256,
            "energy_overlap_definition": "sum_v min(||e1_v||^2,||e2_v||^2) / min(total_energy_1,total_energy_2)",
            "geometry_rows": geometry_rows,
            "latent_rows": latent_rows,
            "lap_semantic_rows": lap_semantic_rows,
            "position_semantic_rows": position_semantic_rows,
            "spectral_rows": spectral_rows,
            "error_pair_rows": error_pair_rows,
            "component_rows": component_rows,
            "centered_rows": centered_rows,
            "fusion_rows": fusion_rows,
        },
    )


def _aggregate_geometry(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for method in ("Initial", *METHODS):
            selected = [row for row in rows if row["split"] == split and row["arm"] == method]
            if not selected:
                continue
            result.append(
                {
                    "split": split,
                    "method": method,
                    "samples": len(selected),
                    "chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
                    "vertex_rms": float(np.mean([row["same_index_recovered_vertex_rms"] for row in selected])),
                    "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
                    "fscore": float(np.mean([row["fscore"] for row in selected])),
                    "normal": float(np.mean([row["normal_consistency"] for row in selected])),
                    "flips": int(sum(row["introduced_flipped_faces"] for row in selected)),
                    "flip_rate": float(sum(row["introduced_flipped_faces"] for row in selected) / sum(row["faces"] for row in selected)),
                    "new_degenerates": int(sum(row["new_degenerate_faces"] for row in selected)),
                    "improved": int(sum(row["improved"] for row in selected)),
                    "worsened": int(sum(row["worsened"] for row in selected)),
                }
            )
    return result


def _aggregate_scalar(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], value_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    result = []
    for key, selected in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output = {field: value for field, value in zip(group_fields, key)}
        output["samples"] = len(selected)
        for field in value_fields:
            values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
            output[f"{field}_mean"] = float(values.mean())
            output[f"{field}_median"] = float(np.median(values))
            output[f"{field}_p25"] = float(np.quantile(values, 0.25))
            output[f"{field}_p75"] = float(np.quantile(values, 0.75))
        result.append(output)
    return result


def _bootstrap_difference(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, Any]:
    keys = sorted(left)
    if keys != sorted(right):
        raise RuntimeError("Paired sample IDs differ.")
    difference = np.asarray([float(left[key][field]) - float(right[key][field]) for key in keys])
    rng = np.random.default_rng(7)
    choices = rng.integers(0, len(keys), size=(10000, len(keys)))
    bootstrap = difference[choices].mean(axis=1)
    return {
        "field": field,
        "samples": len(keys),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "left_wins": int(sum(value < 0 for value in difference)),
        "right_wins": int(sum(value > 0 for value in difference)),
        "ties": int(sum(value == 0 for value in difference)),
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def _correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    predictors = (
        "global_error_cosine",
        "global_energy_overlap",
        "relative_redundancy",
        "component_translation_complementarity",
    )
    for split in ("validation", "test"):
        for pair in PAIRS:
            selected = [row for row in rows if row["split"] == split and row["pair"] == pair]
            y = np.asarray([float(row["fusion_gain"]) for row in selected])
            for field in predictors:
                x = np.asarray([float(row[field]) for row in selected])
                pearson = scipy_stats.pearsonr(x, y)
                spearman = scipy_stats.spearmanr(x, y)
                result.append(
                    {
                        "split": split,
                        "pair": pair,
                        "predictor": field,
                        "n": len(selected),
                        "pearson": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                    }
                )
    return result


def merge(args: argparse.Namespace) -> None:
    payloads = [
        _read(args.output_dir / "shards" / f"mechanism_{split}_{index:02d}.json")
        for split in ("validation", "test")
        for index in range(args.shard_count)
    ]
    keys = (
        "geometry_rows", "latent_rows", "lap_semantic_rows", "position_semantic_rows",
        "spectral_rows", "error_pair_rows", "component_rows", "centered_rows", "fusion_rows",
    )
    merged = {key: [row for payload in payloads for row in payload[key]] for key in keys}
    geometry_aggregate = _aggregate_geometry(merged["geometry_rows"])
    latent_aggregate = _aggregate_scalar(
        merged["latent_rows"], ("split", "pair"),
        ("redundancy_rms", "relative_discrepancy", "cosine", "norm_ratio"),
    )
    lap_semantic_aggregate = _aggregate_scalar(
        merged["lap_semantic_rows"], ("split", "method"),
        ("raw_epe", "raw_rms", "raw_cosine", "top10_epe", "top1_epe"),
    )
    position_semantic_aggregate = _aggregate_scalar(
        merged["position_semantic_rows"], ("split", "method"),
        ("vertex_rms", "vertex_error_mean", "vertex_error_p95"),
    )
    spectral_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for method in METHODS:
            signal = f"{method}_error"
            selected = [row for row in merged["spectral_rows"] if row["split"] == split and row["signal"] == signal]
            total = float(sum(row["total_energy"] for row in selected))
            item: dict[str, Any] = {"split": split, "method": method, "samples": len(selected), "total_energy": total}
            for band in SPECTRAL_BANDS:
                energy = float(sum(row[f"{band}_energy"] for row in selected))
                item[f"{band}_energy"] = energy
                item[f"{band}_fraction"] = energy / max(total, EPSILON)
            spectral_aggregate.append(item)
    error_pair_aggregate = _aggregate_scalar(
        merged["error_pair_rows"], ("split", "pair", "band"),
        ("error_cosine", "energy_overlap"),
    )
    component_aggregate: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for method in METHODS:
            component = np.asarray([
                row["component_translation_error"] for row in merged["component_rows"]
                if row["split"] == split and row["arm"] == method
            ], dtype=np.float64)
            centered = np.asarray([
                row["centered_vertex_rms"] for row in merged["centered_rows"]
                if row["split"] == split and row["arm"] == method
            ], dtype=np.float64)
            component_aggregate.append(
                {
                    "split": split,
                    "method": method,
                    "components": len(component),
                    "component_translation_rms": float(np.sqrt(np.mean(np.square(component)))),
                    "component_translation_mean": float(component.mean()),
                    "centered_deformation_vrms": float(centered.mean()),
                }
            )
    fusion_aggregate = _aggregate_scalar(
        merged["fusion_rows"], ("split", "pair"), ("fusion_gain",)
    )
    for item in fusion_aggregate:
        selected = [row for row in merged["fusion_rows"] if row["split"] == item["split"] and row["pair"] == item["pair"]]
        item["positive_count"] = int(sum(float(row["fusion_gain"]) > 0 for row in selected))
        item["negative_count"] = int(sum(float(row["fusion_gain"]) < 0 for row in selected))
        item["maximum"] = float(max(float(row["fusion_gain"]) for row in selected))
    correlations = _correlations(merged["fusion_rows"])
    paired: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        geo = {
            method: {row["sample_id"]: row for row in merged["geometry_rows"] if row["split"] == split and row["arm"] == method}
            for method in METHODS
        }
        for left_name, right_name in (("Pretrained_B", "Joint_Lap"), ("Pretrained_E", "Joint_Direct"), ("Frozen_BE", "Joint_Hybrid")):
            for field in ("refined_chamfer", "same_index_recovered_vertex_rms", "p2s_p95"):
                row = _bootstrap_difference(geo[left_name], geo[right_name], field)
                row.update({"split": split, "left": left_name, "right": right_name})
                paired.append(row)
        lap = {
            method: {
                row["sample_id"]: row
                for row in merged["lap_semantic_rows"]
                if row["split"] == split and row["method"] == method
            }
            for method in ("Pretrained_B", "Joint_Lap")
        }
        direct = {
            method: {
                row["sample_id"]: row
                for row in merged["position_semantic_rows"]
                if row["split"] == split and row["method"] == method
            }
            for method in ("Pretrained_E", "Joint_Direct")
        }
        for field in ("raw_epe", "raw_rms", "top10_epe", "top1_epe"):
            row = _bootstrap_difference(lap["Pretrained_B"], lap["Joint_Lap"], field)
            row.update({"split": split, "left": "Pretrained_B", "right": "Joint_Lap"})
            paired.append(row)
        for field in ("vertex_rms", "vertex_error_mean", "vertex_error_p95"):
            row = _bootstrap_difference(direct["Pretrained_E"], direct["Joint_Direct"], field)
            row.update({"split": split, "left": "Pretrained_E", "right": "Joint_Direct"})
            paired.append(row)
    checks = {
        "all_read_only": all(payload["read_only"] for payload in payloads),
        "all_shards_present": len(payloads) == 2 * args.shard_count,
        "validation_test_50": all(
            sum(row["split"] == split and row["arm"] == "Joint_Hybrid" for row in merged["geometry_rows"]) == 50
            for split in ("validation", "test")
        ),
        "pcg_converged": all(row.get("pcg_converged", True) for row in merged["geometry_rows"]),
        "joint_checkpoint_identity": len({payload["joint_checkpoint_sha256"] for payload in payloads}) == 1
        and payloads[0]["joint_checkpoint_sha256"] == EXPECTED_JOINT_SHA256,
    }
    summary = {
        "contract_audit": all(checks.values()),
        "contract_checks": checks,
        "metric_protocol": METRIC_PROTOCOL,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "joint_checkpoint": payloads[0]["joint_checkpoint"],
        "joint_checkpoint_sha256": payloads[0]["joint_checkpoint_sha256"],
        "energy_overlap_definition": payloads[0]["energy_overlap_definition"],
        "geometry_aggregate": geometry_aggregate,
        "latent_aggregate": latent_aggregate,
        "lap_semantic_aggregate": lap_semantic_aggregate,
        "position_semantic_aggregate": position_semantic_aggregate,
        "spectral_aggregate": spectral_aggregate,
        "error_pair_aggregate": error_pair_aggregate,
        "component_aggregate": component_aggregate,
        "fusion_aggregate": fusion_aggregate,
        "fusion_correlations": correlations,
        "paired_specialist_statistics": paired,
    }
    _write_json(args.output_dir / "mechanism_summary.json", summary)
    for name, rows in merged.items():
        _write_csv(args.output_dir / f"{name}.csv", rows)
    for name, rows in (
        ("geometry_aggregate", geometry_aggregate),
        ("latent_aggregate", latent_aggregate),
        ("lap_semantic_aggregate", lap_semantic_aggregate),
        ("position_semantic_aggregate", position_semantic_aggregate),
        ("spectral_aggregate", spectral_aggregate),
        ("error_pair_aggregate", error_pair_aggregate),
        ("component_aggregate", component_aggregate),
        ("fusion_aggregate", fusion_aggregate),
        ("fusion_correlations", correlations),
        ("paired_specialist_statistics", paired),
    ):
        _write_csv(args.output_dir / f"{name}.csv", rows)
    print(json.dumps({"contract_audit": summary["contract_audit"], "output": str(args.output_dir)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("shard", "merge"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--arm-b-report", type=Path)
    parser.add_argument("--arm-e-report", type=Path)
    parser.add_argument("--joint-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.phase == "shard":
        for field in ("manifest", "arm_b_report", "arm_e_report", "joint_run"):
            if getattr(args, field) is None:
                parser.error(f"--{field.replace('_', '-')} is required for shard")
        shard(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

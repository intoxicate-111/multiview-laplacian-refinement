#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate Arm I and append the singular lambda=0 limit to the fixed sweep."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_multitopology_rawlap import raw_gt_magnitude_metrics
from evaluate_sofa50_recovery_aware_ablation import (
    METRIC_PROTOCOL,
    _infer_recovery_arm,
    _load_spec,
    _read,
    _runtime_diagnostic_summary,
    _sha256,
    _write_csv,
    _write_json,
)
from mlr.data import Mesh
from mlr.learned_laplacian.hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
    hard_anchor_sparse_recovery_lsmr,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM = "I_lap_plus_refine_lambda0_hard_anchor"
RUN_NAME = "sofa50_v2_sparse_recovery_arm_i_lambda0_hard_anchor_20k_seed7"
FIXED_ARMS = (
    "D_lap_plus_refine_lambda1e-4",
    "C_lap_plus_refine_lambda1e-3",
    "B_lap_plus_refine",
    "E_lap_plus_refine_lambda1e-1",
    "F_lap_plus_refine_lambda1",
)
FIXED_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RECIPES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")
MILD = {"A1", "B1", "C1", "C3", "D1"}


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _recipe(sample_id: str) -> str:
    recipe = sample_id.rsplit("__", 1)[-1]
    if recipe not in RECIPES:
        raise ValueError(f"Unknown coarse recipe in {sample_id}.")
    return recipe


def evaluate(args: argparse.Namespace) -> None:
    run = args.runs_root.resolve() / RUN_NAME
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spec = _load_spec(run, device)
    loss = spec["config"]["training"]["recovery_aware_geometry_loss"]
    recovery = spec["config"]["recovery"]
    if (
        loss.get("solver") != "hard_anchor_lambda0"
        or float(loss["lambda"]) != 0.0
        or float(loss["beta"]) != 1e-2
        or recovery.get("solver")
        != "reduced_hard_anchor_undamped_normal_sparse_lu_float64"
    ):
        raise RuntimeError("Arm-I checkpoint/config does not satisfy the hard-anchor contract.")
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {}
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        arrays[f"{split}_prediction"] = []
        arrays[f"{split}_target"] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            values = _infer_recovery_arm(dataset, index, spec, device)
            prediction = values["prediction_raw"].numpy().astype(np.float64)
            target = values["target_raw"].numpy().astype(np.float64)
            valid = values["valid"].numpy().astype(bool)
            arrays[f"{split}_prediction"].append(prediction[valid])
            arrays[f"{split}_target"].append(target[valid])
            prediction_metrics = raw_gt_magnitude_metrics(
                values["prediction_raw"],
                values["target_raw"],
                torch.ones_like(values["recovery_weight"]),
                values["valid"],
            )
            initial = Mesh(
                torch.as_tensor(static["vertices"]).numpy(),
                torch.as_tensor(static["faces"]).numpy().astype(np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            laplacian, lap_data = uniform_sparse_laplacian(
                initial.faces, initial.num_vertices
            )
            component_count, labels = component_labels(lap_data)
            anchors = deterministic_component_anchor_indices(
                torch.as_tensor(static["edge_index"]), initial.num_vertices
            ).numpy()
            if len(anchors) != int(component_count):
                raise RuntimeError(f"Anchor/component mismatch for {sample_id}.")
            recovered, solver = hard_anchor_sparse_recovery_lsmr(
                laplacian,
                prediction,
                initial.vertices,
                anchors,
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            if not bool(solver["all_converged"]):
                raise RuntimeError(
                    f"Reduced LSMR did not converge for {sample_id}: {solver['axes']}"
                )
            initial_geometry = _geometry_row(
                "v2_strong_smoothing", sample_id, "initial", initial, clean, initial
            )
            clean_geometry = _geometry_row(
                "v2_strong_smoothing", sample_id, "clean", clean, clean, initial
            )
            refined_geometry = _geometry_row(
                "v2_strong_smoothing",
                sample_id,
                ARM,
                Mesh(recovered, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            initial_cd = float(initial_geometry["chamfer"])
            clean_cd = float(clean_geometry["chamfer"])
            refined_cd = float(refined_geometry["chamfer"])
            displacement = recovered - initial.vertices
            displacement_norm = np.linalg.norm(displacement, axis=1)
            component_shifts = np.stack(
                [displacement[labels == component].mean(axis=0) for component in range(component_count)]
            )
            rows.append(
                {
                    "arm": ARM,
                    "lambda": 0.0,
                    "singular_limit_diagnostic": True,
                    "split": split,
                    "sample_id": sample_id,
                    "recipe": _recipe(sample_id),
                    "severity": "mild" if _recipe(sample_id) in MILD else "strong",
                    **prediction_metrics,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "eta": (initial_cd - refined_cd) / (initial_cd - clean_cd),
                    "p2s": float(refined_geometry["p2s"]),
                    "p2s_p95": float(refined_geometry["p2s_p95"]),
                    "fscore": float(refined_geometry["fscore"]),
                    "normal_consistency": float(refined_geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(refined_geometry["introduced_flipped_faces"]),
                    "normalized_flip_rate": float(
                        refined_geometry["introduced_flipped_faces"] / initial.num_faces
                    ),
                    "new_degenerate_faces": int(refined_geometry["new_degenerate_faces"]),
                    "same_index_recovered_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
                    ),
                    "laplacian_residual_rms": float(solver["laplacian_residual_rms"]),
                    "laplacian_residual_max": float(solver["laplacian_residual_max"]),
                    "displacement_mean": float(displacement_norm.mean()),
                    "displacement_rms": float(np.sqrt(np.mean(displacement_norm**2))),
                    "displacement_p95": float(np.quantile(displacement_norm, 0.95)),
                    "displacement_max": float(displacement_norm.max(initial=0.0)),
                    "component_centroid_shift_rms": float(
                        np.sqrt(np.mean(np.sum(component_shifts**2, axis=1)))
                    ),
                    "connected_components": int(component_count),
                    "hard_anchor_count": len(anchors),
                    "anchor_max_abs_error": float(solver["anchor_max_abs_error"]),
                    "lsmr_iterations": int(solver["maximum_iterations"]),
                    "lsmr_condition_estimate": float(solver["maximum_condition_estimate"]),
                    "lsmr_all_converged": bool(solver["all_converged"]),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                }
            )
            print(f"{ARM} {split} {index + 1}/{len(dataset)} {sample_id}", flush=True)
            del values
            torch.cuda.empty_cache()
    _write_json(
        output / "shards" / f"{ARM}.json",
        {
            "arm": ARM,
            "run": str(run),
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "parameter_count": spec["parameter_count"],
            "config": spec["config"],
            "training_metrics": spec["metrics"],
            "training_runtime_diagnostics": _runtime_diagnostic_summary(run),
            "rows": rows,
        },
    )
    np.savez_compressed(
        output / "shards" / f"{ARM}_prediction_arrays.npz",
        **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
    )


def _prediction_summary(payload: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    arrays = np.load(output / "shards" / f"{ARM}_prediction_arrays.npz")
    result: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        prediction = torch.from_numpy(arrays[f"{split}_prediction"])
        target = torch.from_numpy(arrays[f"{split}_target"])
        metrics = raw_gt_magnitude_metrics(
            prediction,
            target,
            torch.ones(len(prediction)),
            torch.ones(len(prediction), dtype=torch.bool),
        )
        result.append({"arm": ARM, "split": split, **metrics})
    return result


def _geometry_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        result.append(
            {
                "arm": ARM,
                "lambda": 0.0,
                "split": split,
                "samples": len(selected),
                **{
                    field: _mean(selected, field)
                    for field in (
                        "initial_chamfer",
                        "refined_chamfer",
                        "relative_chamfer_gain",
                        "eta",
                        "p2s",
                        "p2s_p95",
                        "fscore",
                        "normal_consistency",
                        "normalized_flip_rate",
                        "same_index_recovered_vertex_rms",
                        "laplacian_residual_rms",
                        "displacement_rms",
                        "displacement_p95",
                        "component_centroid_shift_rms",
                    )
                },
                "introduced_flipped_faces": sum(int(row["introduced_flipped_faces"]) for row in selected),
                "new_degenerate_faces": sum(int(row["new_degenerate_faces"]) for row in selected),
                "improved": sum(bool(row["improved"]) for row in selected),
                "worsened": sum(bool(row["worsened"]) for row in selected),
            }
        )
    return result


def _grouped(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for group_type, groups in (("recipe", RECIPES), ("severity", ("mild", "strong"))):
            for group in groups:
                selected = [
                    row
                    for row in rows
                    if row["split"] == split and row[group_type] == group
                ]
                result.append(
                    {
                        "split": split,
                        "group_type": group_type,
                        "group": group,
                        "samples": len(selected),
                        "initial_chamfer": _mean(selected, "initial_chamfer"),
                        "refined_chamfer": _mean(selected, "refined_chamfer"),
                        "mean_per_sample_relative_gain": _mean(selected, "relative_chamfer_gain"),
                        "improved": sum(bool(row["improved"]) for row in selected),
                        "worsened": sum(bool(row["worsened"]) for row in selected),
                        "normalized_flip_rate": _mean(selected, "normalized_flip_rate"),
                        "vertex_rms": _mean(selected, "same_index_recovered_vertex_rms"),
                        "laplacian_residual_rms": _mean(selected, "laplacian_residual_rms"),
                    }
                )
    return result


def _fixed_residual_rows(
    manifest: Path,
    fixed_report: Path,
) -> list[dict[str, Any]]:
    residual_rows: list[dict[str, Any]] = []
    for arm, regularization in zip(FIXED_ARMS, FIXED_LAMBDAS):
        payload = _read(fixed_report / "shards" / f"{arm}.json")
        arrays = np.load(fixed_report / "shards" / f"{arm}_prediction_arrays.npz")
        for split in ("validation", "test"):
            dataset = PreparedMeshDataset.from_manifest(manifest, split)
            arm_rows = [row for row in payload["rows"] if row["split"] == split]
            predictions = arrays[f"{split}_prediction"]
            offset = 0
            for index, row in enumerate(arm_rows):
                static = dataset.load_static(index)
                sample_id = str(static["sample_id"])
                if sample_id != str(row["sample_id"]):
                    raise RuntimeError("Fixed-shard sample ordering mismatch.")
                vertices = int(torch.as_tensor(static["vertices"]).shape[0])
                prediction = predictions[offset : offset + vertices]
                offset += vertices
                if len(prediction) != vertices:
                    raise RuntimeError("Fixed prediction archive excludes vertices unexpectedly.")
                initial = torch.as_tensor(static["vertices"]).numpy().astype(np.float64)
                faces = torch.as_tensor(static["faces"]).numpy().astype(np.int64)
                laplacian, lap_data = uniform_sparse_laplacian(faces, vertices)
                component_count, labels = component_labels(lap_data)
                recovered, solver = regularized_sparse_solve(
                    laplacian,
                    prediction.astype(np.float64),
                    initial,
                    labels,
                    component_count,
                    regularization,
                    atol=1e-12,
                    btol=1e-12,
                    maxiter=100000,
                )
                residual = laplacian @ recovered - prediction
                residual_rows.append(
                    {
                        "arm": arm,
                        "lambda": regularization,
                        "split": split,
                        "sample_id": sample_id,
                        "laplacian_residual_rms": float(
                            np.sqrt(np.mean(np.sum(residual**2, axis=1)))
                        ),
                        "lsmr_all_converged": bool(solver["all_converged"]),
                    }
                )
            if offset != len(predictions):
                raise RuntimeError("Fixed prediction archive length mismatch.")
    return residual_rows


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    fixed = args.fixed_report_dir.resolve()
    payload = _read(output / "shards" / f"{ARM}.json")
    rows = payload["rows"]
    prediction = _prediction_summary(payload, output)
    geometry = _geometry_summary(rows)
    grouped = _grouped(rows)
    fixed_residuals = _fixed_residual_rows(args.manifest.resolve(), fixed)
    fixed_per_sample: list[dict[str, str]] = []
    with (fixed / "per_sample.csv").open("r", encoding="utf-8", newline="") as handle:
        fixed_per_sample = list(csv.DictReader(handle))
    b_by_key = {
        (row["split"], row["sample_id"]): row
        for row in fixed_per_sample
        if row["arm"] == "B_lap_plus_refine"
    }
    b_residual = {
        (row["split"], row["sample_id"]): row
        for row in fixed_residuals
        if row["arm"] == "B_lap_plus_refine"
    }
    paired: list[dict[str, Any]] = []
    for row in rows:
        key = (row["split"], row["sample_id"])
        baseline = b_by_key[key]
        baseline_residual = b_residual[key]
        paired.append(
            {
                "split": row["split"],
                "sample_id": row["sample_id"],
                "lambda0_lower_chamfer": float(row["refined_chamfer"]) < float(baseline["refined_chamfer"]),
                "lambda0_lower_vertex_rms": float(row["same_index_recovered_vertex_rms"]) < float(baseline["same_index_recovered_vertex_rms"]),
                "lambda0_lower_p2s_p95": float(row["p2s_p95"]) < float(baseline["p2s_p95"]),
                "lambda0_better_normal": float(row["normal_consistency"]) > float(baseline["normal_consistency"]),
                "lambda0_lower_flip_rate": float(row["normalized_flip_rate"]) < float(baseline["normalized_flip_rate"]),
                "lambda0_lower_laplacian_residual": float(row["laplacian_residual_rms"]) < float(baseline_residual["laplacian_residual_rms"]),
            }
        )
    fixed_geometry = list(csv.DictReader((fixed / "geometry_summary.csv").open("r", encoding="utf-8", newline="")))
    fixed_prediction = list(csv.DictReader((fixed / "prediction_summary.csv").open("r", encoding="utf-8", newline="")))
    connectivity = _read(args.connectivity_audit.resolve())
    solver_audit = _read(args.solver_audit.resolve())
    runtime = payload.get("training_runtime_diagnostics")
    contract = {
        "contract_audit": bool(
            connectivity["contract_audit"]
            and solver_audit["passed"]
            and int(payload["training_metrics"]["optimizer_steps"]) == 20000
            and int(payload["training_metrics"]["global_batch_meshes"]) == 8
            and all(bool(row["lsmr_all_converged"]) for row in rows)
            and all(float(row["anchor_max_abs_error"]) == 0.0 for row in rows)
            and runtime is not None
            and int(runtime["pcg_failed_solves"]) == 0
            and int(runtime["nan_inf_count"]) == 0
        ),
        "lambda": 0.0,
        "hard_anchor_per_component": True,
        "anchor_selection_uses_gt": False,
        "centroid_constraint": False,
        "soft_positional_regularization": False,
        "hidden_tikhonov_damping": False,
        "metric_protocol": METRIC_PROTOCOL,
    }
    test_pairs = [row for row in paired if row["split"] == "test"]
    paired_counts = {
        field: sum(bool(row[field]) for row in test_pairs)
        for field in (
            "lambda0_lower_chamfer",
            "lambda0_lower_vertex_rms",
            "lambda0_lower_p2s_p95",
            "lambda0_better_normal",
            "lambda0_lower_flip_rate",
            "lambda0_lower_laplacian_residual",
        )
    }
    summary = {
        "contract_audit": contract,
        "connectivity_audit": connectivity,
        "solver_audit": solver_audit,
        "training_runtime_diagnostics": runtime,
        "prediction_lambda0": prediction,
        "geometry_lambda0": geometry,
        "recipe_and_severity_lambda0": grouped,
        "paired_lambda0_vs_lambda1e-2_test": paired_counts,
        "fixed_geometry": fixed_geometry,
        "fixed_prediction": fixed_prediction,
        "fixed_residual_summary": [
            {
                "arm": arm,
                "lambda": regularization,
                "split": split,
                "laplacian_residual_rms": _mean(
                    [row for row in fixed_residuals if row["arm"] == arm and row["split"] == split],
                    "laplacian_residual_rms",
                ),
            }
            for arm, regularization in zip(FIXED_ARMS, FIXED_LAMBDAS)
            for split in ("validation", "test")
        ],
    }
    _write_csv(output / "lambda0_per_sample.csv", rows)
    _write_csv(output / "lambda0_recipe_severity.csv", grouped)
    _write_csv(output / "lambda0_vs_lambda1e-2_paired.csv", paired)
    _write_csv(output / "fixed_lambda_residual_supplement.csv", fixed_residuals)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contract_audit.json", contract)
    test_i = next(row for row in geometry if row["split"] == "test")
    test_b = next(row for row in fixed_geometry if row["split"] == "test" and row["arm"] == "B_lap_plus_refine")
    residual_b = next(row for row in summary["fixed_residual_summary"] if row["split"] == "test" and row["arm"] == "B_lap_plus_refine")
    inverse_amplification = bool(
        test_i["laplacian_residual_rms"] < residual_b["laplacian_residual_rms"]
        and test_i["refined_chamfer"] > float(test_b["refined_chamfer"])
    )
    residual_by_arm = {
        row["arm"]: float(row["laplacian_residual_rms"])
        for row in summary["fixed_residual_summary"]
        if row["split"] == "test"
    }
    fixed_test = sorted(
        (row for row in fixed_geometry if row["split"] == "test"),
        key=lambda row: float(row["lambda"]),
    )
    lines = [
        "# Sofa50 v2 lambda=0 hard-anchor singular-limit diagnostic",
        "",
        f"Contract audit: **{str(contract['contract_audit']).lower()}**.",
        "",
        "Arm I solves the reduced constrained least-squares problem with one exact initial-position anchor per connected component. It uses no centroid constraint, soft positional term or hidden damping.",
        "",
        "## Connectivity and solver audit",
        "",
        f"- Meshes: {connectivity['meshes']}; multi-component: {connectivity['meshes_with_multiple_components']}; components min/mean/max: {connectivity['connected_components_minimum']} / {connectivity['connected_components_mean']:.2f} / {connectivity['connected_components_maximum']}; anchors total: {connectivity['hard_anchors_total']}.",
        f"- Real-mesh direct-sparse/LSMR preflight: **{str(solver_audit['passed']).lower()}**; training failed solves: {runtime['pcg_failed_solves']}; NaN/Inf: {runtime['nan_inf_count']}.",
        "",
        "## Prediction",
        "",
        "| Split | Raw EPE | Raw RMS | Cosine | Bottom90 | Top10 | Top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prediction:
        lines.append(f"| {row['split']} | {row['raw_epe']:.9g} | {row['raw_rms']:.9g} | {row['raw_cosine']:.9g} | {row['bottom90_epe']:.9g} | {row['top10_epe']:.9g} | {row['top1_epe']:.9g} |")
    lines.extend((
        "",
        "## Recovered geometry",
        "",
        "| Split | Initial CD | Refined CD | Gain | Eta | P2S | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened | Vertex RMS | Lap residual | Displ. RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in geometry:
        lines.append(f"| {row['split']} | {row['initial_chamfer']:.9g} | {row['refined_chamfer']:.9g} | {row['relative_chamfer_gain']:+.2%} | {row['eta']:.9g} | {row['p2s']:.9g} | {row['p2s_p95']:.9g} | {row['fscore']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} / {row['normalized_flip_rate']:.4%} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} | {row['same_index_recovered_vertex_rms']:.9g} | {row['laplacian_residual_rms']:.9g} | {row['displacement_rms']:.9g} |")
    lines.extend(("", "## Recipe and severity breakdown", "", "| Split | Group | Initial CD | Final CD | Gain | Improved/worsened | Flip rate | Vertex RMS | Lap residual |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"))
    for row in grouped:
        lines.append(f"| {row['split']} | {row['group']} | {row['initial_chamfer']:.9g} | {row['refined_chamfer']:.9g} | {row['mean_per_sample_relative_gain']:+.2%} | {row['improved']}/{row['worsened']} | {row['normalized_flip_rate']:.4%} | {row['vertex_rms']:.9g} | {row['laplacian_residual_rms']:.9g} |")
    lines.extend((
        "",
        "## Fixed-recovery test comparison",
        "",
        "Lambda=0 is a singular-limit hard-anchor diagnostic; all positive lambdas are ordinary positional-regularized arms.",
        "",
        "| Lambda | Refined CD | Gain | Eta | P2S p95 | Normal | Flip rate | Vertex RMS | Lap residual |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 0 (hard anchor) | {test_i['refined_chamfer']:.9g} | {test_i['relative_chamfer_gain']:+.2%} | {test_i['eta']:.9g} | {test_i['p2s_p95']:.9g} | {test_i['normal_consistency']:.9g} | {test_i['normalized_flip_rate']:.4%} | {test_i['same_index_recovered_vertex_rms']:.9g} | {test_i['laplacian_residual_rms']:.9g} |",
    ))
    for row in fixed_test:
        lines.append(
            f"| {float(row['lambda']):.0e} | {float(row['refined_chamfer']):.9g} | "
            f"{float(row['relative_chamfer_gain']):+.2%} | {float(row['eta']):.9g} | "
            f"{float(row['p2s_p95']):.9g} | {float(row['normal_consistency']):.9g} | "
            f"{float(row['normalized_flip_rate']):.4%} | "
            f"{float(row['same_index_recovered_vertex_rms']):.9g} | "
            f"{residual_by_arm[row['arm']]:.9g} |"
        )
    lines.extend((
        "",
        "## Lambda=0 versus fixed lambda=1e-2",
        "",
        f"On test, lambda=0 wins {paired_counts['lambda0_lower_chamfer']}/50 Chamfer, {paired_counts['lambda0_lower_vertex_rms']}/50 vertex RMS, {paired_counts['lambda0_lower_p2s_p95']}/50 P2S p95, {paired_counts['lambda0_better_normal']}/50 normal, {paired_counts['lambda0_lower_flip_rate']}/50 normalized flip rate, and {paired_counts['lambda0_lower_laplacian_residual']}/50 Laplacian residual comparisons.",
        "",
        "## Conclusion",
        "",
        (
            "The hard-anchored lambda=0 limit fits the predicted differential coordinates more closely but recovers worse geometry than lambda=1e-2. This directly confirms inverse-Laplacian error amplification."
            if inverse_amplification
            else "The lambda=0 residual-versus-geometry result does not satisfy the predeclared inverse-amplification pattern; inspect the paired and recipe tables before drawing a solver-design conclusion."
        ),
        "",
    ))
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--fixed-report-dir", type=Path)
    parser.add_argument("--connectivity-audit", type=Path)
    parser.add_argument("--solver-audit", type=Path)
    args = parser.parse_args()
    if args.merge_only:
        if args.fixed_report_dir is None or args.connectivity_audit is None or args.solver_audit is None:
            parser.error("merge requires fixed-report-dir, connectivity-audit and solver-audit")
        merge(args)
    else:
        if args.runs_root is None:
            parser.error("evaluation requires runs-root")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

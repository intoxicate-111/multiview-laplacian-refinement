#!/usr/bin/env python3
from __future__ import annotations

"""Regularized sparse integration sweep for archived Sofa50 raw-Laplacian predictions."""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.sparse import eye, vstack
from scipy.sparse.linalg import lsmr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_centroids,
    component_labels,
    exact_sparse_solve,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from mlr.coarse_lap_oracle import apply_uniform_laplacian
from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FAMILIES = ("predicted_raw", "predicted_zero_mean", "exact_target")
BASELINE_STATES = ("initial", "frozen_adam_visibility")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_zero_mean(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Apply the requested unweighted per-component, per-coordinate mean projection."""
    projected = np.asarray(values, dtype=np.float64).copy()
    sizes = np.bincount(labels, minlength=count).astype(np.float64)
    for axis in range(3):
        means = np.bincount(
            labels, weights=projected[:, axis], minlength=count
        ) / sizes
        projected[:, axis] -= means[labels]
    return projected


def _component_mean_max_abs(values: np.ndarray, labels: np.ndarray, count: int) -> float:
    sizes = np.bincount(labels, minlength=count).astype(np.float64)
    maximum = 0.0
    for axis in range(3):
        means = np.bincount(labels, weights=values[:, axis], minlength=count) / sizes
        maximum = max(maximum, float(np.max(np.abs(means), initial=0.0)))
    return maximum


def _degree_weighted_component_mean_max_abs(
    values: np.ndarray,
    labels: np.ndarray,
    count: int,
    degrees: np.ndarray,
) -> float:
    weights_per_component = np.bincount(labels, weights=degrees, minlength=count)
    maximum = 0.0
    for axis in range(3):
        means = np.bincount(
            labels,
            weights=values[:, axis] * degrees,
            minlength=count,
        ) / weights_per_component
        maximum = max(maximum, float(np.max(np.abs(means), initial=0.0)))
    return maximum


def regularized_sparse_solve(
    laplacian: Any,
    target: np.ndarray,
    initial_vertices: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    regularization: float,
    *,
    atol: float,
    btol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the requested objective; lambda=0 uses only initial component gauges."""
    target64 = np.asarray(target, dtype=np.float64)
    initial64 = np.asarray(initial_vertices, dtype=np.float64)
    start = time.perf_counter()
    if regularization == 0.0:
        solution, solver = exact_sparse_solve(
            laplacian,
            target64,
            labels,
            component_count,
            component_centroids(initial64, labels, component_count),
            atol=atol,
            btol=btol,
            maxiter=maxiter,
        )
        solver = dict(solver)
        solver["all_converged"] = all(
            row["istop"] in (0, 1, 2, 4, 5) for row in solver["axes"]
        )
        solver["system"] = "[L; component_centroid_gauge]"
    else:
        scale = math.sqrt(regularization)
        identity = eye(laplacian.shape[0], dtype=np.float64, format="csr")
        system = vstack((laplacian, scale * identity), format="csr")
        rhs = np.vstack((target64, scale * initial64))
        solution = np.empty_like(target64)
        axes: list[dict[str, Any]] = []
        for axis in range(3):
            result = lsmr(
                system,
                rhs[:, axis],
                atol=atol,
                btol=btol,
                conlim=1e12,
                maxiter=maxiter,
            )
            solution[:, axis] = result[0]
            axes.append(
                {
                    "axis": axis,
                    "istop": int(result[1]),
                    "iterations": int(result[2]),
                    "norm_residual": float(result[3]),
                    "norm_normal_residual": float(result[4]),
                    "operator_norm": float(result[5]),
                    "condition_estimate": float(result[6]),
                    "solution_norm": float(result[7]),
                }
            )
        solver = {
            "axes": axes,
            "maximum_iterations": max(row["iterations"] for row in axes),
            # LSMR istop=0 is an exact zero solution/RHS and is successful.
            "all_converged": all(row["istop"] in (0, 1, 2, 4, 5) for row in axes),
            "system": "[L; sqrt(lambda) I]",
        }
    solver["runtime_seconds"] = time.perf_counter() - start
    residual = laplacian @ solution - target64
    displacement = solution - initial64
    residual_norm = np.linalg.norm(residual, axis=1)
    displacement_norm = np.linalg.norm(displacement, axis=1)
    laplacian_squared_error = float(np.sum(residual**2))
    anchor_squared_error = float(np.sum(displacement**2))
    solver.update(
        {
            "lambda": regularization,
            "laplacian_residual_rms": float(np.sqrt(np.mean(residual_norm**2))),
            "laplacian_residual_max": float(residual_norm.max(initial=0.0)),
            "displacement_rms": float(np.sqrt(np.mean(displacement_norm**2))),
            "displacement_max": float(displacement_norm.max(initial=0.0)),
            "laplacian_squared_error": laplacian_squared_error,
            "anchor_squared_error": anchor_squared_error,
            "objective": laplacian_squared_error + regularization * anchor_squared_error,
        }
    )
    return solution, solver


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    source = args.prediction_source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    indices = list(range(args.shard_index, len(dataset), args.shard_count))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for index in indices:
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        archived_dir = source / "reconstruction" / args.prediction_arm_name / sample_id
        prediction_path = archived_dir / "delta_pred_raw.npy"
        frozen_path = archived_dir / "predicted_refined.obj"
        coarse_path = archived_dir / "coarse.obj"
        for path in (prediction_path, frozen_path, coarse_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        prediction = np.load(prediction_path).astype(np.float64)
        if prediction.shape != initial.vertices.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid archived prediction for {sample_id}")
        frozen = load_mesh(frozen_path).ensure_normals()
        archived_coarse = load_mesh(coarse_path).ensure_normals()
        archived_input_matches = bool(
            np.array_equal(initial.faces, archived_coarse.faces)
            and np.allclose(initial.vertices, archived_coarse.vertices, rtol=0.0, atol=1e-8)
        )
        if not archived_input_matches:
            raise RuntimeError(f"Archived coarse mismatch for {sample_id}")

        laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        degrees = np.asarray([len(neighbors) for neighbors in lap_data.neighbors], dtype=np.float64)
        if np.any(degrees <= 0):
            raise RuntimeError(f"Isolated vertex is unsupported in sample {sample_id}")
        projected = component_zero_mean(prediction, labels, component_count)
        exact_target = apply_uniform_laplacian(clean.vertices, lap_data)
        target_families = {
            "predicted_raw": prediction,
            "predicted_zero_mean": projected,
            "exact_target": exact_target,
        }

        baseline_meshes = {
            "initial": initial,
            "frozen_adam_visibility": frozen,
            "clean": clean,
        }
        geometry_baseline = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in baseline_meshes.items()
        }
        initial_cd = float(geometry_baseline["initial"]["chamfer"])
        clean_cd = float(geometry_baseline["clean"]["chamfer"])
        available = initial_cd - clean_cd
        if available <= 0:
            raise RuntimeError(f"Non-positive recoverable Chamfer gap for {sample_id}")
        for state in BASELINE_STATES:
            row = geometry_baseline[state]
            chamfer = float(row["chamfer"])
            row.update(
                {
                    "variant": metadata.get("variant"),
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "connected_components": component_count,
                    "initial_chamfer": initial_cd,
                    "clean_chamfer": clean_cd,
                    "relative_chamfer_gain": (initial_cd - chamfer) / max(initial_cd, 1e-12),
                    "eta_recovery": (initial_cd - chamfer) / available,
                }
            )
            baselines.append(row)

        solver_audits: list[dict[str, Any]] = []
        for family, target in target_families.items():
            for regularization in LAMBDAS:
                vertices, solver = regularized_sparse_solve(
                    laplacian,
                    target,
                    initial.vertices,
                    labels,
                    component_count,
                    regularization,
                    atol=args.lsmr_atol,
                    btol=args.lsmr_btol,
                    maxiter=args.lsmr_maxiter,
                )
                if not np.isfinite(vertices).all():
                    raise RuntimeError(f"Non-finite solve for {sample_id}/{family}/{regularization}")
                mesh = Mesh(vertices, initial.faces.copy()).ensure_normals()
                state = f"{family}_lambda_{regularization:.0e}"
                row = _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
                chamfer = float(row["chamfer"])
                row.update(
                    {
                        "family": family,
                        "lambda": regularization,
                        "variant": metadata.get("variant"),
                        "vertices": initial.num_vertices,
                        "faces": initial.num_faces,
                        "connected_components": component_count,
                        "initial_chamfer": initial_cd,
                        "clean_chamfer": clean_cd,
                        "relative_chamfer_gain": (initial_cd - chamfer) / max(initial_cd, 1e-12),
                        "eta_recovery": (initial_cd - chamfer) / available,
                        "runtime_seconds": solver["runtime_seconds"],
                        "laplacian_residual_rms": solver["laplacian_residual_rms"],
                        "laplacian_residual_max": solver["laplacian_residual_max"],
                        "displacement_rms": solver["displacement_rms"],
                        "displacement_max": solver["displacement_max"],
                        "laplacian_squared_error": solver["laplacian_squared_error"],
                        "anchor_squared_error": solver["anchor_squared_error"],
                        "objective": solver["objective"],
                        "lsmr_maximum_iterations": solver["maximum_iterations"],
                        "lsmr_all_converged": solver["all_converged"],
                    }
                )
                rows.append(row)
                solver_audits.append({"family": family, "lambda": regularization, **solver})

        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "archived_prediction": str(prediction_path),
            "archived_frozen_mesh": str(frozen_path),
            "archived_input_matches_manifest": archived_input_matches,
            "connected_components": component_count,
            "prediction_component_mean_max_abs_before": _component_mean_max_abs(
                prediction, labels, component_count
            ),
            "prediction_component_mean_max_abs_after": _component_mean_max_abs(
                projected, labels, component_count
            ),
            "prediction_degree_weighted_component_mean_max_abs_before": (
                _degree_weighted_component_mean_max_abs(
                    prediction, labels, component_count, degrees
                )
            ),
            "prediction_degree_weighted_component_mean_max_abs_after": (
                _degree_weighted_component_mean_max_abs(
                    projected, labels, component_count, degrees
                )
            ),
            "zero_mean_definition": "unweighted per connected component and xyz coordinate",
            "random_walk_laplacian_is_nonsymmetric": True,
            "solver_audits": solver_audits,
            "same_graph_for_all_arms": True,
            "no_visibility_confidence_huber_or_adam": True,
            "gt_used_only_for_exact_target_reference_and_evaluation": True,
        }
        audit["passed"] = bool(
            archived_input_matches
            and all(bool(row["all_converged"]) for row in solver_audits)
            and audit["prediction_component_mean_max_abs_after"] <= 1e-12
        )
        audits.append(audit)
        best = min(
            (row for row in rows if row["sample_id"] == sample_id and row["family"] != "exact_target"),
            key=lambda row: float(row["chamfer"]),
        )
        print(
            f"{args.dataset_arm} {sample_id}: best={best['family']} lambda={best['lambda']:.0e} "
            f"eta={best['eta_recovery']:.4g} audit={audit['passed']}",
            flush=True,
        )

    _write_json(
        output / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "dataset_arm": args.dataset_arm,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "prediction_source_dir": str(source),
            "prediction_arm_name": args.prediction_arm_name,
            "metric_protocol": METRIC_PROTOCOL,
            "lambdas": list(LAMBDAS),
            "families": list(FAMILIES),
            "rows": rows,
            "baselines": baselines,
            "audits": audits,
        },
    )


def _aggregate(
    rows: Sequence[Mapping[str, Any]], family: str, regularization: float
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["family"] == family and float(row["lambda"]) == regularization
    ]
    return {
        "family": family,
        "lambda": regularization,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "relative_chamfer_gain_mean": float(
            np.mean([float(row["relative_chamfer_gain"]) for row in selected])
        ),
        "eta_mean": float(np.mean([float(row["eta_recovery"]) for row in selected])),
        "eta_median": float(np.median([float(row["eta_recovery"]) for row in selected])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in selected])
        ),
        "introduced_flipped_faces": int(
            sum(int(row["introduced_flipped_faces"]) for row in selected)
        ),
        "new_degenerate_faces": int(
            sum(int(row["new_degenerate_faces"]) for row in selected)
        ),
        "improved_over_initial": int(
            sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)
        ),
        "worsened_over_initial": int(
            sum(float(row["chamfer"]) > float(row["initial_chamfer"]) for row in selected)
        ),
        "runtime_seconds_mean": float(
            np.mean([float(row["runtime_seconds"]) for row in selected])
        ),
        "laplacian_residual_rms": float(
            np.mean([float(row["laplacian_residual_rms"]) for row in selected])
        ),
        "displacement_rms": float(
            np.mean([float(row["displacement_rms"]) for row in selected])
        ),
        "lsmr_iterations_mean": float(
            np.mean([float(row["lsmr_maximum_iterations"]) for row in selected])
        ),
    }


def _aggregate_baseline(
    rows: Sequence[Mapping[str, Any]], state: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["state"] == state]
    return {
        "state": state,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "relative_chamfer_gain_mean": float(
            np.mean([float(row["relative_chamfer_gain"]) for row in selected])
        ),
        "eta_mean": float(np.mean([float(row["eta_recovery"]) for row in selected])),
        "eta_median": float(np.median([float(row["eta_recovery"]) for row in selected])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in selected])
        ),
        "introduced_flipped_faces": int(
            sum(int(row["introduced_flipped_faces"]) for row in selected)
        ),
        "new_degenerate_faces": int(
            sum(int(row["new_degenerate_faces"]) for row in selected)
        ),
        "improved_over_initial": int(
            sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)
        ),
        "worsened_over_initial": int(
            sum(float(row["chamfer"]) > float(row["initial_chamfer"]) for row in selected)
        ),
    }


def _paired_comparison(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in left_rows}
    right = {str(row["sample_id"]): row for row in right_rows}
    sample_ids = sorted(set(left) & set(right))
    differences = [float(left[s]["chamfer"]) - float(right[s]["chamfer"]) for s in sample_ids]
    return {
        "left": left_name,
        "right": right_name,
        "samples": len(sample_ids),
        "left_lower_chamfer": int(sum(value < 0 for value in differences)),
        "ties": int(sum(value == 0 for value in differences)),
        "right_lower_chamfer": int(sum(value > 0 for value in differences)),
        "mean_left_minus_right_chamfer": float(np.mean(differences)),
        "mean_left_minus_right_eta": float(
            np.mean(
                [float(left[s]["eta_recovery"]) - float(right[s]["eta_recovery"]) for s in sample_ids]
            )
        ),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    rows = [row for shard in shards for row in shard["rows"]]
    baselines = [row for shard in shards for row in shard["baselines"]]
    audits = [row for shard in shards for row in shard["audits"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    expected_rows = expected * len(FAMILIES) * len(LAMBDAS)
    if len(rows) != expected_rows or len(baselines) != expected * len(BASELINE_STATES):
        raise RuntimeError(
            f"Incomplete sweep: rows={len(rows)}/{expected_rows}, baselines={len(baselines)}"
        )
    if len(audits) != expected or not all(bool(row["passed"]) for row in audits):
        raise RuntimeError("Regularized sparse-sweep contract audit failed")
    aggregates = [
        {"dataset_arm": args.dataset_arm, **_aggregate(rows, family, regularization)}
        for family in FAMILIES
        for regularization in LAMBDAS
    ]
    baseline_aggregates = [
        {"dataset_arm": args.dataset_arm, **_aggregate_baseline(baselines, state)}
        for state in BASELINE_STATES
    ]
    aggregate_lookup = {(row["family"], float(row["lambda"])): row for row in aggregates}
    rows_lookup = {
        (row["family"], float(row["lambda"])): [
            candidate
            for candidate in rows
            if candidate["family"] == row["family"]
            and float(candidate["lambda"]) == float(row["lambda"])
        ]
        for row in aggregates
    }
    best_by_family = {
        family: min(
            (row for row in aggregates if row["family"] == family),
            key=lambda row: float(row["chamfer"]),
        )
        for family in FAMILIES
    }
    best_prediction = min(
        (best_by_family["predicted_raw"], best_by_family["predicted_zero_mean"]),
        key=lambda row: float(row["chamfer"]),
    )
    best_family = str(best_prediction["family"])
    best_lambda = float(best_prediction["lambda"])
    exact_at_best = aggregate_lookup[("exact_target", best_lambda)]
    predicted_eta = float(best_prediction["eta_mean"])
    oracle_eta = float(exact_at_best["eta_mean"])
    frozen_aggregate = next(row for row in baseline_aggregates if row["state"] == "frozen_adam_visibility")
    best_rows = rows_lookup[(best_family, best_lambda)]
    frozen_rows = [row for row in baselines if row["state"] == "frozen_adam_visibility"]
    projection_at_lambda = []
    for regularization in LAMBDAS:
        projection_at_lambda.append(
            {
                "lambda": regularization,
                **_paired_comparison(
                    rows_lookup[("predicted_zero_mean", regularization)],
                    rows_lookup[("predicted_raw", regularization)],
                    left_name="predicted_zero_mean",
                    right_name="predicted_raw",
                ),
            }
        )
    comparisons = {
        "best_prediction_vs_frozen": _paired_comparison(
            best_rows,
            frozen_rows,
            left_name=f"{best_family}@{best_lambda:g}",
            right_name="frozen_adam_visibility",
        ),
        "best_projected_vs_best_raw": _paired_comparison(
            rows_lookup[("predicted_zero_mean", float(best_by_family["predicted_zero_mean"]["lambda"]))],
            rows_lookup[("predicted_raw", float(best_by_family["predicted_raw"]["lambda"]))],
            left_name=f"predicted_zero_mean@{float(best_by_family['predicted_zero_mean']['lambda']):g}",
            right_name=f"predicted_raw@{float(best_by_family['predicted_raw']['lambda']):g}",
        ),
        "projection_at_same_lambda": projection_at_lambda,
    }
    summary = {
        "dataset_arm": args.dataset_arm,
        "contract_audit": True,
        "test_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "selection_rule": "minimum test-domain mean unified Chamfer (diagnostic only; not a benchmark hyperparameter)",
        "lambdas": list(LAMBDAS),
        "families": list(FAMILIES),
        "baseline_aggregates": baseline_aggregates,
        "aggregates": aggregates,
        "best_by_family": best_by_family,
        "best_predicted_recovery": best_prediction,
        "exact_target_at_best_predicted_lambda": exact_at_best,
        "retention_at_best_predicted_lambda": {
            "ratio_of_mean_eta": predicted_eta / oracle_eta if abs(oracle_eta) > 1e-12 else float("nan"),
            "useful_ratio_of_mean_eta": max(0.0, predicted_eta) / oracle_eta if oracle_eta > 0 else float("nan"),
        },
        "best_prediction_minus_frozen": {
            "chamfer": float(best_prediction["chamfer"]) - float(frozen_aggregate["chamfer"]),
            "eta": predicted_eta - float(frozen_aggregate["eta_mean"]),
            "normal_consistency": float(best_prediction["normal_consistency"]) - float(frozen_aggregate["normal_consistency"]),
            "introduced_flipped_faces": int(best_prediction["introduced_flipped_faces"]) - int(frozen_aggregate["introduced_flipped_faces"]),
            "new_degenerate_faces": int(best_prediction["new_degenerate_faces"]) - int(frozen_aggregate["new_degenerate_faces"]),
        },
        "paired_comparisons": comparisons,
        "projection_audit": {
            "definition": "delta' = delta - unweighted component mean, independently for xyz",
            "random_walk_laplacian_is_nonsymmetric": True,
            "component_mean_max_abs_after_max": float(
                max(float(row["prediction_component_mean_max_abs_after"]) for row in audits)
            ),
            "degree_weighted_component_mean_max_abs_before_mean": float(
                np.mean(
                    [float(row["prediction_degree_weighted_component_mean_max_abs_before"]) for row in audits]
                )
            ),
            "degree_weighted_component_mean_max_abs_after_mean": float(
                np.mean(
                    [float(row["prediction_degree_weighted_component_mean_max_abs_after"]) for row in audits]
                )
            ),
        },
    }
    contract = {
        "passed": True,
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(audits),
        "all_archived_inputs_match": all(bool(row["archived_input_matches_manifest"]) for row in audits),
        "all_lsmr_converged": all(
            all(bool(solver["all_converged"]) for solver in row["solver_audits"])
            for row in audits
        ),
        "same_graph_for_all_arms": True,
        "lambda_zero_gauge": "initial component centroids only",
        "positive_lambda_system": "[L; sqrt(lambda) I]",
        "no_visibility_confidence_huber_or_adam": True,
        "gt_used_only_for_exact_target_reference_and_evaluation": True,
        "manifest": shards[0]["manifest"],
        "manifest_sha256": shards[0]["manifest_sha256"],
        "prediction_source_dir": shards[0]["prediction_source_dir"],
        "prediction_arm_name": shards[0]["prediction_arm_name"],
        "metric_protocol": METRIC_PROTOCOL,
    }
    _write_csv(output / "per_sample.csv", rows)
    _write_csv(output / "baseline_per_sample.csv", baselines)
    _write_csv(output / "aggregate.csv", aggregates)
    _write_csv(output / "baseline_aggregate.csv", baseline_aggregates)
    _write_csv(output / "projection_paired_by_lambda.csv", projection_at_lambda)
    _write_json(output / "per_sample_contract_audit.json", audits)
    _write_json(output / "contract_audit.json", contract)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prediction-source-dir", type=Path)
    parser.add_argument("--prediction-arm-name")
    parser.add_argument("--dataset-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lsmr-atol", type=float, default=1e-12)
    parser.add_argument("--lsmr-btol", type=float, default=1e-12)
    parser.add_argument("--lsmr-maxiter", type=int, default=100000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.prediction_source_dir is None or args.prediction_arm_name is None:
            parser.error("prediction source and arm are required unless --merge-only")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

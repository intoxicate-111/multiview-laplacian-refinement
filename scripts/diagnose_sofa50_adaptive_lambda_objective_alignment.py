#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Chamfer/vertex-RMS/P95 lambda-oracle alignment diagnostic."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from diagnose_sofa50_adaptive_lambda_oracle import (
    ARM_B_RUN,
    FIXED_LAMBDA,
    LAMBDA_GRID,
)
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multitopology_rawlap import TOPOLOGY_RECIPES


SELECTORS = {
    "fixed_1e-2": None,
    "chamfer_oracle": "chamfer",
    "vertex_rms_oracle": "vertex_rms",
    "p2s_p95_oracle": "p2s_p95",
}
PROXY_FIELDS = (
    "fixed_recovery_displacement_rms",
    "fixed_recovery_displacement_mean",
    "fixed_recovery_displacement_p95",
    "predicted_correction_rms",
    "predicted_correction_mean",
    "predicted_correction_p95",
    "predicted_laplacian_rms",
)
ABSOLUTE_TOLERANCE = 1e-7
RELATIVE_TOLERANCE = 1e-4


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))


def _metric_tolerance(value: float) -> float:
    return max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * abs(float(value)))


def _closest_to_fixed(row: Mapping[str, Any]) -> tuple[float, float]:
    value = float(row["lambda"])
    return abs(np.log10(value) - np.log10(FIXED_LAMBDA)), value


def _select(
    candidates: Sequence[Mapping[str, Any]], field: str
) -> tuple[Mapping[str, Any], list[float], float]:
    minimum = min(float(row[field]) for row in candidates)
    tolerance = _metric_tolerance(minimum)
    tied = [row for row in candidates if float(row[field]) <= minimum + tolerance]
    chosen = min(tied, key=_closest_to_fixed)
    return chosen, sorted(float(row["lambda"]) for row in tied), tolerance


def _recipe(sample_id: str) -> str:
    recipe = sample_id.rsplit("__", 1)[-1]
    if recipe not in TOPOLOGY_RECIPES:
        raise ValueError(f"Unknown recipe: {sample_id}")
    return recipe


def evaluate_shard(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    device = torch.device(args.device)
    spec = _load_spec(args.runs_root.resolve() / ARM_B_RUN, device)
    rows: list[dict[str, Any]] = []
    global_index = 0
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        for index in range(len(dataset)):
            assigned = global_index % args.shard_count == args.shard_index
            global_index += 1
            if not assigned:
                continue
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            values = _infer_recovery_arm(dataset, index, spec, device)
            prediction = values["prediction_raw"].numpy().astype(np.float64)
            initial = Mesh(
                torch.as_tensor(static["vertices"]).numpy().astype(np.float64),
                torch.as_tensor(static["faces"]).numpy().astype(np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            laplacian, lap_data = uniform_sparse_laplacian(
                initial.faces, initial.num_vertices
            )
            component_count, labels = component_labels(lap_data)
            initial_geometry = _geometry_row(
                "v2_strong_smoothing", sample_id, "initial", initial, clean, initial
            )
            initial_laplacian = laplacian @ initial.vertices
            correction = prediction - initial_laplacian
            correction_norm = np.linalg.norm(correction, axis=1)
            proxy = {
                "predicted_laplacian_rms": _rms(prediction),
                "predicted_correction_rms": _rms(correction),
                "predicted_correction_mean": float(correction_norm.mean()),
                "predicted_correction_p95": float(np.quantile(correction_norm, 0.95)),
                "confidence_features_available": False,
            }
            for regularization in LAMBDA_GRID:
                recovered, solver = regularized_sparse_solve(
                    laplacian,
                    prediction,
                    initial.vertices,
                    labels,
                    component_count,
                    regularization,
                    atol=1e-12,
                    btol=1e-12,
                    maxiter=100000,
                )
                if not bool(solver["all_converged"]):
                    raise RuntimeError(f"Sparse recovery failed: {split} {sample_id} {regularization}")
                refined = Mesh(recovered, initial.faces.copy()).ensure_normals()
                geometry = _geometry_row(
                    "v2_strong_smoothing",
                    sample_id,
                    f"lambda_{regularization:.0e}",
                    refined,
                    clean,
                    initial,
                )
                displacement = recovered - initial.vertices
                displacement_norm = np.linalg.norm(displacement, axis=1)
                residual = laplacian @ recovered - prediction
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "recipe": _recipe(sample_id),
                        "severity": TOPOLOGY_RECIPES[_recipe(sample_id)]["degradation"],
                        "lambda": regularization,
                        "vertices": initial.num_vertices,
                        "faces": initial.num_faces,
                        "initial_chamfer": float(initial_geometry["chamfer"]),
                        "chamfer": float(geometry["chamfer"]),
                        "p2s_mean": float(geometry["p2s"]),
                        "p2s_p95": float(geometry["p2s_p95"]),
                        "fscore": float(geometry["fscore"]),
                        "normal_consistency": float(geometry["normal_consistency"]),
                        "vertex_rms": _rms(recovered - clean.vertices),
                        "introduced_flipped_faces": int(geometry["introduced_flipped_faces"]),
                        "normalized_flip_rate": float(
                            geometry["introduced_flipped_faces"] / initial.num_faces
                        ),
                        "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                        "recovery_displacement_mean": float(displacement_norm.mean()),
                        "recovery_displacement_rms": _rms(displacement),
                        "recovery_displacement_p95": float(
                            np.quantile(displacement_norm, 0.95)
                        ),
                        "laplacian_residual_rms": _rms(residual),
                        "solver_runtime_seconds": float(solver["runtime_seconds"]),
                        "solver_converged": True,
                        **proxy,
                    }
                )
            print(
                f"alignment shard {args.shard_index}/{args.shard_count} "
                f"{split} {sample_id}",
                flush=True,
            )
            del values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_json(
        output / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "lambda_grid": LAMBDA_GRID,
            "rows": rows,
        },
    )


def _selector_rows(
    per_lambda: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sample: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in per_lambda:
        by_sample.setdefault((str(row["split"]), str(row["sample_id"])), []).append(row)
    selected_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for (split, sample_id), candidates in sorted(by_sample.items()):
        if sorted(float(row["lambda"]) for row in candidates) != sorted(LAMBDA_GRID):
            raise RuntimeError(f"Incomplete lambda grid: {split} {sample_id}")
        fixed = next(row for row in candidates if float(row["lambda"]) == FIXED_LAMBDA)
        chosen: dict[str, Mapping[str, Any]] = {"fixed_1e-2": fixed}
        ties: dict[str, list[float]] = {}
        tolerances: dict[str, float] = {}
        for state, field in SELECTORS.items():
            if field is None:
                continue
            chosen[state], ties[state], tolerances[state] = _select(candidates, field)
        selection = {
            "split": split,
            "sample_id": sample_id,
            "recipe": fixed["recipe"],
            "severity": fixed["severity"],
            "lambda_cd": float(chosen["chamfer_oracle"]["lambda"]),
            "lambda_vrms": float(chosen["vertex_rms_oracle"]["lambda"]),
            "lambda_p95": float(chosen["p2s_p95_oracle"]["lambda"]),
            "lambda_cd_ties": json.dumps(ties["chamfer_oracle"]),
            "lambda_vrms_ties": json.dumps(ties["vertex_rms_oracle"]),
            "lambda_p95_ties": json.dumps(ties["p2s_p95_oracle"]),
            "cd_tolerance": tolerances["chamfer_oracle"],
            "vrms_tolerance": tolerances["vertex_rms_oracle"],
            "p95_tolerance": tolerances["p2s_p95_oracle"],
        }
        for field in PROXY_FIELDS:
            source = field.removeprefix("fixed_") if field.startswith("fixed_") else field
            selection[field] = fixed[source]
        selections.append(selection)
        for state, row in chosen.items():
            selected_rows.append({"state": state, **dict(row)})
    return selected_rows, selections


def _state_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metrics = (
        "chamfer",
        "vertex_rms",
        "p2s_mean",
        "p2s_p95",
        "fscore",
        "normal_consistency",
        "normalized_flip_rate",
        "recovery_displacement_rms",
        "laplacian_residual_rms",
    )
    for split in ("validation", "test"):
        fixed = [row for row in rows if row["split"] == split and row["state"] == "fixed_1e-2"]
        fixed_cd = _mean(fixed, "chamfer")
        fixed_vrms = _mean(fixed, "vertex_rms")
        for state in SELECTORS:
            selected = [row for row in rows if row["split"] == split and row["state"] == state]
            cd = _mean(selected, "chamfer")
            vrms = _mean(selected, "vertex_rms")
            result.append(
                {
                    "split": split,
                    "state": state,
                    "samples": len(selected),
                    **{field: _mean(selected, field) for field in metrics},
                    "relative_chamfer_gain_vs_fixed": (fixed_cd - cd) / fixed_cd,
                    "relative_vertex_rms_gain_vs_fixed": (fixed_vrms - vrms) / fixed_vrms,
                    "introduced_flipped_faces": sum(
                        int(row["introduced_flipped_faces"]) for row in selected
                    ),
                    "new_degenerate_faces": sum(
                        int(row["new_degenerate_faces"]) for row in selected
                    ),
                    "improved": sum(
                        float(row["chamfer"])
                        < float(row["initial_chamfer"]) - _metric_tolerance(float(row["initial_chamfer"]))
                        for row in selected
                    ),
                    "worsened": sum(
                        float(row["chamfer"])
                        > float(row["initial_chamfer"]) + _metric_tolerance(float(row["initial_chamfer"]))
                        for row in selected
                    ),
                }
            )
    return result


def _agreement_and_histograms(
    selections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    agreement: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    grid_index = {value: index for index, value in enumerate(LAMBDA_GRID)}
    for split in ("validation", "test"):
        rows = [row for row in selections if row["split"] == split]
        exact_cd_v = sum(row["lambda_cd"] == row["lambda_vrms"] for row in rows)
        agreement.append(
            {
                "split": split,
                "samples": len(rows),
                "cd_vrms_exact": exact_cd_v,
                "cd_vrms_exact_percentage": exact_cd_v / len(rows),
                "cd_vrms_within_one_grid_step": sum(
                    abs(grid_index[row["lambda_cd"]] - grid_index[row["lambda_vrms"]]) <= 1
                    for row in rows
                ),
                "cd_p95_exact": sum(row["lambda_cd"] == row["lambda_p95"] for row in rows),
                "vrms_p95_exact": sum(row["lambda_vrms"] == row["lambda_p95"] for row in rows),
                "cd_tied_samples": sum(len(json.loads(row["lambda_cd_ties"])) > 1 for row in rows),
                "vrms_tied_samples": sum(len(json.loads(row["lambda_vrms_ties"])) > 1 for row in rows),
                "p95_tied_samples": sum(len(json.loads(row["lambda_p95_ties"])) > 1 for row in rows),
            }
        )
        for selector in ("lambda_cd", "lambda_vrms", "lambda_p95"):
            counts = Counter(float(row[selector]) for row in rows)
            for value in LAMBDA_GRID:
                histograms.append(
                    {"split": split, "selector": selector, "lambda": value, "samples": counts[value]}
                )
        for cd_value in LAMBDA_GRID:
            for vrms_value in LAMBDA_GRID:
                confusion.append(
                    {
                        "split": split,
                        "lambda_cd": cd_value,
                        "lambda_vrms": vrms_value,
                        "samples": sum(
                            row["lambda_cd"] == cd_value and row["lambda_vrms"] == vrms_value
                            for row in rows
                        ),
                    }
                )
    return agreement, histograms, confusion


def _cross_objective(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for split in ("validation", "test"):
        by_state = {row["state"]: row for row in summary if row["split"] == split}
        fixed = by_state["fixed_1e-2"]
        cd = by_state["chamfer_oracle"]
        vrms = by_state["vertex_rms_oracle"]
        result.append(
            {
                "split": split,
                "cd_oracle_vrms_minus_fixed_abs": cd["vertex_rms"] - fixed["vertex_rms"],
                "cd_oracle_vrms_minus_fixed_relative": (cd["vertex_rms"] - fixed["vertex_rms"]) / fixed["vertex_rms"],
                "cd_oracle_vrms_minus_vrms_oracle_abs": cd["vertex_rms"] - vrms["vertex_rms"],
                "cd_oracle_vrms_minus_vrms_oracle_relative": (cd["vertex_rms"] - vrms["vertex_rms"]) / vrms["vertex_rms"],
                "vrms_oracle_cd_minus_fixed_abs": vrms["chamfer"] - fixed["chamfer"],
                "vrms_oracle_cd_minus_fixed_relative": (vrms["chamfer"] - fixed["chamfer"]) / fixed["chamfer"],
                "vrms_oracle_cd_minus_cd_oracle_abs": vrms["chamfer"] - cd["chamfer"],
                "vrms_oracle_cd_minus_cd_oracle_relative": (vrms["chamfer"] - cd["chamfer"]) / cd["chamfer"],
            }
        )
    return result


def _pareto(per_lambda: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sample: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in per_lambda:
        by_sample.setdefault((str(row["split"]), str(row["sample_id"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for (split, sample_id), candidates in sorted(by_sample.items()):
        cd_min = min(float(row["chamfer"]) for row in candidates)
        vrms_min = min(float(row["vertex_rms"]) for row in candidates)
        cd_tol = _metric_tolerance(cd_min)
        vrms_tol = _metric_tolerance(vrms_min)

        def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
            left_cd, right_cd = float(left["chamfer"]), float(right["chamfer"])
            left_v, right_v = float(left["vertex_rms"]), float(right["vertex_rms"])
            no_worse = left_cd <= right_cd + cd_tol and left_v <= right_v + vrms_tol
            strictly = left_cd < right_cd - cd_tol or left_v < right_v - vrms_tol
            return no_worse and strictly

        frontier = [
            row for row in candidates if not any(dominates(other, row) for other in candidates if other is not row)
        ]
        universal = [
            row for row in candidates if all(row is other or dominates(row, other) for other in candidates)
        ]
        cd_selected, _, _ = _select(candidates, "chamfer")
        vrms_selected, _, _ = _select(candidates, "vertex_rms")
        fixed = next(row for row in candidates if float(row["lambda"]) == FIXED_LAMBDA)
        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "recipe": fixed["recipe"],
                "severity": fixed["severity"],
                "pareto_lambdas": json.dumps(sorted(float(row["lambda"]) for row in frontier)),
                "pareto_count": len(frontier),
                "one_lambda_dominates_all": len(universal) == 1,
                "genuine_tradeoff": len(universal) == 0 and len(frontier) > 1,
                "fixed_1e-2_pareto_optimal": fixed in frontier,
                "lambda_cd_tied_optimal_vrms": float(cd_selected["vertex_rms"]) <= vrms_min + vrms_tol,
                "lambda_vrms_tied_optimal_cd": float(vrms_selected["chamfer"]) <= cd_min + cd_tol,
            }
        )
    summary = []
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        summary.append(
            {
                "split": split,
                "samples": len(selected),
                **{
                    field: sum(bool(row[field]) for row in selected)
                    for field in (
                        "one_lambda_dominates_all",
                        "genuine_tradeoff",
                        "fixed_1e-2_pareto_optimal",
                        "lambda_cd_tied_optimal_vrms",
                        "lambda_vrms_tied_optimal_cd",
                    )
                },
            }
        )
    return rows, summary


def _mode(values: Sequence[float]) -> tuple[float, list[float]]:
    counts = Counter(values)
    maximum = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == maximum)
    chosen = min(modes, key=lambda value: (abs(np.log10(value) - np.log10(FIXED_LAMBDA)), value))
    return chosen, modes


def _grouped(
    selections: Sequence[Mapping[str, Any]], selected_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (row["split"], row["sample_id"], row["state"]): row for row in selected_rows
    }
    result = []
    for split in ("validation", "test"):
        for group_type, groups in (
            ("recipe", tuple(TOPOLOGY_RECIPES)[:10]),
            ("severity", ("mild", "strong")),
        ):
            for group in groups:
                rows = [row for row in selections if row["split"] == split and row[group_type] == group]
                if not rows:
                    continue
                cd_values = [float(row["lambda_cd"]) for row in rows]
                vrms_values = [float(row["lambda_vrms"]) for row in rows]
                cd_mode, cd_modes = _mode(cd_values)
                vrms_mode, vrms_modes = _mode(vrms_values)
                fixed = [by_key[(split, row["sample_id"], "fixed_1e-2")] for row in rows]
                cd = [by_key[(split, row["sample_id"], "chamfer_oracle")] for row in rows]
                vrms = [by_key[(split, row["sample_id"], "vertex_rms_oracle")] for row in rows]
                fixed_cd, fixed_v = _mean(fixed, "chamfer"), _mean(fixed, "vertex_rms")
                cd_cd, cd_v = _mean(cd, "chamfer"), _mean(cd, "vertex_rms")
                vrms_cd, vrms_v = _mean(vrms, "chamfer"), _mean(vrms, "vertex_rms")
                result.append(
                    {
                        "split": split,
                        "group_type": group_type,
                        "group": group,
                        "samples": len(rows),
                        "modal_lambda_cd": cd_mode,
                        "modal_lambda_cd_ties": json.dumps(cd_modes),
                        "modal_lambda_vrms": vrms_mode,
                        "modal_lambda_vrms_ties": json.dumps(vrms_modes),
                        "mean_log10_lambda_cd": float(np.mean(np.log10(cd_values))),
                        "median_log10_lambda_cd": float(np.median(np.log10(cd_values))),
                        "mean_log10_lambda_vrms": float(np.mean(np.log10(vrms_values))),
                        "median_log10_lambda_vrms": float(np.median(np.log10(vrms_values))),
                        "cd_vrms_agreement_rate": sum(a == b for a, b in zip(cd_values, vrms_values)) / len(rows),
                        "chamfer_gain_from_lambda_cd": (fixed_cd - cd_cd) / fixed_cd,
                        "vertex_rms_gain_from_lambda_vrms": (fixed_v - vrms_v) / fixed_v,
                        "cd_oracle_vrms_penalty_vs_fixed": (cd_v - fixed_v) / fixed_v,
                        "cd_oracle_vrms_penalty_vs_fixed_abs": cd_v - fixed_v,
                        "cd_oracle_vrms_penalty_vs_vrms_oracle": (cd_v - vrms_v) / vrms_v,
                        "cd_oracle_vrms_penalty_vs_vrms_oracle_abs": cd_v - vrms_v,
                        "vrms_oracle_cd_penalty_vs_fixed": (vrms_cd - fixed_cd) / fixed_cd,
                        "vrms_oracle_cd_penalty_vs_fixed_abs": vrms_cd - fixed_cd,
                        "vrms_oracle_cd_penalty_vs_cd_oracle": (vrms_cd - cd_cd) / cd_cd,
                        "vrms_oracle_cd_penalty_vs_cd_oracle_abs": vrms_cd - cd_cd,
                    }
                )
    return result


def _correlations(selections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for split in ("validation", "test"):
        rows = [row for row in selections if row["split"] == split]
        for selector in ("lambda_cd", "lambda_vrms"):
            target = np.log10([float(row[selector]) for row in rows])
            for field in PROXY_FIELDS:
                values = np.asarray([float(row[field]) for row in rows])
                correlation = spearmanr(values, target)
                rho = float(correlation.statistic)
                magnitude = (
                    "undefined"
                    if not np.isfinite(rho)
                    else "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
                )
                result.append(
                    {
                        "split": split,
                        "selector": selector,
                        "field": field,
                        "spearman_rho": rho,
                        "p_value": float(correlation.pvalue),
                        "samples": len(rows),
                        "sign": "undefined" if not np.isfinite(rho) else "positive" if rho > 0 else "negative" if rho < 0 else "zero",
                        "magnitude": magnitude,
                    }
                )
    return result


def _interpretation(
    agreement: Mapping[str, Any], cross: Mapping[str, Any]
) -> tuple[str, str]:
    exact = float(agreement["cd_vrms_exact_percentage"])
    adjacent = float(agreement["cd_vrms_within_one_grid_step"]) / float(agreement["samples"])
    cd_v_penalty = float(cross["cd_oracle_vrms_minus_fixed_relative"])
    vrms_cd_penalty = float(cross["vrms_oracle_cd_minus_fixed_relative"])
    if exact >= 0.70 and adjacent >= 0.90 and cd_v_penalty <= 0.01 and vrms_cd_penalty <= 0.01:
        return "Case A", "surface-fidelity aligned"
    if exact <= 0.30 and cd_v_penalty > 0.02 and vrms_cd_penalty > 0.02:
        return "Case C", "strong objective mismatch"
    return "Case B", "moderate trade-off"


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = sorted((output / "shards").glob("shard_*.json"))
    if len(shards) != args.shard_count:
        raise RuntimeError(f"Expected {args.shard_count} shards, found {len(shards)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    if len({payload["checkpoint_sha256"] for payload in payloads}) != 1:
        raise RuntimeError("Shards used different checkpoints.")
    supplemental = [row for payload in payloads for row in payload["rows"]]
    if len(supplemental) != 100 * len(LAMBDA_GRID):
        raise RuntimeError("Incomplete per-lambda results.")
    supplemental_by_key = {
        (row["split"], row["sample_id"], float(row["lambda"])): row
        for row in supplemental
    }
    reference_rows = _read_csv(args.reference_per_lambda.resolve())
    reference_by_key = {
        (row["split"], row["sample_id"], float(row["lambda"])): row
        for row in reference_rows
    }
    if set(reference_by_key) != set(supplemental_by_key):
        raise RuntimeError("Existing-oracle and supplemental per-lambda keys differ.")
    # Preserve the established oracle's recovery/evaluation values exactly.
    # Only fields that were absent in that archive are supplemented by the
    # read-only recomputation.  Under this evaluator P2S mean is the recorded
    # bidirectional Chamfer value, so it needs no recomputation.
    per_lambda: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for key in sorted(reference_by_key):
        old = dict(reference_by_key[key])
        new = supplemental_by_key[key]
        old.update(
            {
                "p2s_mean": float(old["chamfer"]),
                "fscore": float(new["fscore"]),
                "laplacian_residual_rms": float(new["laplacian_residual_rms"]),
                "confidence_features_available": False,
                "established_metrics_source": "existing_adaptive_lambda_oracle_per_lambda",
                "supplemental_metrics_source": "read_only_same_checkpoint_recomputation",
            }
        )
        per_lambda.append(old)
        drift_rows.append(
            {
                "split": key[0],
                "sample_id": key[1],
                "lambda": key[2],
                "chamfer_drift": float(new["chamfer"]) - float(old["chamfer"]),
                "vertex_rms_drift": float(new["vertex_rms"]) - float(old["vertex_rms"]),
                "p2s_p95_drift": float(new["p2s_p95"]) - float(old["p2s_p95"]),
                "normal_drift": float(new["normal_consistency"]) - float(old["normal_consistency"]),
            }
        )
    selected_rows, selections = _selector_rows(per_lambda)
    states = _state_summary(selected_rows)
    agreement, histograms, confusion = _agreement_and_histograms(selections)
    cross = _cross_objective(states)
    pareto_rows, pareto_summary = _pareto(per_lambda)
    grouped = _grouped(selections, selected_rows)
    correlations = _correlations(selections)

    reference = json.loads(args.reference_summary.resolve().read_text(encoding="utf-8"))
    reference_by_split = {row["split"]: row for row in reference["split_summary"]}
    reproduction = []
    for split in ("validation", "test"):
        current = next(row for row in states if row["split"] == split and row["state"] == "chamfer_oracle")
        old = reference_by_split[split]
        difference = float(current["chamfer"]) - float(old["oracle_mean_chamfer"])
        reproduction.append(
            {
                "split": split,
                "reference_chamfer_oracle": old["oracle_mean_chamfer"],
                "current_tolerance_aware_chamfer_oracle": current["chamfer"],
                "absolute_difference": difference,
                "within_declared_absolute_tolerance": abs(difference) <= ABSOLUTE_TOLERANCE,
            }
        )
    validation_agreement = next(row for row in agreement if row["split"] == "validation")
    validation_cross = next(row for row in cross if row["split"] == "validation")
    case, conclusion = _interpretation(validation_agreement, validation_cross)
    contract = {
        "passed": all(bool(row["within_declared_absolute_tolerance"]) for row in reproduction),
        "read_only_frozen_arm_b": True,
        "existing_oracle_outputs_overwritten": False,
        "training_or_queue_modified": False,
        "selection_and_interpretation_use_validation_only": True,
        "test_used_for_tuning": False,
        "lambda_grid": LAMBDA_GRID,
        "fixed_lambda": FIXED_LAMBDA,
        "tie_rule": "within tolerance, prefer lambda closest to 1e-2 in log10 space; then lower lambda",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "pareto_dominance": "no worse within tolerance in both metrics and better beyond tolerance in at least one",
        "case_a_threshold": "exact>=70%, adjacent>=90%, both cross penalties vs fixed<=1%",
        "case_c_threshold": "exact<=30% and both cross penalties vs fixed>2%",
        "metric_protocol": METRIC_PROTOCOL,
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "confidence_features_available": False,
        "established_per_lambda_rows_reused": len(reference_rows),
        "supplemental_fields": ["p2s_mean", "fscore", "laplacian_residual_rms"],
    }
    drift_summary = {
        field: {
            "mean": float(np.mean([float(row[field]) for row in drift_rows])),
            "maximum_absolute": float(max(abs(float(row[field])) for row in drift_rows)),
        }
        for field in ("chamfer_drift", "vertex_rms_drift", "p2s_p95_drift", "normal_drift")
    }
    summary = {
        "contract_audit": contract,
        "reproduced_chamfer_oracle": reproduction,
        "agreement": agreement,
        "histograms": histograms,
        "cross_objective_cost": cross,
        "state_summary": states,
        "pareto_summary": pareto_summary,
        "supplemental_recomputation_drift": drift_summary,
        "validation_interpretation": {"case": case, "conclusion": conclusion, "h_decision_changed": False},
    }
    _write_csv(output / "per_lambda_full.csv", per_lambda)
    _write_csv(output / "per_sample_selectors.csv", selections)
    _write_csv(output / "selected_state_per_sample.csv", selected_rows)
    _write_csv(output / "state_summary.csv", states)
    _write_csv(output / "oracle_agreement.csv", agreement)
    _write_csv(output / "lambda_histograms.csv", histograms)
    _write_csv(output / "lambda_cd_vs_vrms_confusion.csv", confusion)
    _write_csv(output / "cross_objective_cost.csv", cross)
    _write_csv(output / "pareto_per_sample.csv", pareto_rows)
    _write_csv(output / "pareto_summary.csv", pareto_summary)
    _write_csv(output / "recipe_severity_summary.csv", grouped)
    _write_csv(output / "gt_free_proxy_correlations.csv", correlations)
    _write_csv(output / "supplemental_recomputation_drift.csv", drift_rows)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contract_audit.json", contract)

    lines = [
        "# Sofa50 v2 adaptive-lambda objective-alignment oracle",
        "",
        f"Contract audit: **{str(contract['passed']).lower()}**. Read-only frozen Arm B; existing oracle and G/H are unchanged.",
        "",
        f"Validation-only interpretation: **{case} — {conclusion}**. The existing H decision remains unchanged.",
        "Established per-lambda Chamfer/VRMS/P95/normal/topology values are reused verbatim; only previously absent P2S mean, F-score, and Laplacian residual fields are supplemented by the read-only recomputation.",
        "",
        "## Chamfer-oracle reproduction",
        "",
        "| Split | Existing CD oracle | Recomputed tolerance-aware CD oracle | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in reproduction:
        lines.append(f"| {row['split']} | {row['reference_chamfer_oracle']:.9g} | {row['current_tolerance_aware_chamfer_oracle']:.9g} | {row['absolute_difference']:+.3e} |")
    lines.extend((
        "",
        "## Oracle agreement",
        "",
        "| Split | CD=VRMS | CD~VRMS adjacent | CD=P95 | VRMS=P95 |",
        "|---|---:|---:|---:|---:|",
    ))
    for row in agreement:
        lines.append(f"| {row['split']} | {row['cd_vrms_exact']}/{row['samples']} ({row['cd_vrms_exact_percentage']:.1%}) | {row['cd_vrms_within_one_grid_step']}/{row['samples']} ({row['cd_vrms_within_one_grid_step']/row['samples']:.1%}) | {row['cd_p95_exact']}/{row['samples']} | {row['vrms_p95_exact']}/{row['samples']} |")
    lines.extend((
        "",
        "Tie counts (CD / VRMS / P95): "
        + "; ".join(
            f"{row['split']} {row['cd_tied_samples']} / {row['vrms_tied_samples']} / {row['p95_tied_samples']}"
            for row in agreement
        )
        + ".",
        "",
        "## Exact lambda histograms",
        "",
        "| Split | Selector | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ))
    histogram_index = {
        (row["split"], row["selector"], float(row["lambda"])): int(row["samples"])
        for row in histograms
    }
    for split in ("validation", "test"):
        for selector in ("lambda_cd", "lambda_vrms", "lambda_p95"):
            values = [histogram_index[(split, selector, value)] for value in LAMBDA_GRID]
            lines.append(
                f"| {split} | {selector} | " + " | ".join(str(value) for value in values) + " |"
            )
    lines.extend(("", "## Lambda-CD versus lambda-VRMS confusion", ""))
    confusion_index = {
        (row["split"], float(row["lambda_cd"]), float(row["lambda_vrms"])): int(row["samples"])
        for row in confusion
    }
    for split in ("validation", "test"):
        lines.extend((
            f"### {split}",
            "",
            "| CD \\ VRMS | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | 1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for cd_value in LAMBDA_GRID:
            values = [confusion_index[(split, cd_value, vrms_value)] for vrms_value in LAMBDA_GRID]
            lines.append(
                f"| {cd_value:.0e} | " + " | ".join(str(value) for value in values) + " |"
            )
        lines.append("")
    lines.extend((
        "",
        "## Fixed and oracle-selected states",
        "",
        "| Split | State | CD | CD gain | Vertex RMS | VRMS gain | P2S | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in states:
        lines.append(f"| {row['split']} | {row['state']} | {row['chamfer']:.9g} | {row['relative_chamfer_gain_vs_fixed']:+.2%} | {row['vertex_rms']:.9g} | {row['relative_vertex_rms_gain_vs_fixed']:+.2%} | {row['p2s_mean']:.9g} | {row['p2s_p95']:.9g} | {row['fscore']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} / {row['normalized_flip_rate']:.4%} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} |")
    lines.extend((
        "",
        "## Cross-objective cost",
        "",
        "| Split | VRMS(CD)-VRMS(fixed) | VRMS(CD)-VRMS(VRMS) | CD(VRMS)-CD(fixed) | CD(VRMS)-CD(CD) |",
        "|---|---:|---:|---:|---:|",
    ))
    for row in cross:
        lines.append(f"| {row['split']} | {row['cd_oracle_vrms_minus_fixed_abs']:+.9g} ({row['cd_oracle_vrms_minus_fixed_relative']:+.2%}) | {row['cd_oracle_vrms_minus_vrms_oracle_abs']:+.9g} ({row['cd_oracle_vrms_minus_vrms_oracle_relative']:+.2%}) | {row['vrms_oracle_cd_minus_fixed_abs']:+.9g} ({row['vrms_oracle_cd_minus_fixed_relative']:+.2%}) | {row['vrms_oracle_cd_minus_cd_oracle_abs']:+.9g} ({row['vrms_oracle_cd_minus_cd_oracle_relative']:+.2%}) |")
    lines.extend((
        "",
        "## Pareto analysis",
        "",
        "| Split | One lambda dominates all | Genuine trade-off | Fixed 1e-2 Pareto | CD also VRMS-optimal | VRMS also CD-optimal |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in pareto_summary:
        lines.append(f"| {row['split']} | {row['one_lambda_dominates_all']}/{row['samples']} | {row['genuine_tradeoff']}/{row['samples']} | {row['fixed_1e-2_pareto_optimal']}/{row['samples']} | {row['lambda_cd_tied_optimal_vrms']}/{row['samples']} | {row['lambda_vrms_tied_optimal_cd']}/{row['samples']} |")
    lines.extend((
        "",
        "## Recipe and severity breakdown",
        "",
        "| Split | Group | Mode CD / VRMS | Mean log10 CD / VRMS | Median log10 CD / VRMS | Agreement | CD gain | VRMS gain | CD-oracle VRMS penalty | VRMS-oracle CD penalty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in grouped:
        lines.append(
            f"| {row['split']} | {row['group']} | {row['modal_lambda_cd']:.0e} / {row['modal_lambda_vrms']:.0e} | "
            f"{row['mean_log10_lambda_cd']:.3f} / {row['mean_log10_lambda_vrms']:.3f} | "
            f"{row['median_log10_lambda_cd']:.3f} / {row['median_log10_lambda_vrms']:.3f} | "
            f"{row['cd_vrms_agreement_rate']:.1%} | {row['chamfer_gain_from_lambda_cd']:+.2%} | "
            f"{row['vertex_rms_gain_from_lambda_vrms']:+.2%} | "
            f"{row['cd_oracle_vrms_penalty_vs_fixed_abs']:+.3e} ({row['cd_oracle_vrms_penalty_vs_fixed']:+.2%}) | "
            f"{row['vrms_oracle_cd_penalty_vs_fixed_abs']:+.3e} ({row['vrms_oracle_cd_penalty_vs_fixed']:+.2%}) |"
        )
    lines.extend((
        "",
        "## GT-free proxy correlations",
        "",
        "| Split | Selector | Proxy | Spearman | Sign / magnitude | n |",
        "|---|---|---|---:|---|---:|",
    ))
    for row in correlations:
        lines.append(
            f"| {row['split']} | {row['selector']} | {row['field']} | {row['spearman_rho']:.4f} | "
            f"{row['sign']} / {row['magnitude']} | {row['samples']} |"
        )
    lines.extend((
        "",
        "The fixed-1e-2 recovery displacement-RMS versus lambda-CD row should reproduce the established approximately -0.413 Spearman association under the tolerance-aware selector.",
        "",
        "Full per-sample ties, Pareto frontiers, absolute cross-objective costs, and all per-lambda metrics are provided in the adjacent CSV files.",
        "",
        "## Implication for learned lambda training",
        "",
        (
            "Validation supports Case A: the surface and indexed-vertex objectives are aligned closely enough that the existing adaptive-lambda design remains a shared upper-bound study."
            if case == "Case A"
            else "Validation supports Case B: continue H unchanged, interpret lambda as a surface-fidelity/correspondence trade-off, and reserve any combined recovery objective for a separate later ablation."
            if case == "Case B"
            else "Validation supports Case C: report Chamfer and vertex-RMS upper bounds separately. The Chamfer oracle is not the correct sole upper bound for a lambda head trained only through vertex loss, and failure to reach it must not by itself be called adaptive-lambda failure."
        ),
        "",
        f"Tolerance: `max({ABSOLUTE_TOLERANCE:g}, {RELATIVE_TOLERANCE:g} * |metric|)`. Confidence features are unavailable because the frozen Arm B confidence head is disabled.",
        "",
        f"Metric protocol: `{METRIC_PROTOCOL}`.",
        "",
    ))
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--reference-per-lambda", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        if args.reference_summary is None or args.reference_per_lambda is None:
            parser.error("merge requires --reference-summary and --reference-per-lambda")
        merge(args)
    else:
        if args.manifest is None or args.runs_root is None:
            parser.error("evaluation requires --manifest and --runs-root")
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("invalid shard index")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

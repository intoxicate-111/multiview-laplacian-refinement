#!/usr/bin/env python3
from __future__ import annotations

"""Read-only per-mesh recovery-lambda oracle for the frozen Sofa50-v2 Arm B."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

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


LAMBDA_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)
FIXED_LAMBDA = 1e-2
ARM_B_RUN = "sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _recipe(sample_id: str) -> str:
    recipe = sample_id.rsplit("__", 1)[-1]
    if recipe not in TOPOLOGY_RECIPES:
        raise ValueError(f"Unknown coarse recipe in sample id: {sample_id}")
    return recipe


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(vectors), axis=1))))


def evaluate_shard(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    run = args.runs_root.resolve() / ARM_B_RUN
    spec = _load_spec(run, device)
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
                torch.as_tensor(static["vertices"]).cpu().numpy(),
                torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
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
            proxy = {
                "predicted_laplacian_rms": _rms(prediction),
                "predicted_correction_rms": _rms(prediction - initial_laplacian),
                "predicted_correction_mean": float(
                    np.linalg.norm(prediction - initial_laplacian, axis=1).mean()
                ),
                "predicted_correction_p95": float(
                    np.quantile(
                        np.linalg.norm(prediction - initial_laplacian, axis=1), 0.95
                    )
                ),
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
                displacement = recovered - initial.vertices
                refined = Mesh(recovered, initial.faces.copy()).ensure_normals()
                geometry = _geometry_row(
                    "v2_strong_smoothing",
                    sample_id,
                    f"lambda_{regularization:.0e}",
                    refined,
                    clean,
                    initial,
                )
                row = {
                    "split": split,
                    "sample_id": sample_id,
                    "recipe": _recipe(sample_id),
                    "severity": TOPOLOGY_RECIPES[_recipe(sample_id)]["degradation"],
                    "lambda": regularization,
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "initial_chamfer": float(initial_geometry["chamfer"]),
                    "chamfer": float(geometry["chamfer"]),
                    "p2s_p95": float(geometry["p2s_p95"]),
                    "normal_consistency": float(geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(
                        geometry["introduced_flipped_faces"]
                    ),
                    "normalized_flip_rate": float(
                        geometry["introduced_flipped_faces"] / initial.num_faces
                    ),
                    "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                    "vertex_rms": _rms(recovered - clean.vertices),
                    "recovery_displacement_mean": float(
                        np.linalg.norm(displacement, axis=1).mean()
                    ),
                    "recovery_displacement_rms": _rms(displacement),
                    "recovery_displacement_p95": float(
                        np.quantile(np.linalg.norm(displacement, axis=1), 0.95)
                    ),
                    "solver_runtime_seconds": float(solver["runtime_seconds"]),
                    "solver_converged": bool(solver["all_converged"]),
                    **proxy,
                }
                rows.append(row)
            print(
                f"shard {args.shard_index}/{args.shard_count} {split} "
                f"{sample_id}",
                flush=True,
            )
            del values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_json(
        args.output_dir.resolve() / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "lambda_grid": LAMBDA_GRID,
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "rows": rows,
        },
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _correlations(oracle_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "predicted_laplacian_rms",
        "predicted_correction_rms",
        "predicted_correction_mean",
        "predicted_correction_p95",
        "fixed_displacement_mean",
        "fixed_displacement_rms",
        "fixed_displacement_p95",
    )
    target = np.log10([float(row["lambda_oracle"]) for row in oracle_rows])
    result = []
    for field in fields:
        values = np.asarray([float(row[field]) for row in oracle_rows])
        correlation = spearmanr(values, target)
        result.append(
            {
                "field": field,
                "spearman_rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
                "samples": len(values),
            }
        )
    return result


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = sorted((output / "shards").glob("shard_*.json"))
    if len(shards) != args.shard_count:
        raise RuntimeError(f"Expected {args.shard_count} shards, found {len(shards)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    if len({payload["checkpoint_sha256"] for payload in payloads}) != 1:
        raise RuntimeError("Oracle shards used different checkpoints.")
    rows = [row for payload in payloads for row in payload["rows"]]
    expected = 100 * len(LAMBDA_GRID)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    if not all(bool(row["solver_converged"]) for row in rows):
        raise RuntimeError("At least one sparse solve did not converge.")
    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault((row["split"], row["sample_id"]), []).append(row)
    oracle_rows: list[dict[str, Any]] = []
    for (split, sample_id), candidates in sorted(by_sample.items()):
        if len(candidates) != len(LAMBDA_GRID):
            raise RuntimeError(f"Incomplete lambda grid for {sample_id}")
        oracle = min(candidates, key=lambda row: (float(row["chamfer"]), float(row["lambda"])))
        fixed = next(row for row in candidates if float(row["lambda"]) == FIXED_LAMBDA)
        oracle_rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "recipe": oracle["recipe"],
                "severity": oracle["severity"],
                "lambda_oracle": oracle["lambda"],
                "oracle_chamfer": oracle["chamfer"],
                "oracle_p2s_p95": oracle["p2s_p95"],
                "oracle_vertex_rms": oracle["vertex_rms"],
                "fixed_chamfer": fixed["chamfer"],
                "fixed_p2s_p95": fixed["p2s_p95"],
                "fixed_vertex_rms": fixed["vertex_rms"],
                "oracle_cd_improvement": float(fixed["chamfer"])
                - float(oracle["chamfer"]),
                "oracle_cd_relative_improvement": (
                    float(fixed["chamfer"]) - float(oracle["chamfer"])
                )
                / float(fixed["chamfer"]),
                "predicted_laplacian_rms": fixed["predicted_laplacian_rms"],
                "predicted_correction_rms": fixed["predicted_correction_rms"],
                "predicted_correction_mean": fixed["predicted_correction_mean"],
                "predicted_correction_p95": fixed["predicted_correction_p95"],
                "fixed_displacement_mean": fixed["recovery_displacement_mean"],
                "fixed_displacement_rms": fixed["recovery_displacement_rms"],
                "fixed_displacement_p95": fixed["recovery_displacement_p95"],
            }
        )
    split_summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    correlations: dict[str, Any] = {}
    for split in ("validation", "test"):
        selected = [row for row in oracle_rows if row["split"] == split]
        counts = Counter(float(row["lambda_oracle"]) for row in selected)
        fixed_cd = _mean(selected, "fixed_chamfer")
        oracle_cd = _mean(selected, "oracle_chamfer")
        split_summaries.append(
            {
                "split": split,
                "samples": len(selected),
                "fixed_mean_chamfer": fixed_cd,
                "oracle_mean_chamfer": oracle_cd,
                "oracle_relative_mean_cd_gain": (fixed_cd - oracle_cd) / fixed_cd,
                "fixed_mean_p2s_p95": _mean(selected, "fixed_p2s_p95"),
                "oracle_mean_p2s_p95": _mean(selected, "oracle_p2s_p95"),
                "fixed_mean_vertex_rms": _mean(selected, "fixed_vertex_rms"),
                "oracle_mean_vertex_rms": _mean(selected, "oracle_vertex_rms"),
                "lambda_lt_1e-2": sum(float(row["lambda_oracle"]) < 1e-2 for row in selected),
                "lambda_eq_1e-2": sum(float(row["lambda_oracle"]) == 1e-2 for row in selected),
                "lambda_gt_1e-2": sum(float(row["lambda_oracle"]) > 1e-2 for row in selected),
            }
        )
        for value in LAMBDA_GRID:
            distributions.append(
                {"split": split, "lambda": value, "samples": counts[value]}
            )
        for group_type, values in (
            ("recipe", tuple(TOPOLOGY_RECIPES)[:10]),
            ("severity", ("mild", "strong")),
        ):
            for value in values:
                subset = [row for row in selected if row[group_type] == value]
                if not subset:
                    continue
                grouped.append(
                    {
                        "split": split,
                        "group_type": group_type,
                        "group": value,
                        "samples": len(subset),
                        "oracle_lambda_median": float(
                            np.median([float(row["lambda_oracle"]) for row in subset])
                        ),
                        "fixed_chamfer": _mean(subset, "fixed_chamfer"),
                        "oracle_chamfer": _mean(subset, "oracle_chamfer"),
                        "oracle_relative_cd_gain": _mean(
                            subset, "oracle_cd_relative_improvement"
                        ),
                        "lambda_distribution": json.dumps(
                            dict(sorted(Counter(float(row["lambda_oracle"]) for row in subset).items()))
                        ),
                    }
                )
        correlations[split] = _correlations(selected)
    validation = next(row for row in split_summaries if row["split"] == "validation")
    validation_rows = [row for row in oracle_rows if row["split"] == "validation"]
    nonfixed = sum(float(row["lambda_oracle"]) != FIXED_LAMBDA for row in validation_rows)
    occupied = sum(
        sum(float(row["lambda_oracle"]) == value for row in validation_rows) >= 5
        for value in LAMBDA_GRID
    )
    # Predeclared validation-only gate. Test values never decide whether H is trained.
    gate = {
        "relative_mean_cd_gain_threshold": 0.01,
        "nonfixed_samples_threshold": 10,
        "occupied_lambda_buckets_threshold": 2,
        "observed_relative_mean_cd_gain": validation["oracle_relative_mean_cd_gain"],
        "observed_nonfixed_samples": nonfixed,
        "observed_occupied_lambda_buckets": occupied,
        "passed": bool(
            validation["oracle_relative_mean_cd_gain"] >= 0.01
            and nonfixed >= 10
            and occupied >= 2
        ),
        "selection_split": "validation_only",
    }
    contract = {
        "passed": True,
        "read_only_frozen_arm_b": True,
        "gt_used_only_for_oracle_selection_and_evaluation": True,
        "gt_free_proxy_fields": [row["field"] for row in correlations["validation"]],
        "lambda_grid": LAMBDA_GRID,
        "lambda_1e-4_excluded": True,
        "metric_protocol": METRIC_PROTOCOL,
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
    }
    summary = {
        "contract_audit": contract,
        "adaptive_training_gate": gate,
        "split_summary": split_summaries,
        "lambda_distribution": distributions,
        "grouped_summary": grouped,
        "proxy_correlations": correlations,
    }
    _write_csv(output / "per_lambda.csv", rows)
    _write_csv(output / "per_sample_oracle.csv", oracle_rows)
    _write_csv(output / "split_summary.csv", split_summaries)
    _write_csv(output / "lambda_distribution.csv", distributions)
    _write_csv(output / "grouped_summary.csv", grouped)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contract_audit.json", contract)
    lines = [
        "# Sofa50 v2 adaptive-lambda oracle diagnostic",
        "",
        "Contract audit: **true**. This is a read-only diagnostic using the frozen Arm B predictor.",
        "",
        f"Adaptive-training validation gate: **{str(gate['passed']).lower()}**.",
        "",
        "| Split | Fixed λ=1e-2 CD | Oracle CD | Relative oracle gain | Oracle P2S p95 | Oracle vertex RMS | <1e-2 / =1e-2 / >1e-2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_summaries:
        lines.append(
            f"| {row['split']} | {row['fixed_mean_chamfer']:.9g} | {row['oracle_mean_chamfer']:.9g} | "
            f"{row['oracle_relative_mean_cd_gain']:.2%} | {row['oracle_mean_p2s_p95']:.9g} | "
            f"{row['oracle_mean_vertex_rms']:.9g} | {row['lambda_lt_1e-2']} / "
            f"{row['lambda_eq_1e-2']} / {row['lambda_gt_1e-2']} |"
        )
    lines.extend(
        (
            "",
            "The learned-lambda continuation is permitted only by the validation gate: at least 1% mean-CD headroom, at least 10/50 non-fixed oracle choices, and at least two lambda buckets occupied by five or more validation samples.",
            "",
            f"Metric protocol: `{METRIC_PROTOCOL}`.",
            "",
        )
    )
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.manifest is None or args.runs_root is None:
            parser.error("evaluation requires --manifest and --runs-root")
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("shard-index must be in [0, shard-count)")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

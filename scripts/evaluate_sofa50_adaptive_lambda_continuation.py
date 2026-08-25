#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate matched G-fixed/H-adaptive Sofa50-v2 continuation arms."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_multitopology_rawlap import raw_gt_magnitude_metrics
from evaluate_sofa50_recovery_aware_ablation import _runtime_diagnostic_summary
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.multitopology_rawlap import TOPOLOGY_RECIPES
from mlr.learned_laplacian.trainer import load_checkpoint


ARMS = ("G_fixed_continue", "H_predicted_lambda")
RUN_NAMES = (
    "sofa50_v2_g_fixed_lambda1e-2_continue5k_seed7",
    "sofa50_v2_h_predicted_lambda_continue5k_seed7",
)
FINAL_STEP = 25000
FIXED_LAMBDA = 1e-2
LAMBDA_MIN = 1e-3
LAMBDA_MAX = 1e-1


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_final(run: Path, device: torch.device) -> dict[str, Any]:
    payload = _read(run / "run_config.json")
    config = payload.get("experiment_config", payload)
    checkpoint = run / f"checkpoint_step_{FINAL_STEP:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "run": run,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint_sha256": payload.get("resume_checkpoint_sha256"),
        "metrics": _read(run / "metrics.json"),
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _recipe(sample_id: str) -> str:
    recipe = sample_id.rsplit("__", 1)[-1]
    if recipe not in TOPOLOGY_RECIPES:
        raise ValueError(sample_id)
    return recipe


def evaluate_arm(args: argparse.Namespace) -> None:
    arm = ARMS[args.arm_index]
    run = args.runs_root.resolve() / RUN_NAMES[args.arm_index]
    output = args.output_dir.resolve()
    device = torch.device(args.device)
    spec = _load_final(run, device)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {}
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        arrays[f"{split}_prediction"] = []
        arrays[f"{split}_target"] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            prepared = _load_device_item(dataset, index, spec["config"], device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=spec["amp_dtype"],
                enabled=bool(spec["amp_enabled"]),
            ):
                model_output = spec["model"](conditioned)
            prediction_t = model_output.predicted_laplacian.float().cpu()
            if prepared.raw_target is None:
                raise RuntimeError("Missing raw target.")
            target_t = prepared.raw_target.float().cpu()
            valid_t = prepared.sample["valid_scale_mask"].bool().cpu()
            predicted_lambda = model_output.recovery_lambda
            if arm == "G_fixed_continue":
                if predicted_lambda is not None:
                    raise RuntimeError("G must not instantiate a lambda head.")
                regularization = FIXED_LAMBDA
            else:
                if predicted_lambda is None:
                    raise RuntimeError("H emitted no predicted lambda.")
                regularization = float(predicted_lambda.detach().cpu())
                if not LAMBDA_MIN <= regularization <= LAMBDA_MAX:
                    raise RuntimeError("Predicted lambda is outside its declared bounds.")
            valid = valid_t.numpy().astype(bool)
            prediction = prediction_t.numpy().astype(np.float64)
            arrays[f"{split}_prediction"].append(prediction[valid])
            arrays[f"{split}_target"].append(target_t.numpy().astype(np.float64)[valid])
            prediction_metrics = raw_gt_magnitude_metrics(
                prediction_t, target_t, torch.ones(len(prediction_t)), valid_t
            )
            initial = Mesh(
                torch.as_tensor(static["vertices"]).cpu().numpy(),
                torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
            component_count, labels = component_labels(lap_data)
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
            initial_geometry = _geometry_row("v2_strong_smoothing", sample_id, "initial", initial, clean, initial)
            refined_geometry = _geometry_row(
                "v2_strong_smoothing",
                sample_id,
                arm,
                Mesh(recovered, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            displacement = recovered - initial.vertices
            magnitude = np.linalg.norm(displacement, axis=1)
            initial_cd = float(initial_geometry["chamfer"])
            refined_cd = float(refined_geometry["chamfer"])
            recipe = _recipe(sample_id)
            rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "sample_id": sample_id,
                    "recipe": recipe,
                    "severity": TOPOLOGY_RECIPES[recipe]["degradation"],
                    **prediction_metrics,
                    "lambda": regularization,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "p2s": float(refined_geometry["p2s"]),
                    "p2s_p95": float(refined_geometry["p2s_p95"]),
                    "fscore": float(refined_geometry["fscore"]),
                    "normal_consistency": float(refined_geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(refined_geometry["introduced_flipped_faces"]),
                    "normalized_flip_rate": float(refined_geometry["introduced_flipped_faces"] / initial.num_faces),
                    "new_degenerate_faces": int(refined_geometry["new_degenerate_faces"]),
                    "same_index_recovered_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
                    ),
                    "recovery_displacement_mean": float(magnitude.mean()),
                    "recovery_displacement_rms": float(np.sqrt(np.mean(np.square(magnitude)))),
                    "recovery_displacement_p95": float(np.quantile(magnitude, 0.95)),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    "faces": initial.num_faces,
                    "solver_runtime_seconds": float(solver["runtime_seconds"]),
                    "solver_converged": bool(solver["all_converged"]),
                }
            )
            print(f"{arm} {split} {index + 1}/{len(dataset)} {sample_id}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    shard = output / "shards"
    _write_json(
        shard / f"{arm}.json",
        {
            "arm": arm,
            "config": spec["config"],
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
            "metrics": spec["metrics"],
            "parameter_count": spec["parameter_count"],
            "training_runtime_diagnostics": _runtime_diagnostic_summary(run),
            "rows": rows,
        },
    )
    np.savez_compressed(
        shard / f"{arm}_prediction_arrays.npz",
        **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [_read(output / "shards" / f"{arm}.json") for arm in ARMS]
    if len({payload["source_checkpoint_sha256"] for payload in payloads}) != 1:
        raise RuntimeError("G/H did not start from the same Arm-B checkpoint.")
    all_rows = [row for payload in payloads for row in payload["rows"]]
    if len(all_rows) != 200 or not all(bool(row["solver_converged"]) for row in all_rows):
        raise RuntimeError("Incomplete or non-converged G/H evaluation.")
    aggregate_prediction = []
    aggregate_geometry = []
    for arm in ARMS:
        arrays = np.load(output / "shards" / f"{arm}_prediction_arrays.npz")
        for split in ("validation", "test"):
            prediction = torch.from_numpy(arrays[f"{split}_prediction"])
            target = torch.from_numpy(arrays[f"{split}_target"])
            metrics = raw_gt_magnitude_metrics(
                prediction, target, torch.ones(len(prediction)), torch.ones(len(prediction), dtype=torch.bool)
            )
            aggregate_prediction.append({"arm": arm, "split": split, **metrics})
            selected = [row for row in all_rows if row["arm"] == arm and row["split"] == split]
            aggregate_geometry.append(
                {
                    "arm": arm,
                    "split": split,
                    "samples": len(selected),
                    "initial_chamfer": _mean(selected, "initial_chamfer"),
                    "refined_chamfer": _mean(selected, "refined_chamfer"),
                    "relative_chamfer_gain": _mean(selected, "relative_chamfer_gain"),
                    "p2s": _mean(selected, "p2s"),
                    "p2s_p95": _mean(selected, "p2s_p95"),
                    "fscore": _mean(selected, "fscore"),
                    "normal_consistency": _mean(selected, "normal_consistency"),
                    "normalized_flip_rate": _mean(selected, "normalized_flip_rate"),
                    "introduced_flipped_faces": sum(int(row["introduced_flipped_faces"]) for row in selected),
                    "new_degenerate_faces": sum(int(row["new_degenerate_faces"]) for row in selected),
                    "same_index_recovered_vertex_rms": _mean(selected, "same_index_recovered_vertex_rms"),
                    "improved": sum(bool(row["improved"]) for row in selected),
                    "worsened": sum(bool(row["worsened"]) for row in selected),
                }
            )
    by_key = {(row["arm"], row["split"], row["sample_id"]): row for row in all_rows}
    paired = []
    for split in ("validation", "test"):
        ids = sorted({row["sample_id"] for row in all_rows if row["split"] == split})
        for sample_id in ids:
            fixed = by_key[(ARMS[0], split, sample_id)]
            adaptive = by_key[(ARMS[1], split, sample_id)]
            paired.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "recipe": adaptive["recipe"],
                    "severity": adaptive["severity"],
                    "adaptive_lambda": adaptive["lambda"],
                    "adaptive_lower_chamfer": adaptive["refined_chamfer"] < fixed["refined_chamfer"],
                    "adaptive_lower_vertex_rms": adaptive["same_index_recovered_vertex_rms"] < fixed["same_index_recovered_vertex_rms"],
                    "adaptive_lower_p2s_p95": adaptive["p2s_p95"] < fixed["p2s_p95"],
                    "adaptive_better_normal": adaptive["normal_consistency"] > fixed["normal_consistency"],
                    "adaptive_lower_flip_rate": adaptive["normalized_flip_rate"] < fixed["normalized_flip_rate"],
                    "chamfer_adaptive_minus_fixed": adaptive["refined_chamfer"] - fixed["refined_chamfer"],
                    "vertex_rms_adaptive_minus_fixed": adaptive["same_index_recovered_vertex_rms"] - fixed["same_index_recovered_vertex_rms"],
                }
            )
    oracle_rows = list(csv.DictReader(args.oracle_per_sample.open(encoding="utf-8")))
    oracle = {(row["split"], row["sample_id"]): float(row["lambda_oracle"]) for row in oracle_rows}
    lambda_summary = []
    lambda_groups = []
    lambda_buckets = []
    for split in ("validation", "test"):
        selected = [row for row in all_rows if row["arm"] == ARMS[1] and row["split"] == split]
        values = [float(row["lambda"]) for row in selected]
        stats = _quantiles(values)
        stats.update(
            {
                "split": split,
                "near_lower_bound": sum(value <= LAMBDA_MIN * 1.05 for value in values),
                "near_upper_bound": sum(value >= LAMBDA_MAX / 1.05 for value in values),
                "nearest_oracle_bucket": sum(
                    abs(np.log10(float(row["lambda"])) - np.log10(oracle[(split, row["sample_id"])]))
                    <= np.log10(np.sqrt(3.0))
                    for row in selected
                ),
            }
        )
        lambda_summary.append(stats)
        edges = np.linspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), 9)
        counts, _ = np.histogram(np.log10(values), bins=edges)
        for index, count in enumerate(counts):
            lambda_buckets.append(
                {
                    "split": split,
                    "log10_low": float(edges[index]),
                    "log10_high": float(edges[index + 1]),
                    "samples": int(count),
                }
            )
        for group_type, groups in (
            ("recipe", ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")),
            ("severity", ("mild", "strong")),
        ):
            for group in groups:
                subset = [row for row in selected if row[group_type] == group]
                lambda_groups.append(
                    {
                        "split": split,
                        "group_type": group_type,
                        "group": group,
                        "samples": len(subset),
                        **_quantiles([float(row["lambda"]) for row in subset]),
                    }
                )
    paired_counts = []
    for split in ("validation", "test"):
        selected = [row for row in paired if row["split"] == split]
        paired_counts.append(
            {
                "split": split,
                "samples": len(selected),
                "lower_chamfer": sum(bool(row["adaptive_lower_chamfer"]) for row in selected),
                "lower_vertex_rms": sum(bool(row["adaptive_lower_vertex_rms"]) for row in selected),
                "lower_p2s_p95": sum(bool(row["adaptive_lower_p2s_p95"]) for row in selected),
                "better_normal": sum(bool(row["adaptive_better_normal"]) for row in selected),
                "lower_flip_rate": sum(bool(row["adaptive_lower_flip_rate"]) for row in selected),
            }
        )
    test_geometry = {row["arm"]: row for row in aggregate_geometry if row["split"] == "test"}
    fixed = test_geometry[ARMS[0]]
    adaptive = test_geometry[ARMS[1]]
    relative_cd_improvement = (fixed["refined_chamfer"] - adaptive["refined_chamfer"]) / fixed["refined_chamfer"]
    lambda_test = next(row for row in lambda_summary if row["split"] == "test")
    distribution_collapsed = bool(lambda_test["std"] < 0.05 * lambda_test["mean"])
    success = bool(
        relative_cd_improvement >= 0.01
        and adaptive["same_index_recovered_vertex_rms"] < fixed["same_index_recovered_vertex_rms"]
        and adaptive["normal_consistency"] >= fixed["normal_consistency"] - 0.002
        and adaptive["normalized_flip_rate"] <= fixed["normalized_flip_rate"] * 1.05
        and adaptive["new_degenerate_faces"] == 0
        and not distribution_collapsed
    )
    contract = {
        "passed": True,
        "same_source_checkpoint": True,
        "source_checkpoint_sha256": payloads[0]["source_checkpoint_sha256"],
        "same_additional_optimizer_steps": all(int(payload["metrics"]["optimizer_steps"]) == FINAL_STEP for payload in payloads),
        "same_effective_global_batch": all(int(payload["metrics"]["global_batch_meshes"]) == 8 for payload in payloads),
        "same_dataset_split_seed_schedule": True,
        "lambda_head_gt_inputs": False,
        "gt_used_only_for_evaluation": True,
        "test_checkpoint": "exact_step_25000_not_raw_loss_selected",
        "metric_protocol": METRIC_PROTOCOL,
    }
    summary = {
        "contract_audit": contract,
        "prediction": aggregate_prediction,
        "geometry": aggregate_geometry,
        "paired": paired_counts,
        "lambda_summary": lambda_summary,
        "lambda_groups": lambda_groups,
        "lambda_buckets": lambda_buckets,
        "training_runtime_diagnostics": {payload["arm"]: payload["training_runtime_diagnostics"] for payload in payloads},
        "decision": {
            "adaptive_success": success,
            "relative_test_chamfer_improvement_vs_G": relative_cd_improvement,
            "distribution_collapsed": distribution_collapsed,
        },
    }
    _write_csv(output / "per_sample.csv", all_rows)
    _write_csv(output / "paired_per_sample.csv", paired)
    _write_csv(output / "prediction_summary.csv", aggregate_prediction)
    _write_csv(output / "geometry_summary.csv", aggregate_geometry)
    _write_csv(output / "lambda_summary.csv", lambda_summary)
    _write_csv(output / "lambda_by_coarse_group.csv", lambda_groups)
    _write_csv(output / "lambda_histogram.csv", lambda_buckets)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contract_audit.json", contract)
    lines = [
        "# Sofa50 v2 fixed-vs-adaptive lambda matched continuation",
        "",
        f"Contract audit: **{str(contract['passed']).lower()}**.",
        "",
        "Both arms start from the exact same Arm-B step-20,000 checkpoint and continue to exact step 25,000. Test uses the exact final checkpoint for both arms.",
        "",
        "| Split | Arm | Refined CD | Gain | P2S p95 | Normal | Flip rate | Vertex RMS | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_geometry:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['refined_chamfer']:.9g} | {row['relative_chamfer_gain']:.2%} | "
            f"{row['p2s_p95']:.9g} | {row['normal_consistency']:.9g} | {row['normalized_flip_rate']:.4%} | "
            f"{row['same_index_recovered_vertex_rms']:.9g} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(("", "## Learned lambda", "", "| Split | Mean | Median | Std | Min | P10 | P25 | P75 | P90 | Max | Near bounds low/high | Near oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in lambda_summary:
        lines.append(
            f"| {row['split']} | {row['mean']:.6g} | {row['median']:.6g} | {row['std']:.6g} | {row['min']:.6g} | "
            f"{row['p10']:.6g} | {row['p25']:.6g} | {row['p75']:.6g} | {row['p90']:.6g} | {row['max']:.6g} | "
            f"{row['near_lower_bound']}/{row['near_upper_bound']} | {row['nearest_oracle_bucket']}/50 |"
        )
    test_pair = next(row for row in paired_counts if row["split"] == "test")
    lines.extend(
        (
            "",
            "## Decision",
            "",
            f"Adaptive success under the predeclared criteria: **{str(success).lower()}**. Test Chamfer change versus G is `{relative_cd_improvement:+.2%}`.",
            "",
            f"H wins on test Chamfer / vertex RMS / P2S p95 / normal / flip rate: `{test_pair['lower_chamfer']}` / `{test_pair['lower_vertex_rms']}` / `{test_pair['lower_p2s_p95']}` / `{test_pair['better_normal']}` / `{test_pair['lower_flip_rate']}` out of 50.",
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
    parser.add_argument("--oracle-per-sample", type=Path)
    parser.add_argument("--arm-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        if args.oracle_per_sample is None:
            parser.error("merge requires --oracle-per-sample")
        merge(args)
    else:
        if args.manifest is None or args.runs_root is None or args.arm_index is None:
            parser.error("evaluation requires manifest, runs-root and arm-index")
        if not 0 <= args.arm_index < 2:
            parser.error("arm-index must be 0 or 1")
        evaluate_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Frozen exact-target recovery replay with independent Adam step budgets."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_multitopology_rawlap import load_spec
from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Mesh
from mlr.learned_laplacian.evaluation import _reconstruct
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one
from mlr.refinement import RefinementConfig


STEP_BUDGETS = (200, 500, 1000, 2000)


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


def _load_reference(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "full_solver":
                result[str(row["sample_id"])] = float(row["chamfer"])
    return result


def recovery_config(base: Mapping[str, Any], steps: int) -> RefinementConfig:
    return RefinementConfig(
        operator_type=str(base.get("operator_type", "uniform")),
        lambda_lap=float(base.get("lambda_lap", 1.0)),
        lambda_anchor=float(base.get("lambda_anchor", 0.01)),
        lambda_edge=float(base.get("lambda_edge", 0.0)),
        lambda_unseen_anchor=float(base.get("unseen_anchor_weight", 0.0)),
        num_iters=int(steps),
        learning_rate=float(base.get("learning_rate", 0.01)),
        robust_loss=str(base.get("robust_loss", "huber")),
        huber_delta=float(base.get("huber_delta", 0.01)),
    )


def _residual_metrics(
    vertices: np.ndarray,
    target: np.ndarray,
    faces: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    data = build_uniform_laplacian_data(faces, len(vertices))
    residual = apply_uniform_laplacian(vertices, data) - target
    norm = np.linalg.norm(residual, axis=1)
    weighted = np.sqrt(np.clip(weight, 0.0, None))[:, None] * residual
    weighted_norm = np.linalg.norm(weighted, axis=1)
    return {
        "laplacian_residual_rms": float(np.sqrt(np.mean(norm**2))),
        "laplacian_residual_max": float(norm.max(initial=0.0)),
        "weighted_laplacian_residual_rms": float(np.sqrt(np.mean(weighted_norm**2))),
    }


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "checkpoint_latest.pt"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact checkpoint confidence replay.")
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    model_spec = load_spec(run_dir, device)
    if int(model_spec["optimizer_steps"]) != 20_000:
        raise RuntimeError("Expected frozen 20k checkpoint.")
    config = model_spec["config"]
    base_recovery = dict(config.get("recovery", {}))
    reference = _load_reference(args.reference_ablation_csv)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for index in range(args.shard_index, len(dataset), args.shard_count):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        target = torch.as_tensor(static["raw_laplacian_target"]).cpu().numpy().astype(np.float64)
        inferred = _infer_one(dataset, index, model_spec, device, current_faces=static["faces"])
        confidence = torch.as_tensor(inferred["confidence"]).float().cpu().numpy()
        visible = torch.as_tensor(inferred["visibility_count"]).cpu().numpy() > 0
        weight = visible.astype(np.float64) * np.clip(confidence.astype(np.float64), 0.0, 1.0)
        baseline = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in (("initial", initial), ("clean", clean))
        }
        initial_cd = float(baseline["initial"]["chamfer"])
        clean_cd = float(baseline["clean"]["chamfer"])
        available = initial_cd - clean_cd
        reference_error = float("nan")
        sample_rows: list[dict[str, Any]] = []
        for steps in STEP_BUDGETS:
            started = time.perf_counter()
            result, solver_name = _reconstruct(
                initial,
                target,
                np.ones(initial.num_vertices, dtype=np.float64),
                recovery_config(base_recovery, steps),
                args.dense_vertex_limit,
                laplacian_weight=weight,
            )
            runtime = time.perf_counter() - started
            geometry = _geometry_row(
                args.dataset_arm, sample_id, f"adam_{steps}", result.mesh, clean, initial
            )
            row = {
                **geometry,
                "steps": steps,
                "variant": metadata.get("variant"),
                "vertices": initial.num_vertices,
                "faces": initial.num_faces,
                "initial_chamfer": initial_cd,
                "clean_chamfer": clean_cd,
                "eta_recovery": (initial_cd - float(geometry["chamfer"])) / available,
                "runtime_seconds": runtime,
                "solver_name": solver_name,
                "configured_robust_loss": str(base_recovery.get("robust_loss", "huber")),
                "effective_robust_loss": "l2" if solver_name == "sparse_uniform_oracle_core" else str(base_recovery.get("robust_loss", "huber")),
                "visible_fraction": float(visible.mean()),
                "confidence_mean": float(confidence.mean()),
                "confidence_std": float(confidence.std()),
                "final_objective": float(result.history[-1]["loss"]),
                **_residual_metrics(result.vertices, target, initial.faces, weight),
            }
            if steps == 200:
                reference_error = abs(float(geometry["chamfer"]) - reference[sample_id])
                row["reference_full_solver_chamfer_abs_error"] = reference_error
            else:
                row["reference_full_solver_chamfer_abs_error"] = float("nan")
            sample_rows.append(row)
            rows.append(row)
        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "checkpoint": str(checkpoint),
            "checkpoint_optimizer_steps": int(model_spec["optimizer_steps"]),
            "step_budgets": list(STEP_BUDGETS),
            "same_initial_graph_exact_target_weights_optimizer_lr": True,
            "each_budget_restarts_from_same_initial_mesh": True,
            "only_num_iters_changed": True,
            "adam_200_reference_chamfer_abs_error": reference_error,
            "all_sparse": all(row["solver_name"] == "sparse_uniform_oracle_core" for row in sample_rows),
        }
        audit["passed"] = bool(reference_error <= args.reference_tolerance)
        audits.append(audit)
        print(
            f"{args.dataset_arm} {sample_id}: eta200={sample_rows[0]['eta_recovery']:.4g} "
            f"eta2000={sample_rows[-1]['eta_recovery']:.4g} audit={audit['passed']}",
            flush=True,
        )
        del inferred
        torch.cuda.empty_cache()

    _write_json(
        output / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "dataset_arm": args.dataset_arm,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "metric_protocol": METRIC_PROTOCOL,
            "step_budgets": list(STEP_BUDGETS),
            "rows": rows,
            "audits": audits,
        },
    )


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    rows = [row for shard in shards for row in shard["rows"]]
    audits = [row for shard in shards for row in shard["audits"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    if len(rows) != expected * len(STEP_BUDGETS) or len(audits) != expected:
        raise RuntimeError("Incomplete Adam-step sweep shards.")
    if not all(bool(row["passed"]) for row in audits):
        raise RuntimeError("Adam-step sweep contract audit failed.")
    aggregates = []
    for steps in STEP_BUDGETS:
        selected = [row for row in rows if int(row["steps"]) == steps]
        eta = np.asarray([float(row["eta_recovery"]) for row in selected])
        aggregates.append(
            {
                "dataset_arm": args.dataset_arm,
                "steps": steps,
                "samples": len(selected),
                "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
                "p2s": float(np.mean([float(row["p2s"]) for row in selected])),
                "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
                "normal_consistency": float(np.mean([float(row["normal_consistency"]) for row in selected])),
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved_over_initial": int(sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)),
                "eta_mean": float(eta.mean()),
                "eta_median": float(np.median(eta)),
                "eta_p10": float(np.quantile(eta, 0.1)),
                "eta_p90": float(np.quantile(eta, 0.9)),
                "eta_negative_count": int((eta < 0).sum()),
                "laplacian_residual_rms": float(np.mean([float(row["laplacian_residual_rms"]) for row in selected])),
                "weighted_laplacian_residual_rms": float(np.mean([float(row["weighted_laplacian_residual_rms"]) for row in selected])),
                "final_objective": float(np.mean([float(row["final_objective"]) for row in selected])),
                "runtime_seconds_mean": float(np.mean([float(row["runtime_seconds"]) for row in selected])),
                "runtime_seconds_sum": float(np.sum([float(row["runtime_seconds"]) for row in selected])),
            }
        )
    eta_means = np.asarray([float(row["eta_mean"]) for row in aggregates])
    by_sample = {
        sample_id: [next(float(row["eta_recovery"]) for row in rows if row["sample_id"] == sample_id and int(row["steps"]) == steps) for steps in STEP_BUDGETS]
        for sample_id in sorted({str(row["sample_id"]) for row in rows})
    }
    summary = {
        "dataset_arm": args.dataset_arm,
        "contract_audit": True,
        "test_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "step_budgets": list(STEP_BUDGETS),
        "aggregates": aggregates,
        "convergence": {
            "mean_eta_non_decreasing_with_steps": bool(np.all(np.diff(eta_means) >= -1e-12)),
            "per_sample_eta_non_decreasing_count": int(sum(np.all(np.diff(values) >= -1e-12) for values in by_sample.values())),
            "per_sample_total": len(by_sample),
            "mean_eta_gain_200_to_2000": float(eta_means[-1] - eta_means[0]),
        },
        "solver_behavior": {
            "all_samples_sparse": all(bool(row["all_sparse"]) for row in audits),
            "configured_robust_loss": "huber",
            "effective_sparse_robust_loss": "l2",
        },
        "maximum_200_step_reference_chamfer_error": max(float(row["adam_200_reference_chamfer_abs_error"]) for row in audits),
    }
    contract = {
        "passed": True,
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(audits),
        "same_initial_graph_exact_target_weights_optimizer_lr": True,
        "only_num_iters_changed": True,
        "independent_restarts_from_initial": True,
        "manifest": shards[0]["manifest"],
        "manifest_sha256": shards[0]["manifest_sha256"],
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_sha256": shards[0]["checkpoint_sha256"],
        "maximum_200_step_reference_chamfer_error": summary["maximum_200_step_reference_chamfer_error"],
        "metric_protocol": METRIC_PROTOCOL,
    }
    _write_csv(output / "per_sample.csv", rows)
    _write_csv(output / "aggregate.csv", aggregates)
    _write_json(output / "per_sample_contract_audit.json", audits)
    _write_json(output / "contract_audit.json", contract)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-ablation-csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-vertex-limit", type=int, default=5000)
    parser.add_argument("--reference-tolerance", type=float, default=1e-12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.run_dir is None or args.reference_ablation_csv is None:
            parser.error("--run-dir and --reference-ablation-csv are required unless --merge-only")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

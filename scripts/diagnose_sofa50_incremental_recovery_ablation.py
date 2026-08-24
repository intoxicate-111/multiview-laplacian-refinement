#!/usr/bin/env python3
from __future__ import annotations

"""Incremental exact-target recovery ablation for matched Sofa50 v1/v2 meshes."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_multitopology_rawlap import load_spec
from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import _reconstruct
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multitopology_rawlap import raw_uniform_laplacian
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one
from mlr.refinement import RefinementConfig


ARM_ORDER = (
    "pure_laplacian_l2",
    "plus_anchor",
    "plus_visibility",
    "plus_confidence",
    "plus_huber",
    "full_solver",
)


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


def recovery_arm_specs(recovery: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return cumulative requested settings without changing the frozen budget."""

    base = {
        "operator_type": str(recovery.get("operator_type", "uniform")),
        "lambda_lap": float(recovery.get("lambda_lap", 1.0)),
        "lambda_anchor": 0.0,
        "lambda_edge": 0.0,
        "lambda_unseen_anchor": 0.0,
        "num_iters": int(recovery.get("num_iters", 200)),
        "learning_rate": float(recovery.get("learning_rate", 0.01)),
        "robust_loss": "l2",
        "huber_delta": float(recovery.get("huber_delta", 0.01)),
        "weight_mode": "uniform",
    }
    specs: dict[str, dict[str, Any]] = {}
    specs["pure_laplacian_l2"] = dict(base)
    specs["plus_anchor"] = {
        **base,
        "lambda_anchor": float(recovery.get("lambda_anchor", 0.01)),
    }
    specs["plus_visibility"] = {
        **specs["plus_anchor"],
        "weight_mode": "hard_any_view_visibility",
    }
    specs["plus_confidence"] = {
        **specs["plus_visibility"],
        "weight_mode": "visibility_times_learned_confidence",
    }
    specs["plus_huber"] = {
        **specs["plus_confidence"],
        "robust_loss": "huber",
    }
    specs["full_solver"] = {
        "operator_type": str(recovery.get("operator_type", "uniform")),
        "lambda_lap": float(recovery.get("lambda_lap", 1.0)),
        "lambda_anchor": float(recovery.get("lambda_anchor", 0.01)),
        "lambda_edge": float(recovery.get("lambda_edge", 0.0)),
        "lambda_unseen_anchor": float(recovery.get("unseen_anchor_weight", 0.0)),
        "num_iters": int(recovery.get("num_iters", 200)),
        "learning_rate": float(recovery.get("learning_rate", 0.01)),
        "robust_loss": str(recovery.get("robust_loss", "huber")),
        "huber_delta": float(recovery.get("huber_delta", 0.01)),
        "weight_mode": "visibility_times_learned_confidence",
    }
    return specs


def _refinement_config(spec: Mapping[str, Any]) -> RefinementConfig:
    return RefinementConfig(
        operator_type=str(spec["operator_type"]),
        lambda_lap=float(spec["lambda_lap"]),
        lambda_anchor=float(spec["lambda_anchor"]),
        lambda_edge=float(spec["lambda_edge"]),
        lambda_unseen_anchor=float(spec["lambda_unseen_anchor"]),
        num_iters=int(spec["num_iters"]),
        learning_rate=float(spec["learning_rate"]),
        robust_loss=str(spec["robust_loss"]),
        huber_delta=float(spec["huber_delta"]),
    )


def effective_terms(spec: Mapping[str, Any], solver_name: str) -> dict[str, Any]:
    sparse = solver_name == "sparse_uniform_oracle_core"
    return {
        "effective_robust_loss": "l2" if sparse else str(spec["robust_loss"]),
        "configured_robust_loss": str(spec["robust_loss"]),
        "huber_actually_active": bool(not sparse and spec["robust_loss"] == "huber"),
        "edge_term_actually_active": bool(not sparse and float(spec["lambda_edge"]) > 0),
        "anchor_actually_active": bool(float(spec["lambda_anchor"]) > 0),
        "visibility_actually_active": spec["weight_mode"] != "uniform",
        "confidence_actually_active": spec["weight_mode"] == "visibility_times_learned_confidence",
    }


def _weights(spec: Mapping[str, Any], visible: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    mode = str(spec["weight_mode"])
    if mode == "uniform":
        return np.ones(len(visible), dtype=np.float64)
    if mode == "hard_any_view_visibility":
        return visible.astype(np.float64)
    if mode == "visibility_times_learned_confidence":
        return visible.astype(np.float64) * np.clip(confidence.astype(np.float64), 0.0, 1.0)
    raise ValueError(mode)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
        "p10": float(np.quantile(array, 0.1)) if len(array) else float("nan"),
        "p90": float(np.quantile(array, 0.9)) if len(array) else float("nan"),
        "negative_count": int((array < 0).sum()),
    }


def identify_collapse(increments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in increments if str(row["arm"]) != "pure_laplacian_l2"]
    if not candidates:
        return {"arm": None, "mean_eta_delta": float("nan"), "classification": "not_available"}
    worst = min(candidates, key=lambda row: float(row["mean_eta_delta_from_previous"]))
    delta = float(worst["mean_eta_delta_from_previous"])
    return {
        "arm": str(worst["arm"]),
        "previous_arm": str(worst["previous_arm"]),
        "mean_eta_delta": delta,
        "classification": "largest_efficiency_collapse" if delta < 0 else "no_incremental_collapse",
    }


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint = run_dir / "checkpoint_latest.pt"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for confidence inference.")
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    spec = load_spec(run_dir, device)
    config = spec["config"]
    recovery = dict(config.get("recovery", {}))
    specs = recovery_arm_specs(recovery)
    if tuple(specs) != ARM_ORDER:
        raise RuntimeError("Recovery arm ordering changed.")
    if int(spec["optimizer_steps"]) != 20_000:
        raise RuntimeError("Expected the frozen 20k checkpoint.")

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    indices = list(range(args.shard_index, len(dataset), args.shard_count))
    for index in indices:
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        target = torch.as_tensor(static["raw_laplacian_target"]).float().cpu()
        recomputed = torch.as_tensor(raw_uniform_laplacian(clean), dtype=torch.float32)
        target_error = float(torch.max(torch.abs(target - recomputed)))
        if target_error > args.target_tolerance:
            raise RuntimeError(f"Exact target mismatch for {sample_id}: {target_error}")

        inferred = _infer_one(dataset, index, spec, device, current_faces=static["faces"])
        confidence = torch.as_tensor(inferred["confidence"]).float().cpu().numpy()
        visible = torch.as_tensor(inferred["visibility_count"]).cpu().numpy() > 0
        baseline = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in (("initial", initial), ("clean", clean))
        }
        initial_cd = float(baseline["initial"]["chamfer"])
        clean_cd = float(baseline["clean"]["chamfer"])
        available = initial_cd - clean_cd
        previous_vertices = initial.vertices
        previous_cd = initial_cd
        full_vertices: np.ndarray | None = None
        plus_huber_vertices: np.ndarray | None = None
        sample_solvers: dict[str, str] = {}

        for arm in ARM_ORDER:
            arm_spec = specs[arm]
            weight = _weights(arm_spec, visible, confidence)
            result, solver_name = _reconstruct(
                initial,
                target.numpy(),
                np.ones(initial.num_vertices, dtype=np.float64),
                _refinement_config(arm_spec),
                args.dense_vertex_limit,
                laplacian_weight=weight,
            )
            sample_solvers[arm] = solver_name
            geometry = _geometry_row(args.dataset_arm, sample_id, arm, result.mesh, clean, initial)
            chamfer = float(geometry["chamfer"])
            eta = (initial_cd - chamfer) / available if available > 1e-12 else float("nan")
            effective = effective_terms(arm_spec, solver_name)
            rows.append(
                {
                    **geometry,
                    "arm": arm,
                    "variant": metadata.get("variant"),
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "initial_chamfer": initial_cd,
                    "clean_chamfer": clean_cd,
                    "available_gap": available,
                    "recovered_gap": initial_cd - chamfer,
                    "eta_recovery": eta,
                    "previous_arm": "initial" if arm == ARM_ORDER[0] else ARM_ORDER[ARM_ORDER.index(arm) - 1],
                    "chamfer_improvement_from_previous": previous_cd - chamfer,
                    "vertex_rms_change_from_previous": float(
                        np.sqrt(np.mean(np.sum((result.vertices - previous_vertices) ** 2, axis=1)))
                    ),
                    "solver_name": solver_name,
                    "configured_lambda_anchor": float(arm_spec["lambda_anchor"]),
                    "configured_lambda_edge": float(arm_spec["lambda_edge"]),
                    "configured_lambda_unseen_anchor": float(arm_spec["lambda_unseen_anchor"]),
                    "configured_weight_mode": str(arm_spec["weight_mode"]),
                    "visible_fraction": float(visible.mean()),
                    "confidence_mean": float(confidence.mean()),
                    "confidence_std": float(confidence.std()),
                    **effective,
                }
            )
            previous_vertices = result.vertices
            previous_cd = chamfer
            if arm == "plus_huber":
                plus_huber_vertices = result.vertices.copy()
            elif arm == "full_solver":
                full_vertices = result.vertices.copy()

        assert plus_huber_vertices is not None and full_vertices is not None
        huber_full_max = float(np.max(np.abs(plus_huber_vertices - full_vertices), initial=0.0))
        reference_error = float("nan")
        if args.reference_oracle_dir is not None:
            reference_path = args.reference_oracle_dir.resolve() / sample_id / "predicted_refined.obj"
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            reference = load_mesh(reference_path)
            reference_error = float(np.max(np.abs(reference.vertices - full_vertices), initial=0.0))
        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "checkpoint": str(checkpoint),
            "checkpoint_optimizer_steps": int(spec["optimizer_steps"]),
            "target_recompute_max_abs_float32_error": target_error,
            "same_initial_for_all_arms": True,
            "same_exact_target_for_all_arms": True,
            "same_graph_for_all_arms": True,
            "same_iterations_and_learning_rate": all(
                int(value["num_iters"]) == int(recovery.get("num_iters", 200))
                and float(value["learning_rate"]) == float(recovery.get("learning_rate", 0.01))
                for value in specs.values()
            ),
            "solver_names": sample_solvers,
            "plus_huber_vs_full_max_abs_vertex_error": huber_full_max,
            "full_vs_prior_exact_oracle_max_abs_vertex_error": reference_error,
            "passed": bool(
                target_error <= args.target_tolerance
                and np.array_equal(initial.faces, clean.faces)
                and (not np.isfinite(reference_error) or reference_error <= args.reference_tolerance)
            ),
        }
        audits.append(audit)
        print(
            f"{args.dataset_arm} {sample_id}: V={initial.num_vertices} "
            f"eta pure={rows[-6]['eta_recovery']:.4g} full={rows[-1]['eta_recovery']:.4g} "
            f"solver={sample_solvers['full_solver']} audit={audit['passed']}",
            flush=True,
        )
        del inferred
        torch.cuda.empty_cache()

    shard = {
        "dataset_arm": args.dataset_arm,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metric_protocol": METRIC_PROTOCOL,
        "arm_order": list(ARM_ORDER),
        "arm_specs": specs,
        "rows": rows,
        "audits": audits,
    }
    _write_json(output / "shards" / f"shard_{args.shard_index:02d}.json", shard)


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    rows = [row for shard in shards for row in shard["rows"]]
    audits = [row for shard in shards for row in shard["audits"]]
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    expected = len(dataset)
    if len(rows) != expected * len(ARM_ORDER) or len(audits) != expected:
        raise RuntimeError("Incomplete recovery ablation shards.")
    if not all(bool(row["passed"]) for row in audits):
        raise RuntimeError("Per-sample recovery contract audit failed.")
    by_sample_arm = {(str(row["sample_id"]), str(row["arm"])): row for row in rows}
    aggregates: list[dict[str, Any]] = []
    increments: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(ARM_ORDER):
        selected = [row for row in rows if row["arm"] == arm]
        eta = [float(row["eta_recovery"]) for row in selected]
        aggregates.append(
            {
                "dataset_arm": args.dataset_arm,
                "arm": arm,
                "samples": len(selected),
                "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
                "p2s": float(np.mean([float(row["p2s"]) for row in selected])),
                "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
                "fscore": float(np.mean([float(row["fscore"]) for row in selected])),
                "normal_consistency": float(np.mean([float(row["normal_consistency"]) for row in selected])),
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved_over_initial": int(sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)),
                "eta_mean": float(np.mean(eta)),
                "eta_median": float(np.median(eta)),
                "eta_p10": float(np.quantile(eta, 0.1)),
                "eta_p90": float(np.quantile(eta, 0.9)),
                "eta_negative_count": int(sum(value < 0 for value in eta)),
                "dense_samples": int(sum(row["solver_name"] == "dense_refinement" for row in selected)),
                "sparse_samples": int(sum(row["solver_name"] == "sparse_uniform_oracle_core" for row in selected)),
                "huber_active_samples": int(sum(bool(row["huber_actually_active"]) for row in selected)),
            }
        )
        previous_arm = "initial" if arm_index == 0 else ARM_ORDER[arm_index - 1]
        sample_deltas = []
        eta_deltas = []
        for sample_id in {str(row["sample_id"]) for row in selected}:
            current = by_sample_arm[(sample_id, arm)]
            current_cd = float(current["chamfer"])
            current_eta = float(current["eta_recovery"])
            if previous_arm == "initial":
                previous_cd = float(current["initial_chamfer"])
                previous_eta = 0.0
            else:
                previous = by_sample_arm[(sample_id, previous_arm)]
                previous_cd = float(previous["chamfer"])
                previous_eta = float(previous["eta_recovery"])
            sample_deltas.append(previous_cd - current_cd)
            eta_deltas.append(current_eta - previous_eta)
        increments.append(
            {
                "dataset_arm": args.dataset_arm,
                "arm": arm,
                "previous_arm": previous_arm,
                "mean_chamfer_improvement_from_previous": float(np.mean(sample_deltas)),
                "median_chamfer_improvement_from_previous": float(np.median(sample_deltas)),
                "mean_eta_delta_from_previous": float(np.mean(eta_deltas)),
                "median_eta_delta_from_previous": float(np.median(eta_deltas)),
                "improved_samples": int(sum(value > 0 for value in sample_deltas)),
                "worsened_samples": int(sum(value < 0 for value in sample_deltas)),
                "unchanged_samples": int(sum(value == 0 for value in sample_deltas)),
            }
        )

    max_reference_error = max(
        (float(row["full_vs_prior_exact_oracle_max_abs_vertex_error"]) for row in audits if np.isfinite(float(row["full_vs_prior_exact_oracle_max_abs_vertex_error"]))),
        default=float("nan"),
    )
    summary = {
        "dataset_arm": args.dataset_arm,
        "contract_audit": True,
        "test_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "arm_order": list(ARM_ORDER),
        "arm_specs": shards[0]["arm_specs"],
        "aggregates": aggregates,
        "increments": increments,
        "collapse": identify_collapse(increments),
        "solver_routing": {
            "dense_vertex_limit": args.dense_vertex_limit,
            "dense_test_samples": aggregates[-1]["dense_samples"],
            "sparse_test_samples": aggregates[-1]["sparse_samples"],
            "huber_actually_active_test_samples": aggregates[-1]["huber_active_samples"],
            "sparse_path_ignores_configured_robust_loss": True,
        },
        "confidence": {
            "mean_of_sample_means": float(np.mean([float(row["confidence_mean"]) for row in rows if row["arm"] == "full_solver"])),
            "mean_of_sample_stds": float(np.mean([float(row["confidence_std"]) for row in rows if row["arm"] == "full_solver"])),
        },
        "visibility": {
            "mean_visible_fraction": float(np.mean([float(row["visible_fraction"]) for row in rows if row["arm"] == "full_solver"])),
        },
        "plus_huber_vs_full_max_abs_vertex_error": max(float(row["plus_huber_vs_full_max_abs_vertex_error"]) for row in audits),
        "full_vs_prior_exact_oracle_max_abs_vertex_error": max_reference_error,
    }
    audit = {
        "passed": True,
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(audits),
        "same_exact_target_initial_graph_iterations_learning_rate": True,
        "only_cumulative_recovery_terms_changed": True,
        "metric_protocol": METRIC_PROTOCOL,
        "manifest": shards[0]["manifest"],
        "manifest_sha256": shards[0]["manifest_sha256"],
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_sha256": shards[0]["checkpoint_sha256"],
        "maximum_target_recompute_error": max(float(row["target_recompute_max_abs_float32_error"]) for row in audits),
        "maximum_full_vs_prior_exact_oracle_vertex_error": max_reference_error,
        "configured_vs_effective_solver_behavior_reported": True,
    }
    _write_csv(output / "per_sample.csv", rows)
    _write_csv(output / "aggregate.csv", aggregates)
    _write_csv(output / "incremental_effects.csv", increments)
    _write_json(output / "per_sample_contract_audit.json", audits)
    _write_json(output / "contract_audit.json", audit)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-oracle-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-vertex-limit", type=int, default=5000)
    parser.add_argument("--target-tolerance", type=float, default=1e-7)
    parser.add_argument("--reference-tolerance", type=float, default=5e-5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required unless --merge-only is used")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Replace frozen Adam recovery with an initial-gauge sparse solve of archived predictions."""

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


STATES = ("initial", "frozen_adam_visibility", "predicted_sparse", "exact_sparse_oracle")


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


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    source = args.prediction_source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
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
        initial_centroids = component_centroids(initial.vertices, labels, component_count)
        predicted_vertices, predicted_solver = exact_sparse_solve(
            laplacian,
            prediction,
            labels,
            component_count,
            initial_centroids,
            atol=args.lsmr_atol,
            btol=args.lsmr_btol,
            maxiter=args.lsmr_maxiter,
        )
        # This arm is explicitly an oracle reference. GT is not used by the
        # predicted-sparse arm or its component-centroid gauge.
        exact_target = apply_uniform_laplacian(clean.vertices, lap_data)
        oracle_vertices, oracle_solver = exact_sparse_solve(
            laplacian,
            exact_target,
            labels,
            component_count,
            initial_centroids,
            atol=args.lsmr_atol,
            btol=args.lsmr_btol,
            maxiter=args.lsmr_maxiter,
        )
        predicted_sparse = Mesh(predicted_vertices, initial.faces.copy()).ensure_normals()
        exact_sparse = Mesh(oracle_vertices, initial.faces.copy()).ensure_normals()
        meshes = {
            "initial": initial,
            "frozen_adam_visibility": frozen,
            "predicted_sparse": predicted_sparse,
            "exact_sparse_oracle": exact_sparse,
        }
        geometry = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in meshes.items()
        }
        initial_cd = float(geometry["initial"]["chamfer"])
        clean_cd = float(_geometry_row(args.dataset_arm, sample_id, "clean", clean, clean, initial)["chamfer"])
        available = initial_cd - clean_cd
        for state in STATES:
            row = geometry[state]
            chamfer = float(row["chamfer"])
            row.update(
                {
                    "state": state,
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
            rows.append(row)
        eta_pred = float(geometry["predicted_sparse"]["eta_recovery"])
        eta_oracle = float(geometry["exact_sparse_oracle"]["eta_recovery"])
        pred_residual = laplacian @ predicted_vertices - prediction
        oracle_residual = laplacian @ oracle_vertices - exact_target
        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "archived_prediction": str(prediction_path),
            "archived_frozen_mesh": str(frozen_path),
            "archived_input_matches_manifest": archived_input_matches,
            "connected_components": component_count,
            "predicted_lsmr": predicted_solver,
            "oracle_lsmr": oracle_solver,
            "predicted_equation_residual_rms": float(
                np.sqrt(np.mean(np.sum(pred_residual**2, axis=1)))
            ),
            "oracle_equation_residual_rms": float(
                np.sqrt(np.mean(np.sum(oracle_residual**2, axis=1)))
            ),
            "eta_predicted_sparse": eta_pred,
            "eta_oracle_sparse": eta_oracle,
            "eta_retention_predicted_over_oracle": eta_pred / eta_oracle if abs(eta_oracle) > 1e-12 else float("nan"),
            "predicted_sparse_uses_gt_target": False,
            "predicted_sparse_uses_gt_gauge": False,
            "predicted_sparse_gauge_source": "initial_mesh_component_centroids",
            "no_visibility_confidence_anchor_huber_or_adam": True,
        }
        audit["passed"] = bool(
            archived_input_matches
            and predicted_solver["all_converged"]
            and oracle_solver["all_converged"]
            and np.array_equal(predicted_sparse.faces, initial.faces)
            and np.array_equal(exact_sparse.faces, initial.faces)
        )
        audits.append(audit)
        print(
            f"{args.dataset_arm} {sample_id}: frozen={geometry['frozen_adam_visibility']['eta_recovery']:.4g} "
            f"pred_sparse={eta_pred:.4g} oracle_sparse={eta_oracle:.4g} "
            f"retention={audit['eta_retention_predicted_over_oracle']:.4g} audit={audit['passed']}",
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
            "states": list(STATES),
            "rows": rows,
            "audits": audits,
        },
    )


def _aggregate(rows: Sequence[Mapping[str, Any]], state: str) -> dict[str, Any]:
    selected = [row for row in rows if row["state"] == state]
    return {
        "state": state,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "relative_chamfer_gain_mean": float(np.mean([float(row["relative_chamfer_gain"]) for row in selected])),
        "eta_mean": float(np.mean([float(row["eta_recovery"]) for row in selected])),
        "eta_median": float(np.median([float(row["eta_recovery"]) for row in selected])),
        "normal_consistency": float(np.mean([float(row["normal_consistency"]) for row in selected])),
        "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
        "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
        "improved_over_initial": int(sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)),
        "worsened_over_initial": int(sum(float(row["chamfer"]) > float(row["initial_chamfer"]) for row in selected)),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    rows = [row for shard in shards for row in shard["rows"]]
    audits = [row for shard in shards for row in shard["audits"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    if len(rows) != expected * len(STATES) or len(audits) != expected:
        raise RuntimeError("Incomplete predicted sparse-solve shards")
    if not all(bool(row["passed"]) for row in audits):
        raise RuntimeError("Predicted sparse-solve contract audit failed")
    aggregates = [{"dataset_arm": args.dataset_arm, **_aggregate(rows, state)} for state in STATES]
    by_state = {row["state"]: row for row in aggregates}
    by_sample = {
        (str(row["sample_id"]), str(row["state"])): row for row in rows
    }
    paired = []
    retentions = []
    for sample_id in sorted({str(row["sample_id"]) for row in rows}):
        frozen = by_sample[(sample_id, "frozen_adam_visibility")]
        sparse = by_sample[(sample_id, "predicted_sparse")]
        oracle = by_sample[(sample_id, "exact_sparse_oracle")]
        retention = float(sparse["eta_recovery"]) / float(oracle["eta_recovery"])
        retentions.append(retention)
        paired.append(
            {
                "dataset_arm": args.dataset_arm,
                "sample_id": sample_id,
                "pred_sparse_minus_frozen_chamfer": float(sparse["chamfer"]) - float(frozen["chamfer"]),
                "pred_sparse_minus_frozen_eta": float(sparse["eta_recovery"]) - float(frozen["eta_recovery"]),
                "pred_sparse_better_chamfer": float(sparse["chamfer"]) < float(frozen["chamfer"]),
                "eta_retention_predicted_over_oracle": retention,
            }
        )
    pred_eta = float(by_state["predicted_sparse"]["eta_mean"])
    oracle_eta = float(by_state["exact_sparse_oracle"]["eta_mean"])
    summary = {
        "dataset_arm": args.dataset_arm,
        "contract_audit": True,
        "test_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "aggregates": aggregates,
        "retention": {
            "ratio_of_mean_eta_predicted_over_oracle": pred_eta / oracle_eta,
            "per_sample_ratio_mean": float(np.mean(retentions)),
            "per_sample_ratio_median": float(np.median(retentions)),
            "per_sample_ratio_p10": float(np.quantile(retentions, 0.1)),
            "per_sample_ratio_p90": float(np.quantile(retentions, 0.9)),
        },
        "predicted_sparse_vs_frozen": {
            "mean_chamfer_difference": float(np.mean([float(row["pred_sparse_minus_frozen_chamfer"]) for row in paired])),
            "mean_eta_difference": float(np.mean([float(row["pred_sparse_minus_frozen_eta"]) for row in paired])),
            "predicted_sparse_better_chamfer_count": int(sum(bool(row["pred_sparse_better_chamfer"]) for row in paired)),
            "frozen_better_or_equal_chamfer_count": int(sum(not bool(row["pred_sparse_better_chamfer"]) for row in paired)),
        },
        "solver": {
            "method": "scipy.sparse.linalg.lsmr",
            "gauge": "initial_mesh_component_centroids",
            "visibility": False,
            "confidence": False,
            "positional_anchor": False,
            "huber": False,
            "adam": False,
        },
    }
    contract = {
        "passed": True,
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(audits),
        "all_archived_inputs_match": all(bool(row["archived_input_matches_manifest"]) for row in audits),
        "all_lsmr_converged": all(bool(row["predicted_lsmr"]["all_converged"]) and bool(row["oracle_lsmr"]["all_converged"]) for row in audits),
        "predicted_solve_uses_no_gt": True,
        "only_initial_component_centroids_fix_nullspace": True,
        "no_visibility_confidence_anchor_huber_or_adam": True,
        "manifest": shards[0]["manifest"],
        "manifest_sha256": shards[0]["manifest_sha256"],
        "prediction_source_dir": shards[0]["prediction_source_dir"],
        "prediction_arm_name": shards[0]["prediction_arm_name"],
        "metric_protocol": METRIC_PROTOCOL,
    }
    _write_csv(output / "per_sample.csv", rows)
    _write_csv(output / "paired_predicted_sparse_vs_frozen.csv", paired)
    _write_csv(output / "aggregate.csv", aggregates)
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

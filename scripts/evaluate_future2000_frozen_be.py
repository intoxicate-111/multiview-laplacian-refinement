#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate frozen Future2000 Arm-E and dense Arm-B+E fusion.

Validation shards evaluate a declared lambda grid.  Test shards require a
validation-produced lock file and evaluate exactly that one lambda.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_future2000_external_baseline import _audit_source_identity
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
    _topology_change,
)
from mlr.learned_laplacian.controlled_displacement import (
    CURRENT_GRAPH_LAPLACIAN,
    DIRECT_VERTEX_DISPLACEMENT,
    prediction_semantics,
)
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    recovery_forward_audit,
)
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.target_scaling import prediction_to_raw_laplacian
from run_sofa50_same_initial_ours import spec


DEFAULT_LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
PCG_TOLERANCE = 1e-4
PCG_MAXIMUM_ITERATIONS = 2048
FIELDS = (
    "initial_chamfer",
    "refined_chamfer",
    "initial_p2s_mean",
    "refined_p2s_mean",
    "initial_p2s_p95",
    "refined_p2s_p95",
    "initial_fscore",
    "refined_fscore",
    "initial_normal_consistency",
    "refined_normal_consistency",
    "chamfer_improvement_rate",
    "improved",
    "output_connectivity_preserved",
    "introduced_flipped_faces",
    "new_degenerate_faces",
    "same_index_recovered_vertex_rms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _parse_lambdas(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 or not np.isfinite(item) for item in values):
        raise ValueError("All lambda values must be positive and finite")
    if len(set(values)) != len(values):
        raise ValueError("Lambda grid contains duplicates")
    return values


def _manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): dict(row) for row in payload["samples"]}


def _geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    initial: Mesh,
    gt: Mesh,
    before: dict[str, Any],
    *,
    surface_samples: int,
    metric_seed: int,
    fscore_threshold: float,
) -> dict[str, Any]:
    result_mesh = Mesh(vertices, faces.copy()).ensure_normals()
    after = evaluate_mesh_geometry(
        result_mesh,
        gt,
        surface_samples=surface_samples,
        seed=metric_seed,
        fscore_threshold=fscore_threshold,
    )
    mapping = {
        "chamfer": "chamfer",
        "p2s_mean": "point_to_surface_bidirectional_mean",
        "p2s_p95": "point_to_surface_bidirectional_p95",
        "fscore": "fscore",
        "normal_consistency": "normal_consistency",
    }
    row: dict[str, Any] = {}
    for short, key in mapping.items():
        row[f"initial_{short}"] = float(before[key])
        row[f"refined_{short}"] = float(after[key])
    initial_cd = row["initial_chamfer"]
    refined_cd = row["refined_chamfer"]
    row["chamfer_improvement_rate"] = (initial_cd - refined_cd) / initial_cd
    row["improved"] = refined_cd < initial_cd
    row["output_connectivity_preserved"] = True
    topology = _topology_change(initial.vertices, vertices, faces)
    row["introduced_flipped_faces"] = int(topology["introduced_flips"])
    row["new_degenerate_faces"] = int(topology["new_degeneracies"])
    row["same_index_recovered_vertex_rms"] = float(
        np.sqrt(np.mean(np.sum((vertices - gt.vertices) ** 2, axis=1)))
    )
    return row


def _forward(
    prepared: Any,
    model_spec: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    conditioned = _exact_query_sample(prepared.sample, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=model_spec["amp_dtype"],
        enabled=bool(model_spec["amp_enabled"]),
    ):
        output = model_spec["model"](conditioned)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return output.predicted_laplacian.float().detach(), elapsed


def _pcg(
    delta: torch.Tensor,
    anchor: torch.Tensor,
    static: dict[str, Any],
    regularization: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    edge_index = torch.as_tensor(static["edge_index"], dtype=torch.long, device=device)
    degree = torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device)
    delta64 = delta.to(device=device, dtype=torch.float64)
    anchor64 = anchor.to(device=device, dtype=torch.float64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        recovered, audit = recovery_forward_audit(
            delta64,
            anchor64,
            edge_index,
            degree,
            regularization=regularization,
            maximum_iterations=PCG_MAXIMUM_ITERATIONS,
            tolerance=PCG_TOLERANCE,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    solver = {
        "pcg_iterations": int(audit.iterations),
        "pcg_converged": bool(audit.converged),
        "pcg_relative_residual": float(audit.relative_residual),
        "pcg_runtime_seconds": time.perf_counter() - started,
        "pcg_tolerance": PCG_TOLERANCE,
        "pcg_maximum_iterations": PCG_MAXIMUM_ITERATIONS,
        "pcg_dtype": "float64",
    }
    return recovered.detach().cpu().numpy(), solver


def _base_row(
    static: dict[str, Any],
    source: dict[str, Any],
    *,
    split: str,
    index: int,
    method: str,
    b_spec: dict[str, Any],
    e_spec: dict[str, Any],
    manifest_sha: str,
) -> dict[str, Any]:
    sample_id = str(static["sample_id"])
    row = {
        "status": "success",
        "split": split,
        "sample_id": sample_id,
        "object_id": sample_id.rpartition("__v")[0],
        "sample_index": index,
        "method": method,
        "manifest_sha256": manifest_sha,
        "arm_b_checkpoint": str(b_spec["checkpoint"]),
        "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
        "arm_b_checkpoint_epoch": b_spec["checkpoint_epoch"],
        "arm_e_checkpoint": str(e_spec["checkpoint"]),
        "arm_e_checkpoint_sha256": e_spec["checkpoint_sha256"],
        "arm_e_checkpoint_epoch": e_spec["checkpoint_epoch"],
        "gt_used_for_prediction_or_recovery": False,
        "models_retrained": False,
    }
    row.update(_audit_source_identity(static, source))
    return row


def _save_test_sample(root: Path, method: str, sample_id: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    target = root / "results" / method / "samples" / sample_id
    target.mkdir(parents=True, exist_ok=True)
    save_mesh(Mesh(vertices, faces.copy()).ensure_normals(), target / "refined.obj")
    np.save(target / "refined_vertices.npy", vertices)


def run(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    device = torch.device(args.device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} {args.split} samples, found {len(dataset)}")
    selected = list(range(args.shard_index, len(dataset), args.shard_count))
    provenance = _manifest_rows(args.manifest.resolve())
    manifest_sha = sha256(args.manifest.resolve())

    b_spec = spec(
        args.arm_b_run.resolve(), device,
        view_chunk_size=args.view_chunk_size,
        checkpoint_name="checkpoint_best.pt",
        expected_checkpoint_sha256=args.arm_b_checkpoint_sha256,
    )
    e_spec = spec(
        args.arm_e_run.resolve(), device,
        view_chunk_size=args.view_chunk_size,
        checkpoint_name="checkpoint_best.pt",
        expected_checkpoint_sha256=args.arm_e_checkpoint_sha256,
    )
    if prediction_semantics(b_spec["config"]) != CURRENT_GRAPH_LAPLACIAN:
        raise RuntimeError("Arm-B checkpoint is not a current-graph Laplacian predictor")
    if prediction_semantics(e_spec["config"]) != DIRECT_VERTEX_DISPLACEMENT:
        raise RuntimeError("Arm-E checkpoint is not a direct displacement predictor")
    if args.split == "validation":
        lambdas = _parse_lambdas(args.lambda_grid)
        lock = None
        output_file = args.output_dir / "validation" / "shards" / f"validation_shard_{args.shard_index:03d}.csv"
    else:
        if args.lambda_lock is None or not args.lambda_lock.is_file():
            raise ValueError("Test evaluation requires an existing --lambda-lock")
        lock = json.loads(args.lambda_lock.read_text(encoding="utf-8"))
        if lock.get("selection_split") != "validation" or not lock.get("contract_audit"):
            raise RuntimeError("Lambda lock is not an audited validation-only selection")
        expected = {
            "manifest_sha256": manifest_sha,
            "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
            "arm_e_checkpoint_sha256": e_spec["checkpoint_sha256"],
        }
        for key, value in expected.items():
            if lock.get(key) != value:
                raise RuntimeError(f"Lambda lock {key} mismatch")
        lambdas = (float(lock["selected_lambda"]),)
        output_file = args.output_dir / "test" / "shards" / f"test_shard_{args.shard_index:03d}.csv"
    if output_file.exists() and not args.force:
        print(f"resume: {output_file}")
        return

    rows: list[dict[str, Any]] = []
    peak_gpu = 0
    for progress, index in enumerate(selected, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        faces = torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64)
        initial_vertices = torch.as_tensor(static["vertices"]).cpu().numpy().astype(np.float64)
        gt_vertices = torch.as_tensor(static["gt_vertices"]).cpu().numpy().astype(np.float64)
        gt_faces = torch.as_tensor(static["gt_faces"]).cpu().numpy().astype(np.int64)
        initial = Mesh(initial_vertices, faces).ensure_normals()
        gt = Mesh(gt_vertices, gt_faces).ensure_normals()
        before = evaluate_mesh_geometry(
            initial, gt,
            surface_samples=args.surface_samples,
            seed=args.metric_seed,
            fscore_threshold=args.fscore_threshold,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prepared_b = _load_device_item(dataset, index, b_spec["config"], device)
        b_output, b_forward = _forward(prepared_b, b_spec, device)
        h = prepared_b.sample["local_edge_length"].float()
        epsilon = float(b_spec["config"].get("target_scaling", {}).get("epsilon", 1e-12))
        delta = prediction_to_raw_laplacian(
            b_output,
            h,
            input_representation=str(b_spec["config"].get("target_mode")),
            eps=epsilon,
        )
        prepared_e = _load_device_item(dataset, index, e_spec["config"], device)
        e_output, e_forward = _forward(prepared_e, e_spec, device)
        current = prepared_e.sample["vertices"].float()
        if tuple(e_output.shape) != tuple(current.shape):
            raise RuntimeError(f"{sample_id}: Arm-E output shape mismatch")
        direct = current + e_output
        direct_np = direct.detach().cpu().numpy().astype(np.float64)

        common = _base_row(
            static, source, split=args.split, index=index, method="arm_e",
            b_spec=b_spec, e_spec=e_spec, manifest_sha=manifest_sha,
        )
        e_row = dict(common)
        e_row.update(_geometry(
            direct_np, faces, initial, gt, before,
            surface_samples=args.surface_samples,
            metric_seed=args.metric_seed,
            fscore_threshold=args.fscore_threshold,
        ))
        e_row.update({
            "lambda": "",
            "model_forward_seconds": e_forward,
            "sparse_solve_seconds": 0.0,
            "total_compute_seconds": e_forward,
            "pcg_iterations": "",
            "pcg_converged": "",
            "pcg_relative_residual": "",
        })
        rows.append(e_row)
        if args.split == "test":
            _save_test_sample(args.output_dir, "arm_e", sample_id, direct_np, faces)

        for regularization in lambdas:
            recovered, solver = _pcg(delta, direct, static, regularization, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{sample_id} lambda={regularization}: PCG failed: {solver}")
            h_row = _base_row(
                static, source, split=args.split, index=index, method="hybrid",
                b_spec=b_spec, e_spec=e_spec, manifest_sha=manifest_sha,
            )
            h_row.update(_geometry(
                recovered, faces, initial, gt, before,
                surface_samples=args.surface_samples,
                metric_seed=args.metric_seed,
                fscore_threshold=args.fscore_threshold,
            ))
            h_row.update(solver)
            h_row.update({
                "lambda": regularization,
                "model_forward_seconds": b_forward + e_forward,
                "sparse_solve_seconds": solver["pcg_runtime_seconds"],
                "total_compute_seconds": b_forward + e_forward + solver["pcg_runtime_seconds"],
            })
            rows.append(h_row)
            if args.split == "test":
                _save_test_sample(args.output_dir, "hybrid", sample_id, recovered, faces)
        if device.type == "cuda":
            peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated(device)))
        print(f"{args.split} {progress}/{len(selected)} {sample_id}", flush=True)

    _write_csv(output_file, rows)
    _write_json(output_file.with_suffix(".metadata.json"), {
        "contract_audit": True,
        "split": args.split,
        "sample_count": len(selected),
        "row_count": len(rows),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "arm_b_checkpoint": str(b_spec["checkpoint"]),
        "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
        "arm_e_checkpoint": str(e_spec["checkpoint"]),
        "arm_e_checkpoint_sha256": e_spec["checkpoint_sha256"],
        "lambda_values": list(lambdas),
        "lambda_lock": None if lock is None else str(args.lambda_lock.resolve()),
        "selection_split": "validation" if lock is not None else None,
        "pcg_tolerance": PCG_TOLERANCE,
        "pcg_maximum_iterations": PCG_MAXIMUM_ITERATIONS,
        "surface_samples": args.surface_samples,
        "metric_seed": args.metric_seed,
        "fscore_threshold": args.fscore_threshold,
        "peak_gpu_memory_bytes": peak_gpu,
        "gt_used_for_prediction_or_recovery": False,
        "models_retrained": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-e-run", required=True, type=Path)
    parser.add_argument("--arm-b-checkpoint-sha256", required=True)
    parser.add_argument("--arm-e-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--lambda-grid", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument("--lambda-lock", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-chunk-size", type=int, default=4)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

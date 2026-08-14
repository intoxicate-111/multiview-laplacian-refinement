#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.controlled_displacement import (
    DIRECT_VERTEX_DISPLACEMENT,
    prediction_semantics,
    recover_direct_displacement,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.evaluation import _reconstruct, evaluate_mesh_geometry
from mlr.learned_laplacian.losses import laplacian_prediction_metrics
from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from mlr.learned_laplacian.target_scaling import prediction_to_raw_laplacian
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.visibility_recovery import (
    confidence_aware_recovery_weight,
)
from mlr.refinement import RefinementConfig


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    manifest = args.manifest.resolve()
    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", "validation", "test")
    }
    validate_disjoint_splits(*datasets.values())
    test = datasets["test"]
    if len(test) != 1000:
        raise ValueError(f"Expected 1000 held-out test samples, found {len(test)}.")

    laplacian = _load_spec(args.laplacian_run, device, expected_semantics=None)
    displacement = _load_spec(
        args.displacement_run,
        device,
        expected_semantics=DIRECT_VERTEX_DISPLACEMENT,
    )
    _assert_fair_pair(laplacian["config"], displacement["config"])
    output = args.output_dir.resolve()
    shard_dir = output / "shards"
    mesh_dir = output / "meshes"
    shard_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index in range(args.shard_index, len(test), args.shard_count):
        static = test.load_static(index)
        prepared = _prepare_item_for_use(
            _prepare_object_static(static, laplacian["config"]),
            laplacian["config"],
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        lap_output, lap_seconds = _forward(laplacian, prepared.sample, device)
        displacement_output, displacement_seconds = _forward(
            displacement, prepared.sample, device
        )
        if lap_output.confidence_prediction is None:
            raise RuntimeError("Laplacian checkpoint has no confidence output.")
        if displacement_output.confidence_prediction is None:
            raise RuntimeError("Displacement checkpoint has no confidence output.")

        current_vertices = static["vertices"].detach().cpu()
        faces = static["faces"].detach().cpu().long()
        gt_vertices = static["gt_vertices"].detach().cpu()
        gt_faces = static["gt_faces"].detach().cpu().long()
        h = prepared.sample["local_edge_length"].float().detach().cpu()
        valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
        lap_prediction_output = lap_output.predicted_laplacian.float().detach().cpu()
        lap_prediction_raw = prediction_to_raw_laplacian(
            lap_prediction_output,
            h,
            input_representation=str(laplacian["config"]["target_mode"]),
            eps=float(
                laplacian["config"].get("target_scaling", {}).get("epsilon", 1e-12)
            ),
        )
        lap_target_raw = static["raw_laplacian_target"].float().detach().cpu()
        lap_confidence = lap_output.confidence_prediction.float().detach().cpu()
        visibility = prepared.sample["visibility"].detach().cpu()
        recovery_weight = confidence_aware_recovery_weight(
            visibility,
            lap_confidence,
            num_vertices=len(current_vertices),
        ).detach().cpu()

        current_mesh = Mesh(current_vertices.numpy(), faces.numpy()).ensure_normals()
        gt_mesh = Mesh(gt_vertices.numpy(), gt_faces.numpy()).ensure_normals()
        recovery_start = time.perf_counter()
        recovered, solver = _reconstruct(
            current_mesh,
            lap_prediction_raw.numpy(),
            np.ones(len(current_vertices), dtype=np.float64),
            _refinement_config(laplacian["config"]),
            int(laplacian["config"].get("recovery", {}).get("dense_vertex_limit", 5000)),
            laplacian_weight=recovery_weight.numpy(),
        )
        lap_recovery_seconds = time.perf_counter() - recovery_start
        direct_prediction = (
            displacement_output.predicted_laplacian.float().detach().cpu()
        )
        direct_vertices = recover_direct_displacement(
            current_vertices, direct_prediction
        )
        direct_mesh = Mesh(direct_vertices.numpy(), faces.numpy()).ensure_normals()

        metrics = {
            "initial": evaluate_mesh_geometry(
                current_mesh,
                gt_mesh,
                surface_samples=args.surface_samples,
                seed=args.metric_seed,
                fscore_threshold=args.fscore_threshold,
            ),
            "laplacian": evaluate_mesh_geometry(
                recovered.mesh,
                gt_mesh,
                surface_samples=args.surface_samples,
                seed=args.metric_seed,
                fscore_threshold=args.fscore_threshold,
            ),
            "displacement": evaluate_mesh_geometry(
                direct_mesh,
                gt_mesh,
                surface_samples=args.surface_samples,
                seed=args.metric_seed,
                fscore_threshold=args.fscore_threshold,
            ),
        }
        sample_id = str(static["sample_id"])
        sample_dir = mesh_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_mesh(current_mesh, sample_dir / "initial.obj")
        save_mesh(recovered.mesh, sample_dir / "laplacian_refined.obj")
        save_mesh(direct_mesh, sample_dir / "displacement_refined.obj")
        row = {
            "sample_id": sample_id,
            "object_id": str(static.get("metadata", {}).get("object_id", "")),
            "variant_index": int(static.get("metadata", {}).get("variant_index", -1)),
            "vertex_count": len(current_vertices),
            "face_count": len(faces),
            "view_count": int(prepared.sample["num_views"]),
            "laplacian_solver": solver,
            "laplacian_forward_seconds": lap_seconds,
            "laplacian_recovery_seconds": lap_recovery_seconds,
            "displacement_forward_seconds": displacement_seconds,
            **_geometry_columns(metrics),
            **_laplacian_columns(
                lap_prediction_raw,
                lap_target_raw,
                lap_confidence,
                visibility.any(dim=0),
                valid,
            ),
            **_displacement_columns(
                direct_prediction,
                static["target_positions"].detach().cpu() - current_vertices,
                displacement_output.confidence_prediction.float().detach().cpu(),
                visibility.any(dim=0),
            ),
        }
        rows.append(row)
        print(
            f"shard {args.shard_index}: {len(rows)} samples; index={index}; "
            f"initial={row['initial_chamfer']:.8g}; "
            f"lap={row['laplacian_chamfer']:.8g}; "
            f"disp={row['displacement_chamfer']:.8g}",
            flush=True,
        )

    csv_path = shard_dir / f"per_sample_shard_{args.shard_index:03d}.csv"
    _write_csv(csv_path, rows)
    metadata = {
        "status": "completed",
        "manifest": str(manifest),
        "test_samples_total": len(test),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_samples": len(rows),
        "surface_samples": args.surface_samples,
        "metric_seed": args.metric_seed,
        "fscore_threshold": args.fscore_threshold,
        "laplacian": _serializable_spec(laplacian),
        "displacement": _serializable_spec(displacement),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else None
        ),
        "csv": str(csv_path),
    }
    (shard_dir / f"metadata_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _load_spec(
    run_dir: Path, device: torch.device, expected_semantics: str | None
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config = run_payload.get("experiment_config", run_payload)
    semantic = prediction_semantics(config)
    if expected_semantics is not None and semantic != expected_semantics:
        raise ValueError(
            f"Expected {expected_semantics!r} checkpoint, found {semantic!r}."
        )
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "checkpoint_latest.pt"
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "run_dir": run_dir,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "prediction_semantics": semantic,
    }


def _forward(spec: Mapping[str, Any], sample: Mapping[str, Any], device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        output = spec["model"](sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def _refinement_config(config: Mapping[str, Any]) -> RefinementConfig:
    value = config.get("recovery", {})
    return RefinementConfig(
        operator_type=str(value.get("operator_type", "uniform")),
        lambda_lap=float(value.get("lambda_lap", 1.0)),
        lambda_anchor=float(value.get("lambda_anchor", 0.01)),
        lambda_edge=float(value.get("lambda_edge", 0.0)),
        lambda_unseen_anchor=float(value.get("unseen_anchor_weight", 0.0)),
        num_iters=int(value.get("num_iters", 200)),
        learning_rate=float(value.get("learning_rate", 0.01)),
        robust_loss=str(value.get("robust_loss", "huber")),
        huber_delta=float(value.get("huber_delta", 0.01)),
    )


def _geometry_columns(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    fields = (
        "chamfer",
        "point_to_surface_bidirectional_mean",
        "point_to_surface_bidirectional_p95",
        "fscore",
        "normal_consistency",
    )
    for method, values in metrics.items():
        for field in fields:
            short = field.replace("point_to_surface_bidirectional_", "p2s_")
            result[f"{method}_{short}"] = values[field]
    initial = float(metrics["initial"]["chamfer"])
    for method in ("laplacian", "displacement"):
        refined = float(metrics[method]["chamfer"])
        result[f"{method}_chamfer_improvement_rate"] = (
            (initial - refined) / initial if initial > 0 else 0.0
        )
        result[f"{method}_improved"] = refined < initial
    return result


def _laplacian_columns(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    visible: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    metrics = laplacian_prediction_metrics(prediction, target, valid_mask=valid)
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {
        "laplacian_raw_target_epe": float(error[valid].mean()),
        "laplacian_raw_cosine": float(metrics["global_cosine"]),
        "laplacian_visible_raw_epe": _masked_mean(error, visible & valid),
        "laplacian_unseen_raw_epe": _masked_mean(error, (~visible) & valid),
        **_confidence_columns("laplacian", confidence, visible),
    }


def _displacement_columns(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, Any]:
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {
        "displacement_target_epe": float(error.mean()),
        "displacement_visible_epe": _masked_mean(error, visible),
        "displacement_unseen_epe": _masked_mean(error, ~visible),
        **_confidence_columns("displacement", confidence, visible),
    }


def _confidence_columns(
    prefix: str, confidence: torch.Tensor, visible: torch.Tensor
) -> dict[str, Any]:
    return {
        f"{prefix}_confidence_mean": float(confidence.mean()),
        f"{prefix}_confidence_std": float(confidence.std(unbiased=False)),
        f"{prefix}_confidence_min": float(confidence.min()),
        f"{prefix}_confidence_max": float(confidence.max()),
        f"{prefix}_visible_confidence_mean": _masked_mean(confidence, visible),
        f"{prefix}_unseen_confidence_mean": _masked_mean(confidence, ~visible),
        f"{prefix}_visible_vertices": int(visible.sum()),
        f"{prefix}_unseen_vertices": int((~visible).sum()),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = values[mask]
    return None if selected.numel() == 0 else float(selected.mean())


def _assert_fair_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    ignored = {
        "method",
        "prediction_semantics",
        "target_semantics",
        "target_definition",
        "recovery",
        "experiment_metadata",
    }
    left_control = {key: value for key, value in left.items() if key not in ignored}
    right_control = {key: value for key, value in right.items() if key not in ignored}
    if left_control != right_control:
        raise ValueError("Laplacian/displacement configs are not a controlled pair.")


def _serializable_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(spec["run_dir"]),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "prediction_semantics": spec["prediction_semantics"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Evaluation shard produced no rows.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--laplacian-run", required=True, type=Path)
    parser.add_argument("--displacement-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--surface-samples", default=3000, type=int)
    parser.add_argument("--metric-seed", default=7, type=int)
    parser.add_argument("--fscore-threshold", default=0.01, type=float)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

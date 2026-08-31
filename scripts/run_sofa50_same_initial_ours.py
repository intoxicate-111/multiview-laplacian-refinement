#!/usr/bin/env python3
from __future__ import annotations

"""Run the frozen canonical HF learned-Laplacian model for the controlled benchmark."""

import argparse
import copy
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    _infer_one,
    _raw_metrics,
    _recover_raw_one,
    _run_config,
)
from mlr.learned_laplacian.synthetic_current_hf_resolution_ablation import _sample_gt_groups
from mlr.learned_laplacian.trainer import load_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): dict(row) for row in payload["samples"]}


def spec(
    run_dir: Path,
    device: torch.device,
    *,
    view_chunk_size: int | None = None,
    checkpoint_name: str = "checkpoint_latest.pt",
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    source_config = _run_config(run_dir)
    config = copy.deepcopy(source_config)
    if view_chunk_size is not None:
        if view_chunk_size < 1:
            raise ValueError("view_chunk_size must be positive")
        config.setdefault("image_encoder", {})["view_chunk_size"] = view_chunk_size
    checkpoint = run_dir / checkpoint_name
    if checkpoint.name != checkpoint_name or checkpoint.parent != run_dir:
        raise ValueError("checkpoint_name must be a plain file name")
    checkpoint_sha256 = sha256(checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, found {checkpoint_sha256}"
        )
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    expected_optimizer_steps = int(
        source_config.get("multi_object_training", {}).get("max_optimizer_steps", -1)
    )
    optimizer_steps_value = payload.get("optimizer_steps")
    actual_optimizer_steps = (
        None if optimizer_steps_value is None else int(optimizer_steps_value)
    )
    if checkpoint_name == "checkpoint_latest.pt" and (
        expected_optimizer_steps < 1
        or actual_optimizer_steps != expected_optimizer_steps
    ):
        raise ValueError(
            "Checkpoint is not the completed configured run: "
            f"expected {expected_optimizer_steps}, found {actual_optimizer_steps}"
        )
    if checkpoint_name != "checkpoint_latest.pt" and expected_checkpoint_sha256 is None:
        raise ValueError(
            "A non-latest checkpoint requires expected_checkpoint_sha256 for a "
            "frozen-selection audit"
        )
    return {
        "config": config,
        "source_config": source_config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(payload["epoch"]),
        "optimizer_steps": actual_optimizer_steps,
        "inference_view_chunk_size": view_chunk_size,
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--view-chunk-size",
        type=int,
        help="Execution-only image-view chunking; this does not alter model weights.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != 25:
        raise ValueError(f"Expected 25 test samples, found {len(dataset)}")
    provenance = benchmark_rows(args.manifest)
    selected = [
        index
        for index, sample_id in enumerate(dataset.sample_ids)
        if index % args.shard_count == args.shard_index
        and (args.sample_id is None or sample_id == args.sample_id)
    ]
    if args.sample_id is not None and len(selected) != 1:
        raise ValueError(f"Expected one selected sample, found {selected}")
    model_spec = spec(
        args.run_dir.resolve(),
        device,
        view_chunk_size=args.view_chunk_size,
    )
    rows = []
    for index in selected:
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        initial_path = Path(source["common_initial_mesh"])
        if sha256(initial_path) != source["common_initial_mesh_sha256"]:
            raise RuntimeError(f"Common initial SHA changed: {sample_id}")
        initial = load_mesh(initial_path)
        faces = static["faces"].detach().cpu().numpy()
        vertices = static["vertices"].detach().cpu().numpy()
        identity = bool(
            initial.num_vertices == len(vertices)
            and initial.num_faces == len(faces)
            and np.array_equal(initial.faces, faces)
            and float(np.max(np.abs(initial.vertices - vertices))) <= 1e-6
        )
        if not identity:
            raise RuntimeError(f"Common initial identity failed: {sample_id}")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        values = _infer_one(
            dataset,
            index,
            model_spec,
            device,
            current_faces=static["faces"],
        )
        sample_dir = args.output_dir / "samples" / sample_id
        recovery_dir = sample_dir / "recovery"
        recovery, _ = _recover_raw_one(
            static,
            values["prediction_raw"],
            values["prediction_normalized"],
            values["confidence"],
            recovery_dir,
            model_spec["config"],
        )
        canonical_final = sample_dir / "refined.obj"
        shutil.copy2(recovery_dir / "predicted_refined.obj", canonical_final)
        runtime = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        np.save(sample_dir / "predicted_raw_laplacian.npy", values["prediction_raw"].numpy())
        np.save(sample_dir / "predicted_confidence.npy", values["confidence"].numpy())
        np.save(sample_dir / "recovery_weight.npy", values["recovery_weight"].numpy())
        np.savez_compressed(
            sample_dir / "visibility_used.npz",
            visibility=static["visibility_backface_and_occlusion"].numpy(),
        )
        (sample_dir / "recovery_config.json").write_text(
            json.dumps(model_spec["config"]["recovery"], indent=2) + "\n",
            encoding="utf-8",
        )
        (sample_dir / "method_config.json").write_text(
            json.dumps(model_spec["config"], indent=2) + "\n", encoding="utf-8"
        )
        metrics = _raw_metrics(
            values["prediction_raw"],
            values["target_raw"],
            values["recovery_weight"],
            values["valid"],
        )
        groups = _sample_gt_groups(
            values["prediction_raw"], values["target_raw"], values["valid"]
        )
        row = {
            "sample_id": sample_id,
            "method": "ours",
            "status": "completed",
            "common_initial_mesh": str(initial_path),
            "common_initial_mesh_sha256": source["common_initial_mesh_sha256"],
            "initial_vertex_count": initial.num_vertices,
            "initial_face_count": initial.num_faces,
            "common_initial_identity_audit": identity,
            "image_directory": source["image_directory"],
            "camera_and_gt_container": source["camera_and_gt_container"],
            "view_count": 28,
            "checkpoint": str(model_spec["checkpoint"]),
            "checkpoint_sha256": model_spec["checkpoint_sha256"],
            "checkpoint_optimizer_steps": model_spec["optimizer_steps"],
            "inference_view_chunk_size": model_spec["inference_view_chunk_size"],
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": peak,
            "method_config_path": str(sample_dir / "method_config.json"),
            "final_mesh": str(canonical_final),
            "final_vertex_count": int(recovery["recovered_vertices"] if "recovered_vertices" in recovery else initial.num_vertices),
            "final_face_count": initial.num_faces,
            "coordinate_transform_to_gt": "identity",
            "output_connectivity_preserved": True,
            **metrics,
            **groups,
            **recovery,
        }
        # Recovery metrics do not expose counts; canonical recovery preserves V/F.
        row["final_vertex_count"] = initial.num_vertices
        row["final_face_count"] = initial.num_faces
        (sample_dir / "status.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"ours sample={sample_id} chamfer={row['reconstruction_chamfer']:.9g}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_dir / f"per_sample_shard_{args.shard_index:03d}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"method": "ours", "completed": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

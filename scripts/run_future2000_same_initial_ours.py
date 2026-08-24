#!/usr/bin/env python3
from __future__ import annotations

"""Run one frozen learned-Laplacian arm on selected Future2000 test samples."""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_future2000_external_baseline import _audit_source_identity, _evaluate
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    _infer_one,
    _recover_raw_one,
)
from run_sofa50_same_initial_ours import spec


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != args.expected_test_samples:
        raise ValueError(
            f"Expected {args.expected_test_samples} test samples, found {len(dataset)}"
        )
    if args.selection is None:
        selected_ids = [str(value) for value in dataset.sample_ids]
    else:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selected_ids = [str(value) for value in selection["sample_ids"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selection contains duplicate sample IDs")
    unknown = sorted(set(selected_ids) - set(dataset.sample_ids))
    if unknown:
        raise ValueError(f"Selected sample IDs are absent from test split: {unknown}")
    if args.sample_id is None:
        assigned_ids = selected_ids[args.shard_index :: args.shard_count]
    else:
        if args.sample_id not in selected_ids:
            raise ValueError("--sample-id is not part of the frozen selection")
        assigned_ids = [args.sample_id]

    provenance_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = {
        str(row["sample_id"]): dict(row) for row in provenance_payload["samples"]
    }
    index_by_id = {str(value): index for index, value in enumerate(dataset.sample_ids)}
    model_spec = spec(
        args.run_dir.resolve(),
        device,
        view_chunk_size=args.view_chunk_size,
    )
    output = args.output_dir.resolve() / "ours"
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for sample_id in assigned_ids:
        index = index_by_id[sample_id]
        static = dataset.load_static(index)
        source_identity = _audit_source_identity(static, provenance[sample_id])
        sample_dir = output / "samples" / sample_id
        recovery_dir = sample_dir / "recovery"
        sample_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        values = _infer_one(
            dataset,
            index,
            model_spec,
            device,
            current_faces=static["faces"],
        )
        recovery, _ = _recover_raw_one(
            static,
            values["prediction_raw"],
            values["prediction_normalized"],
            values["confidence"],
            recovery_dir,
            model_spec["config"],
        )
        final_mesh = sample_dir / "refined.obj"
        shutil.copy2(recovery_dir / "predicted_refined.obj", final_mesh)
        current_mesh = Mesh(
            static["vertices"].detach().cpu().numpy(),
            static["faces"].detach().cpu().numpy(),
        ).ensure_normals()
        save_mesh(current_mesh, sample_dir / "initial.obj")
        from mlr.io import load_mesh

        refined = load_mesh(final_mesh).ensure_normals()
        metrics = _evaluate(static, refined, args)
        runtime = time.perf_counter() - started
        peak = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else None
        )
        np.save(sample_dir / "predicted_raw_laplacian.npy", values["prediction_raw"].numpy())
        np.save(sample_dir / "predicted_confidence.npy", values["confidence"].numpy())
        row = {
            "sample_id": sample_id,
            "method": "ours",
            "status": "completed",
            "failure_stage": "",
            "failure_reason": "",
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": peak,
            "vertex_count": refined.num_vertices,
            "face_count": refined.num_faces,
            "final_mesh": str(final_mesh),
            "coordinate_transform_to_gt": "identity",
            "method_config_path": str(args.run_dir.resolve() / "run_config.json"),
            "checkpoint": str(model_spec["checkpoint"]),
            "checkpoint_sha256": model_spec["checkpoint_sha256"],
            "checkpoint_optimizer_steps": model_spec["optimizer_steps"],
            "inference_view_chunk_size": model_spec["inference_view_chunk_size"],
            **source_identity,
            "adapter_initial_mesh_sha256": source_identity["common_initial_mesh_sha256"],
            "adapter_initial_vertex_count": source_identity["initial_vertex_count"],
            "adapter_initial_face_count": source_identity["initial_face_count"],
            "adapter_initial_max_abs_vertex_error": 0.0,
            "adapter_initial_faces_exact": True,
            "common_initial_identity_audit": True,
            **metrics,
        }
        (sample_dir / "status.json").write_text(
            json.dumps({"status": "completed", "row": row}, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        print(
            f"ours shard={args.shard_index} sample={sample_id} "
            f"chamfer={row['refined_chamfer']:.9g}",
            flush=True,
        )

    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    csv_path = shard_dir / f"per_sample_shard_{args.shard_index:03d}.csv"
    if not rows:
        raise ValueError("Ours shard has no assigned samples")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "method": "ours",
        "status": "completed",
        "pinned_commit": model_spec["checkpoint_sha256"],
        "repository": "intoxicate-111/multiview-laplacian-refinement",
        "manifest": str(args.manifest.resolve()),
        "selection": str(args.selection.resolve()) if args.selection is not None else None,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_samples": len(rows),
        "completed_samples": len(rows),
        "failed_samples": 0,
        "checkpoint_optimizer_steps": model_spec["optimizer_steps"],
        "csv": str(csv_path),
    }
    (shard_dir / f"metadata_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Optional frozen subset; omit to evaluate the complete test split.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-test-samples", type=int, default=1000)
    parser.add_argument("--view-chunk-size", type=int, default=4)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

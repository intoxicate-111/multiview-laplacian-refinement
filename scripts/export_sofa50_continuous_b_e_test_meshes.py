#!/usr/bin/env python3
from __future__ import annotations

"""Export refined test meshes from a continuous pretrained B+E checkpoint."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(sample_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")
    return value or "sample"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        required=True,
        action="append",
        nargs=2,
        metavar=("LABEL", "MANIFEST"),
        help="Dataset label and prepared manifest; may be repeated.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regularization", type=float)
    parser.add_argument("--maximum-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.distributed.init_process_group(backend="gloo")

    run = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    run_payload = _read(run / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    settings = config["training"]["hybrid_single_geometry_loss"]
    regularization = (
        float(args.regularization)
        if args.regularization is not None
        else float(settings["lambda"])
    )
    maximum_iterations = (
        int(args.maximum_iterations)
        if args.maximum_iterations is not None
        else int(settings["maximum_iterations"])
    )
    tolerance = (
        float(args.tolerance)
        if args.tolerance is not None
        else float(settings["tolerance"])
    )

    if args.device == "cuda" and distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Run config did not instantiate two complete B/E networks")
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256(checkpoint)
    export_rows: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    global_index = 0

    for label, manifest_text in args.dataset:
        manifest = Path(manifest_text).resolve()
        dataset = PreparedMeshDataset.from_manifest(manifest, "test")
        dataset_output = output_dir / label
        dataset_output.mkdir(parents=True, exist_ok=True)
        dataset_counts[label] = len(dataset)

        for index in range(len(dataset)):
            assigned_rank = global_index % world_size
            global_index += 1
            if assigned_rank != rank:
                continue
            static = dataset.load_static(index)
            prepared = _load_device_item(dataset, index, config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                prediction = model(conditioned)
            direct = prediction.direct_vertex_displacement_prediction
            if direct is None:
                raise RuntimeError("Continuous B/E checkpoint omitted the direct branch")
            recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
                prediction.predicted_laplacian.detach().double(),
                prepared.sample["vertices"].double() + direct.detach().double(),
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"].double(),
                regularization=regularization,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            sample_id = str(static["sample_id"])
            if not audit.converged:
                raise RuntimeError(f"{label}/{sample_id}: PCG did not converge")

            faces = np.asarray(static["faces"], dtype=np.int64)
            mesh = Mesh(
                recovered.detach().cpu().numpy(), faces.copy()
            ).ensure_normals()
            mesh_path = dataset_output / f"{index:03d}_{_safe_name(sample_id)}.obj"
            save_mesh(mesh, mesh_path)
            export_rows.append(
                {
                    "dataset": label,
                    "sample_index": index,
                    "sample_id": sample_id,
                    "mesh": str(mesh_path.relative_to(output_dir)),
                    "mesh_sha256": _sha256(mesh_path),
                    "vertices": int(mesh.num_vertices),
                    "faces": int(mesh.num_faces),
                    "pcg_iterations": int(audit.iterations),
                    "pcg_relative_residual": float(audit.relative_residual),
                }
            )
            print(
                f"export rank={rank}/{world_size} {label} "
                f"{index + 1}/{len(dataset)} {sample_id}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "read_only_checkpoint": True,
        "split": "test",
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "shard": {"rank": rank, "world_size": world_size},
        "recovery": {
            "operator": "uniform_random_walk_current_graph",
            "lambda": regularization,
            "tolerance": tolerance,
            "maximum_iterations": maximum_iterations,
        },
        "dataset_counts": dataset_counts,
        "total_meshes": len(export_rows),
        "meshes": export_rows,
    }
    shard_manifest_path = output_dir / (
        f"EXPORT_MANIFEST.shard{rank:03d}-of-{world_size:03d}.json"
    )
    shard_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if distributed:
        torch.distributed.barrier()
    if rank != 0:
        torch.distributed.destroy_process_group()
        return 0

    if distributed:
        merged_rows: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            path = output_dir / (
                f"EXPORT_MANIFEST.shard{shard_rank:03d}-of-{world_size:03d}.json"
            )
            shard_payload = _read(path)
            if shard_payload["checkpoint_sha256"] != checkpoint_sha256:
                raise RuntimeError(f"Checkpoint mismatch in {path}")
            merged_rows.extend(shard_payload["meshes"])
        export_rows = sorted(
            merged_rows,
            key=lambda row: (str(row["dataset"]), int(row["sample_index"])),
        )
        payload["meshes"] = export_rows
        payload["total_meshes"] = len(export_rows)
        payload["shard"] = {"rank": "merged", "world_size": world_size}

    expected_total = sum(dataset_counts.values())
    if len(export_rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} exported meshes, found {len(export_rows)}"
        )
    manifest_path = output_dir / "EXPORT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_counts": dataset_counts,
                "total_meshes": len(export_rows),
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if distributed:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

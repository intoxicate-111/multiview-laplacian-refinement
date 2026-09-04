#!/usr/bin/env python3
"""Cache audited frozen Arm-E positional anchors for all Sofa50-v2 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.distributed import (
    destroy_distributed,
    distributed_barrier,
    initialize_distributed,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


EXPECTED_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"
EXPECTED_COUNTS = {"train": 400, "validation": 50, "test": 50}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def archived_predictions(report: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shard = report / "shards" / "E_direct_vertex_residual.json"
    arrays_path = report / "shards" / "E_direct_vertex_residual_prediction_arrays.npz"
    payload = json.loads(shard.read_text(encoding="utf-8"))
    if payload["checkpoint_sha256"] != EXPECTED_E_SHA256:
        raise RuntimeError("Archived Arm-E prediction metadata has the wrong checkpoint")
    arrays = np.load(arrays_path, allow_pickle=False)
    result: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        ids = [str(item) for item in payload["split_ids"][split]]
        rows = [row for row in payload["rows"] if row["split"] == split]
        if [str(row["sample_id"]) for row in rows] != ids:
            raise RuntimeError(f"Archived Arm-E {split} row order differs from split IDs")
        flat = np.asarray(arrays[f"{split}_prediction"], dtype=np.float64)
        offset = 0
        for sample_id, row in zip(ids, rows):
            count = int(row["vertices"])
            result[sample_id] = flat[offset : offset + count].copy()
            offset += count
        if offset != len(flat):
            raise RuntimeError(f"Archived Arm-E {split} vertex counts do not close")
    return result, {
        "metadata": str(shard.resolve()),
        "metadata_sha256": sha256_file(shard),
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256_file(arrays_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-e-config", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()

    context = initialize_distributed(args.device)
    try:
        if context.world_size != 4:
            raise RuntimeError("Frozen-anchor cache must run on exactly four ranks")
        checkpoint = args.arm_e_checkpoint.resolve()
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != EXPECTED_E_SHA256:
            raise RuntimeError(f"Arm-E checkpoint SHA mismatch: {checkpoint_sha}")
        config = json.loads(args.arm_e_config.read_text(encoding="utf-8"))
        if config.get("prediction_semantics") != "direct_vertex_displacement":
            raise RuntimeError("Arm-E config is not direct vertex displacement")
        model = _build_model(config, None, False).to(context.device)
        load_checkpoint(checkpoint, model, map_location=context.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        amp_enabled, amp_dtype = _amp_settings(config, context.device)
        archived, archive_audit = archived_predictions(args.arm_e_report.resolve())

        output = args.output_dir.resolve()
        anchors_dir = output / "anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        global_order = 0
        for split in ("train", "validation", "test"):
            dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
            if len(dataset) != EXPECTED_COUNTS[split]:
                raise RuntimeError(f"Unexpected {split} count: {len(dataset)}")
            for index in range(len(dataset)):
                order = global_order + index
                if order % context.world_size != context.rank:
                    continue
                static = dataset.load_static(index)
                sample_id = str(static["sample_id"])
                if split == "train":
                    prepared = _load_device_item(dataset, index, config, context.device)
                    conditioned = _exact_query_sample(prepared.sample, context.device)
                    with torch.inference_mode(), torch.autocast(
                        device_type=context.device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        output_value = model(conditioned)
                    displacement = (
                        output_value.predicted_laplacian.detach().float().cpu().numpy()
                    )
                    source_name = "fresh_frozen_checkpoint_inference"
                else:
                    # These are the exact already-locked arrays used by all prior
                    # matched-v2 E and B+E reports. Re-running the same checkpoint
                    # on a different GPU architecture is not bitwise invariant,
                    # so reusing the archive is the only way to obey the contract
                    # that E's validation/test predictions must not change.
                    displacement = np.asarray(archived[sample_id], dtype=np.float32)
                    source_name = "exact_locked_archived_prediction_array"
                vertices = np.asarray(static["vertices"], dtype=np.float32)
                if displacement.shape != vertices.shape:
                    raise RuntimeError(f"{sample_id}: Arm-E output shape mismatch")
                anchor = np.ascontiguousarray(vertices + displacement, dtype=np.float32)
                safe = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
                relative = Path("anchors") / f"{order:04d}_{safe}.npy"
                path = output / relative
                np.save(path, anchor, allow_pickle=False)
                records.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "split_order": index,
                        "global_order": order,
                        "path": str(relative),
                        "vertex_count": int(anchor.shape[0]),
                        "initial_vertices_sha256": array_sha256(vertices),
                        "displacement_sha256": array_sha256(displacement),
                        "anchor_sha256": sha256_file(path),
                        "detached": True,
                        "source": source_name,
                    }
                )
            global_order += len(dataset)

        shard = {
            "rank": context.rank,
            "world_size": context.world_size,
            "records": records,
        }
        (output / f"shard_rank{context.rank:02d}.json").write_text(
            json.dumps(shard, indent=2) + "\n", encoding="utf-8"
        )
        distributed_barrier(context)
        if context.is_main:
            merged: list[dict[str, Any]] = []
            for rank in range(context.world_size):
                value = json.loads(
                    (output / f"shard_rank{rank:02d}.json").read_text(encoding="utf-8")
                )
                merged.extend(value["records"])
            merged.sort(key=lambda row: int(row["global_order"]))
            if len(merged) != 500 or len({row["sample_id"] for row in merged}) != 500:
                raise RuntimeError("Frozen anchor merge did not produce 500 unique samples")
            if [int(row["global_order"]) for row in merged] != list(range(500)):
                raise RuntimeError("Frozen anchor global ordering is incomplete")
            counts = {
                split: sum(row["split"] == split for row in merged)
                for split in EXPECTED_COUNTS
            }
            if counts != EXPECTED_COUNTS:
                raise RuntimeError(f"Frozen anchor split counts differ: {counts}")
            metadata = {
                "contract_audit": True,
                "format": "sofa50_frozen_arm_e_positional_anchor_v1",
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256_file(args.manifest.resolve()),
                "arm_e_config": str(args.arm_e_config.resolve()),
                "arm_e_config_sha256": sha256_file(args.arm_e_config.resolve()),
                "arm_e_checkpoint": str(checkpoint),
                "arm_e_checkpoint_sha256": checkpoint_sha,
                "arm_e_mode": "eval_inference_mode_all_parameters_requires_grad_false",
                "anchor_equation": "V_P = V_0 + DeltaV_E",
                "model_input_exclusion": "cache is attached only after dataset load and removed before predictor forward",
                "world_size": context.world_size,
                "split_counts": counts,
                "archived_validation_test_audit": archive_audit,
                "validation_test_prediction_policy": (
                    "exact locked archived Arm-E arrays; no validation/test reinference"
                ),
                "train_prediction_policy": (
                    "frozen eval inference on the training inputs using the exact checkpoint"
                ),
                "records": merged,
            }
            metadata_path = output / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({
                "metadata": str(metadata_path),
                "records": len(merged),
                "split_counts": counts,
                "validation_test_arrays_reused_exactly": True,
            }, indent=2), flush=True)
        distributed_barrier(context)
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()

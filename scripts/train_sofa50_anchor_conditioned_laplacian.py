#!/usr/bin/env python3
"""Train the single controlled Sofa50 Arm-B_P frozen-anchor ablation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_multi_mesh_laplacian as base
from mlr.learned_laplacian.distributed import (
    destroy_distributed,
    distributed_barrier,
    initialize_distributed,
)
from mlr.learned_laplacian.frozen_anchor_cache import FrozenAnchorCache, FrozenAnchorDataset
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from mlr.learned_laplacian.multi_trainer import train_multi_object


EXPECTED_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--anchor-cache-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    controlled = config.get("controlled_ablation", {})
    recovery = config["training"]["recovery_aware_geometry_loss"]
    if controlled.get("arm") != "B_P":
        raise RuntimeError("Config is not the controlled Arm-B_P ablation")
    if recovery.get("anchor_mode") != "cached_frozen_vertices":
        raise RuntimeError("Arm-B_P config is not using the frozen positional anchor")
    if float(recovery.get("lambda")) != 0.01 or float(recovery.get("beta")) != 0.01:
        raise RuntimeError("Arm-B_P lambda/beta differ from the locked contract")
    if int(config["multi_object_training"]["max_optimizer_steps"]) != 20_000:
        raise RuntimeError("Arm-B_P must run exactly 20,000 optimizer steps")

    context = initialize_distributed(args.device)
    try:
        accumulation = int(
            config["multi_object_training"]["gradient_accumulation_meshes"]
        )
        if context.world_size != 4 or context.world_size * accumulation != 8:
            raise RuntimeError(
                "Arm-B_P requires exactly four ranks and effective global batch 8"
            )
        base._validate_expected_split_counts(args.manifest, config)
        cache = FrozenAnchorCache(
            args.anchor_cache_metadata,
            expected_checkpoint_sha256=EXPECTED_E_SHA256,
        )
        if context.is_main:
            base._write_run_metadata(args.output_dir, args.manifest, config)
        distributed_barrier(context)

        train_base = PreparedMeshDataset.from_manifest(args.manifest, "train")
        validation_base = PreparedMeshDataset.from_manifest(args.manifest, "validation")
        validate_disjoint_splits(train_base, validation_base)
        train_dataset = FrozenAnchorDataset(train_base, cache)
        validation_dataset = FrozenAnchorDataset(validation_base, cache)
        loading_seconds = 0.0
        if context.is_main:
            print(
                f"registered {len(train_dataset)} frozen-anchor training meshes and "
                f"{len(validation_dataset)} validation meshes; "
                f"world_size={context.world_size}, accumulation={accumulation}, "
                f"effective_global_batch={context.world_size * accumulation}",
                flush=True,
            )
        started = time.perf_counter()
        result = train_multi_object(
            train_dataset,
            validation_dataset,
            config,
            output_dir=args.output_dir,
            device_override=str(context.device),
            initial_loading_seconds=loading_seconds,
        )
        if context.is_main:
            summary = {
                "arm": "B_P",
                "train_meshes": len(train_dataset),
                "validation_meshes": len(validation_dataset),
                "best_epoch": result.best_epoch,
                "best_selection_loss": result.best_selection_loss,
                "optimizer_steps": result.optimizer_steps,
                "distributed_world_size": result.distributed_world_size,
                "effective_global_batch": context.world_size * accumulation,
                "runtime_seconds": result.runtime_seconds,
                "wrapper_wall_seconds": time.perf_counter() - started,
                "output_dir": str(args.output_dir.resolve()),
            }
            (args.output_dir / "bp_training_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, indent=2), flush=True)
        distributed_barrier(context)
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()

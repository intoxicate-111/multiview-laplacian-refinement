#!/usr/bin/env python3
"""Build the exact Arm-B_P config from the locked Arm-B_0 config."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_locked_b0(config: dict[str, Any]) -> None:
    training = config["training"]
    recovery = training["recovery_aware_geometry_loss"]
    multi = config["multi_object_training"]
    expected = {
        "target_mode": "raw_laplacian",
        "loss": "huber",
        "huber_delta": 0.01,
        "recovery_enabled": True,
        "recovery_lambda": 0.01,
        "recovery_beta": 0.01,
        "maximum_iterations": 256,
        "tolerance": 0.0001,
        "max_optimizer_steps": 20_000,
        "seed": 7,
    }
    actual = {
        "target_mode": config.get("target_mode"),
        "loss": training.get("loss"),
        "huber_delta": training.get("huber_delta"),
        "recovery_enabled": recovery.get("enabled"),
        "recovery_lambda": recovery.get("lambda"),
        "recovery_beta": recovery.get("beta"),
        "maximum_iterations": recovery.get("maximum_iterations"),
        "tolerance": recovery.get("tolerance"),
        "max_optimizer_steps": multi.get("max_optimizer_steps"),
        "seed": config.get("seed"),
    }
    if actual != expected:
        raise RuntimeError(f"Arm-B_0 config is not the locked contract: {actual!r}")
    if config["dataset"]["expected_split_counts"] != {
        "train": 400,
        "validation": 50,
        "test": 50,
    }:
        raise RuntimeError("Arm-B_0 split contract differs from 400/50/50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-b0-config", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--anchor-cache-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=8)
    args = parser.parse_args()

    if args.world_size != 4:
        raise RuntimeError("This controlled run requires exactly four GPUs")
    if args.global_batch != 8 or args.global_batch % args.world_size:
        raise RuntimeError("The controlled run requires global batch 8")
    e_checkpoint = args.arm_e_checkpoint.resolve()
    e_sha = sha256_file(e_checkpoint)
    if e_sha != EXPECTED_E_SHA256:
        raise RuntimeError(f"Arm-E checkpoint SHA mismatch: {e_sha}")

    source = json.loads(args.arm_b0_config.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise TypeError("Arm-B_0 config must be a JSON object")
    require_locked_b0(source)
    config = copy.deepcopy(source)
    recovery = config["training"]["recovery_aware_geometry_loss"]
    recovery["anchor_mode"] = "cached_frozen_vertices"
    recovery["frozen_anchor_source"] = "Arm-E direct displacement, detached"
    config["multi_object_training"]["gradient_accumulation_meshes"] = (
        args.global_batch // args.world_size
    )
    config["recovery"]["anchor"] = "lambda_times_frozen_arm_e_positional_vertex_l2"
    config["controlled_ablation"] = {
        "arm": "B_P",
        "comparison_reference": "B_0",
        "only_algorithmic_change": "recovery anchor V0 -> frozen V_P",
        "initialization": "from_scratch_same_seed",
        "arm_b0_config": str(args.arm_b0_config.resolve()),
        "arm_b0_config_sha256": sha256_file(args.arm_b0_config.resolve()),
        "arm_e_checkpoint": str(e_checkpoint),
        "arm_e_checkpoint_sha256": e_sha,
        "anchor_cache_metadata": str(args.anchor_cache_metadata.resolve()),
        "world_size": args.world_size,
        "gradient_accumulation_meshes_per_rank": args.global_batch // args.world_size,
        "effective_global_batch": args.global_batch,
        "gpu_type": "NVIDIA L40",
        "execution_note": (
            "B_P uses 4 L40 ranks with accumulation 2. This preserves B_0's "
            "effective global batch 8 and 50 optimizer steps per 400-mesh epoch."
        ),
    }
    require_locked_b0(config)
    if recovery["anchor_mode"] != "cached_frozen_vertices":
        raise AssertionError("Failed to set frozen positional anchor")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config["controlled_ablation"], indent=2))


if __name__ == "__main__":
    main()

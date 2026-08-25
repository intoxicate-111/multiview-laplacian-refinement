#!/usr/bin/env python3
from __future__ import annotations

"""Derive controlled Cotangent single-loss hybrid configs from Arm U."""

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _lambda_tag(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lambda", dest="regularization", required=True, type=float)
    parser.add_argument("--max-optimizer-steps", type=int, default=20000)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.regularization <= 0 or args.max_optimizer_steps < 1:
        raise ValueError("lambda and max-optimizer-steps must be positive.")
    if args.pilot and args.world_size != 2:
        raise ValueError("The controlled pilot uses exactly 2 L40 GPUs.")
    if not args.pilot and args.world_size != 8:
        raise ValueError("The controlled full run uses exactly 8 Blackwell GPUs.")

    config = _read(args.uniform_config.resolve())
    settings = config["training"]["hybrid_single_geometry_loss"]
    settings["operator"] = "symmetric_cotangent_stiffness"
    settings["cotangent_relative_area_epsilon"] = 1e-12
    settings["lambda"] = args.regularization
    multi = config["multi_object_training"]
    multi["max_optimizer_steps"] = args.max_optimizer_steps
    multi["gradient_accumulation_meshes"] = 4 if args.pilot else 1
    if args.pilot:
        multi["checkpoint_optimizer_steps"] = [args.max_optimizer_steps]
    else:
        multi["checkpoint_optimizer_steps"] = [5000, 10000, 15000, 20000]

    tag = _lambda_tag(args.regularization)
    config["method"] = "sofa50_v2_cotangent_hybrid_single_loss"
    config["recovery"]["operator"] = (
        "symmetric_cotangent_stiffness_no_mass_normalization"
    )
    config["recovery"]["lambda"] = args.regularization
    metadata = config["experiment_metadata"]
    metadata.update(
        {
            "experiment": "Sofa50_v2_uniform_vs_cotangent_single_loss_hybrid",
            "arm": f"cotangent_single_loss_hybrid_lambda{tag}",
            "operator": "symmetric_cotangent_stiffness",
            "operator_geometry_source": "input_coarse_mesh_vertices_and_faces",
            "mass_normalization": False,
            "negative_cotangent_weights_retained": True,
            "cotangent_relative_area_epsilon": 1e-12,
            "fixed_lambda_source": (
                "validation_only_short_matched_pilot"
                if not args.pilot
                else "predeclared_validation_pilot_grid"
            ),
            "distributed_world_size": args.world_size,
            "gradient_accumulation_meshes_per_rank": 4 if args.pilot else 1,
            "effective_global_batch_meshes": 8,
            "training_gpu_model": (
                "NVIDIA L40" if args.pilot else "NVIDIA RTX PRO 6000 Blackwell Server Edition"
            ),
            "pilot": bool(args.pilot),
            "pilot_optimizer_steps": args.max_optimizer_steps if args.pilot else None,
        }
    )
    config["target_definition"] = (
        "latent_delta_pred_unsupervised;cotangent_operator_built_from_input_geometry"
    )
    config["target_semantics"] = (
        "both_output_branches_are_latent_and_receive_only_final_geometry_gradient"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

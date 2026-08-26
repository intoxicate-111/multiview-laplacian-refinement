#!/usr/bin/env python3
from __future__ import annotations

"""Build the controlled two-specialist continuous B/E hybrid configuration."""

import argparse
import json
from pathlib import Path


CHECKPOINT_STEPS = (0, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--arm-b-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.baseline.read_text(encoding="utf-8"))
    model = config["model"]
    model["hybrid_direct_head"] = {
        "enabled": False,
        "reason": "complete Arm-E specialist supplies the direct latent",
    }
    model["recovery_lambda_head"] = {"enabled": False}
    model["two_branch_pretrained_hybrid"] = {
        "enabled": True,
        "shared_backbone": False,
        "arm_b_checkpoint": str(args.arm_b_checkpoint),
        "arm_e_checkpoint": str(args.arm_e_checkpoint),
        "arm_b_role": "latent_uniform_laplacian",
        "arm_e_role": "latent_direct_displacement",
    }

    training = config["training"]
    training["learning_rate"] = 1e-4
    training["weight_decay"] = float(training.get("weight_decay", 0.0))
    training["gradient_clip_norm"] = float(training.get("gradient_clip_norm", 0.0))
    training["recovery_aware_geometry_loss"] = {"enabled": False}
    training["direct_vertex_runtime_diagnostics"] = False
    training["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "operator": "uniform_random_walk",
        "lambda": 3e-2,
        "maximum_iterations": 2048,
        "tolerance": 1e-8,
        "compute_dtype": "float64",
        "runtime_diagnostics": True,
        "validation_surface_samples": 3000,
        "loss": "mean_i_squared_l2_VH_minus_Vclean",
        "auxiliary_laplacian_loss": False,
        "auxiliary_direct_vertex_loss": False,
        "gt_inside_recovery": False,
    }
    training["vertex_sampling"] = {"mode": "full"}
    config["confidence"] = {
        "enabled": False,
        "recovery_weight": "none",
        "quantile_bins": 5,
    }

    multi = config["multi_object_training"]
    multi.update(
        {
            "epochs": 400,
            "max_optimizer_steps": 20000,
            "gradient_accumulation_meshes": 2,
            "validation_every_epochs": 4,
            "checkpoint_optimizer_steps": list(CHECKPOINT_STEPS),
            "checkpoint_every_epochs": 0,
            "checkpoint_epochs": [],
            "report_every_optimizer_steps": 50,
            "shuffle": True,
            "early_stopping": {
                "enabled": False,
                "patience_validations": 15,
                "min_delta": 1e-4,
            },
        }
    )
    config["experiment_metadata"] = {
        "experiment": "Sofa50_v2_continuous_pretrained_B_E_hybrid",
        "initialization": "exact_selected_complete_Arm_B_plus_complete_Arm_E",
        "shared_backbone": False,
        "optimizer": "fresh_Adam_over_both_pretrained_networks",
        "continuation_learning_rate": 1e-4,
        "distributed_world_size": 4,
        "gradient_accumulation_meshes_per_rank": 2,
        "effective_global_batch_meshes": 8,
        "checkpoint_selector": "validation_final_hybrid_unified_v2_chamfer_only",
        "only_training_loss": "mean_i_squared_l2_VH_minus_Vclean",
        "lambda": 3e-2,
        "operator": "uniform_random_walk_I_minus_DinvA",
        "test_or_ood_used_for_selection": False,
    }
    config["method"] = "sofa50_v2_continuous_pretrained_b_e_hybrid"
    config["recovery"] = {
        "mode": "differentiable_uniform_laplacian_direct_anchor",
        "lambda": 3e-2,
        "additional_input_anchor": False,
        "visibility_gate": False,
        "confidence_weighting": False,
        "robust_loss": None,
        "optimizer": None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

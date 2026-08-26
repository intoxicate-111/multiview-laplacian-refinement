#!/usr/bin/env python3
from __future__ import annotations

"""Build the validation-selected old-domain native-1920 B+E continuation config."""

import argparse
import json
from pathlib import Path
from typing import Any


CHECKPOINT_STEPS = (0, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)
LAMBDA_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--lambda-selection", required=True, type=Path)
    parser.add_argument("--arm-b-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selection = read_object(args.lambda_selection.resolve())
    if not selection.get("contract_audit"):
        raise RuntimeError("Frozen validation lambda selection did not pass its contract audit")
    if selection.get("selection_split") != "validation" or selection.get("test_accessed") is not False:
        raise RuntimeError("Continuation lambda must be selected on validation without test access")
    if tuple(float(value) for value in selection.get("lambda_grid", ())) != LAMBDA_GRID:
        raise RuntimeError("Frozen validation lambda grid differs from the frozen contract")
    selected_lambda = float(selection["selected_lambda"])
    if selected_lambda not in LAMBDA_GRID:
        raise RuntimeError("Selected lambda is outside the frozen validation grid")
    for checkpoint in (args.arm_b_checkpoint, args.arm_e_checkpoint):
        if not checkpoint.resolve().is_file():
            raise FileNotFoundError(checkpoint)

    config = read_object(args.baseline.resolve())
    model = config["model"]
    model["hybrid_direct_head"] = {
        "enabled": False,
        "reason": "complete independent Arm-E specialist supplies direct displacement",
    }
    model["recovery_lambda_head"] = {"enabled": False}
    model["two_branch_pretrained_hybrid"] = {
        "enabled": True,
        "shared_backbone": False,
        "arm_b_checkpoint": str(args.arm_b_checkpoint.resolve()),
        "arm_e_checkpoint": str(args.arm_e_checkpoint.resolve()),
        "arm_b_role": "raw_uniform_laplacian_current_graph",
        "arm_e_role": "direct_vertex_displacement",
    }

    training = config["training"]
    training["learning_rate"] = 1e-4
    training["weight_decay"] = 0.0
    training["gradient_clip_norm"] = 1.0
    training["recovery_aware_geometry_loss"] = {"enabled": False}
    training["direct_vertex_runtime_diagnostics"] = False
    training["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "operator": "uniform_random_walk",
        "lambda": selected_lambda,
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
            "epochs": 1000,
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
        "experiment": "Sofa50_old_domain_native1920_continuous_pretrained_B_E_hybrid",
        "initialization": "validation_selected_complete_Arm_B_plus_complete_Arm_E",
        "shared_backbone": False,
        "parameter_count_expected": 1652230,
        "optimizer": "fresh_Adam_over_both_complete_pretrained_networks",
        "continuation_learning_rate": 1e-4,
        "distributed_world_size": 4,
        "gradient_accumulation_meshes_per_rank": 2,
        "effective_global_batch_meshes": 8,
        "checkpoint_selector": "validation_final_hybrid_unified_surface_chamfer_only",
        "only_training_loss": "mean_i_squared_l2_VH_minus_Vclean",
        "lambda": selected_lambda,
        "lambda_selection": str(args.lambda_selection.resolve()),
        "lambda_selection_split": "validation",
        "operator": "uniform_random_walk_I_minus_DinvA",
        "input_resolution": 1920,
        "views": 28,
        "native_input_contract": "decode_exact_native_1920_png_no_resize_or_downsample",
        "test_used_for_selection": False,
        "sealed_test_policy": "test remains inaccessible until all validation selections are locked",
    }
    config["method"] = "sofa50_old_domain_native1920_continuous_pretrained_b_e_hybrid"
    config["recovery"] = {
        "mode": "differentiable_uniform_laplacian_direct_anchor",
        "lambda": selected_lambda,
        "additional_input_anchor": False,
        "visibility_gate": False,
        "confidence_weighting": False,
        "robust_loss": None,
        "optimizer": None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

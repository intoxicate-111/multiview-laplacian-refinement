#!/usr/bin/env python3
from __future__ import annotations

"""Build the controlled Sofa50 v2 single-loss direct--Laplacian hybrid config."""

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    if args.world_size != 8:
        raise ValueError("The controlled primary run requires exactly 8 Blackwell GPUs.")

    config = _read(args.baseline.resolve())
    config["method"] = "sofa50_v2_end_to_end_direct_laplacian_hybrid_single_loss"
    config["confidence"] = {
        "enabled": False,
        "quantile_bins": 5,
        "recovery_weight": "none",
    }
    model = config.setdefault("model", {})
    model["hybrid_direct_head"] = {
        "enabled": True,
        "source": "shared_post_graph_features",
        "architecture": "mirror_of_canonical_laplacian_output_mlp",
        "initialization": "normal_seeded_from_scratch_no_B_or_E_checkpoint",
    }
    model["recovery_lambda_head"] = {"enabled": False}
    training = config["training"]
    training["recovery_aware_geometry_loss"] = {"enabled": False}
    training["hybrid_single_geometry_loss"] = {
        "enabled": True,
        "lambda": 3e-2,
        "maximum_iterations": 2048,
        "tolerance": 1e-8,
        "compute_dtype": "float64",
        "runtime_diagnostics": True,
        "validation_surface_samples": 3000,
        "loss": "mean_i_squared_l2_VH_minus_Vclean",
        "auxiliary_laplacian_loss": False,
        "auxiliary_direct_vertex_loss": False,
        "confidence_loss": False,
        "gt_inside_recovery": False,
    }
    multi = config["multi_object_training"]
    multi["gradient_accumulation_meshes"] = 1
    multi["max_optimizer_steps"] = 20000
    multi["checkpoint_optimizer_steps"] = [5000, 10000, 15000, 20000]
    config["recovery"] = {
        "solver": "differentiable_float64_PCG",
        "objective": "||L V-delta_pred||_2^2+lambda||V-V_direct||_2^2",
        "lambda": 3e-2,
        "V_direct": "V_input+delta_v_direct_pred",
        "additional_V_input_anchor": False,
        "visibility_gate": False,
        "confidence_weighting": False,
        "robust_loss": None,
        "adam_vertex_optimization": False,
        "operator": "uniform_random_walk_current_graph",
        "tolerance_change": (
            "1e-4_to_1e-8_to_resolve_previously_observed_PCG_LSMR_solution_discrepancy;"
            "objective_and_lambda_unchanged"
        ),
    }
    config["experiment_metadata"] = {
        "experiment": "Sofa50_v2_end_to_end_direct_laplacian_hybrid_single_loss",
        "arm": "single_final_geometry_loss_hybrid_lambda3e-2",
        "initialization": "from_scratch",
        "checkpoint_selector": "validation_final_hybrid_unified_v2_chamfer_only",
        "test_or_ood_selection": False,
        "views": 28,
        "input_resolution": 960,
        "distributed_world_size": 8,
        "gradient_accumulation_meshes_per_rank": 1,
        "effective_global_batch_meshes": 8,
        "training_gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "fixed_lambda_source": "frozen_B_plus_E_validation_only_Chamfer_sweep",
        "no_B_E_head_initialization": True,
        "no_auxiliary_branch_supervision": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

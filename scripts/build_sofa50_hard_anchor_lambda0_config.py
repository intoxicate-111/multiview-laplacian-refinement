#!/usr/bin/env python3
from __future__ import annotations

"""Build the isolated Sofa50 v2 lambda=0 hard-anchor Arm-I config."""

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
    parser.add_argument("--arm-b-run-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = _read(args.arm_b_run_config.resolve())
    source = payload.get("experiment_config", payload)
    if not isinstance(source, dict):
        raise ValueError("Arm-B run config has no experiment_config object.")
    config = json.loads(json.dumps(source))
    source_loss = config["training"]["recovery_aware_geometry_loss"]
    if not bool(source_loss["enabled"]):
        raise ValueError("Source Arm B is not recovery-aware.")
    if float(source_loss["lambda"]) != 1e-2 or float(source_loss["beta"]) != 1e-2:
        raise ValueError("Source must be Arm B with lambda=1e-2 and beta=1e-2.")
    if int(config["multi_object_training"]["max_optimizer_steps"]) != 20000:
        raise ValueError("Source Arm B must use 20,000 optimizer steps.")

    config["method"] = "sofa50_recovery_aware_hard_anchor_lambda0"
    config["training"]["recovery_aware_geometry_loss"] = {
        "enabled": True,
        "solver": "hard_anchor_lambda0",
        "lambda": 0.0,
        "adaptive_lambda": False,
        "beta": 1e-2,
        "maximum_iterations": 2048,
        "tolerance": 1e-4,
        "compute_dtype": "float64",
        "differentiation": "implicit_custom_autograd_reduced_system",
        "clean_vertex_use": "training_loss_only_never_model_input",
        "runtime_diagnostics": True,
    }
    config["multi_object_training"]["max_optimizer_steps"] = 20000
    config["multi_object_training"]["gradient_accumulation_meshes"] = 1
    config["multi_object_training"]["checkpoint_optimizer_steps"] = [
        5000,
        10000,
        15000,
        20000,
    ]
    config["recovery"] = {
        "solver": "reduced_hard_anchor_undamped_normal_sparse_lu_float64",
        "operator_type": "uniform_random_walk_current_graph",
        "lambda": 0.0,
        "laplacian_equations": "all_vertices",
        "anchor": "lowest_global_vertex_index_per_connected_component",
        "anchor_coordinates": "exact_initial_coarse_mesh_position",
        "unknown_elimination": True,
        "centroid_constraint": False,
        "soft_positional_penalty": False,
        "hidden_tikhonov_damping": False,
        "visibility_gate": False,
        "confidence_weighting": False,
        "robust_loss": None,
        "optimizer": None,
        "training_implementation": "scipy_undamped_reduced_normal_sparse_lu_float64",
        "evaluation_reference": "scipy_lsmr_reduced_columns_float64",
    }
    metadata = config.setdefault("experiment_metadata", {})
    metadata.update(
        {
            "experiment": "Sofa50_v2_recovery_aware_lambda0_hard_anchor",
            "arm": "I_lap_plus_refine_lambda0_hard_anchor",
            "initialization": "from_scratch",
            "distributed_world_size": 8,
            "effective_global_batch_meshes": 8,
            "gradient_accumulation_meshes_per_rank": 1,
            "optimizer_steps": 20000,
            "hard_anchor_selection_uses_gt": False,
            "hard_anchor_selection": "minimum_index_per_component",
            "conceptual_change_only": "lambda0_reduced_hard_anchor_recovery",
            "solver_compute_dtype": "float64",
            "solver_residual_tolerance": 1e-4,
            "iterative_maximum_iterations_not_used_by_direct_solver": 2048,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Build fail-closed old-domain native-1920 Arm-B or Arm-E config."""

import argparse
import json
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value.get("experiment_config", value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("B", "E"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distributed-world-size", type=int, default=4)
    parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument(
        "--training-gpu-model",
        default="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    )
    args = parser.parse_args()
    if args.effective_global_batch % args.distributed_world_size:
        raise ValueError("effective global batch must be divisible by world size")
    accumulation = args.effective_global_batch // args.distributed_world_size

    config = json.loads(json.dumps(read_config(args.baseline.resolve())))
    config["dataset"] = {
        "name": "Sofa50SyntheticCurrent28ViewNative1920",
        "expected_split_counts": {"train": 200, "validation": 25, "test": 25},
        "objects": 50,
        "variants_per_object": 5,
        "object_level_split": True,
        "degradation_profile": "historical_synthetic_current_v1",
        "clean_reference_compatibility": "gt_vertices_same_topology",
    }
    config["target_mode"] = "raw_laplacian"
    config["target_semantics"] = "same_topology_clean_reference_native_target"
    config["target_definition"] = "delta_target_raw=L_U(current_graph)@V_clean"
    config["target_scaling"] = {
        "method": "square_of_mean_incident_edge_length",
        "source": "input_prediction_mesh_auxiliary_compatibility_fields_only",
        "applied_to_training_target": False,
        "epsilon": 1e-12,
        "clip_max_norm": None,
    }
    config["local_query_jitter"]["frozen"] = [
        "clean_reference", "target", "graph", "connectivity", "operator"
    ]
    config["renderer_visibility"]["source"] = (
        "precomputed_current_graph_depth_tested_face_id_buffer"
    )
    # These are execution-only memory controls already established by the
    # historical native-1920 run.  They do not add parameters or alter inputs.
    config["image_encoder"]["view_chunk_size"] = 4
    config["image_encoder"]["gradient_checkpointing"] = True
    config["confidence"] = {
        "enabled": False,
        "quantile_bins": 5,
        "recovery_weight": "none",
    }
    multi = config["multi_object_training"]
    multi["epochs"] = 1000
    multi["max_optimizer_steps"] = 20000
    multi["gradient_accumulation_meshes"] = accumulation
    multi["checkpoint_optimizer_steps"] = [5000, 10000, 15000, 20000]
    multi["checkpoint_every_epochs"] = 0
    multi["checkpoint_epochs"] = []
    multi["validation_every_epochs"] = 8

    common_metadata = {
        "experiment": "Sofa50_old_domain_native1920_independent_specialists",
        "initialization": "from_scratch",
        "capacity": "C2",
        "feature_resolution": "F2",
        "views": 28,
        "input_resolution": 1920,
        "render_resolution": 1920,
        "native_input_contract": "decode_exact_native_1920_png_no_resize_or_downsample",
        "distributed_world_size": args.distributed_world_size,
        "gradient_accumulation_meshes_per_rank": accumulation,
        "effective_global_batch_meshes": args.effective_global_batch,
        "training_gpu_model": args.training_gpu_model,
        "execution_only_native1920_memory_controls": (
            "view_chunk_size=4;gradient_checkpointing=true"
        ),
        "sealed_test_policy": (
            "test unavailable to training_validation_checkpoint_selection_and_lambda_selection"
        ),
    }

    training = config["training"]
    if args.arm == "B":
        config["method"] = "sofa50_old_domain_native1920_recovery_aware_arm_b"
        config["prediction_semantics"] = "current_graph_laplacian"
        training["loss"] = "huber"
        training["huber_delta"] = 0.01
        training["prediction_loss_space"] = "output_representation"
        training["recovery_aware_geometry_loss"] = {
            "enabled": True,
            "lambda": 0.01,
            "beta": 0.01,
            "maximum_iterations": 256,
            "tolerance": 1e-4,
            "compute_dtype": "float32",
            "differentiation": "implicit_custom_autograd",
            "clean_vertex_use": "training_loss_only_never_model_input",
            "runtime_diagnostics": True,
        }
        config["recovery"] = {
            "solver": "regularized_sparse_least_squares",
            "operator_type": "uniform_random_walk_current_graph",
            "lambda": 0.01,
            "laplacian_equations": "all_vertices",
            "anchor": "lambda_times_input_vertex_l2",
            "visibility_gate": False,
            "confidence_weighting": False,
            "robust_loss": None,
            "optimizer": None,
            "standalone_implementation": "scipy_lsmr_augmented_system",
        }
        config["experiment_metadata"] = common_metadata | {
            "arm": "B",
            "beta": 0.01,
            "lambda": 0.01,
            "loss": "huber_raw_laplacian_plus_beta_recovered_vertex_mse",
            "checkpoint_selection": "established_arm_b_validation_objective",
            "no_h2_target_normalization": True,
            "no_proxy_or_target_transfer": True,
        }
    else:
        config["method"] = "sofa50_old_domain_native1920_direct_vertex_arm_e"
        config["prediction_semantics"] = "direct_vertex_displacement"
        config["target_semantics"] = "same_index_clean_minus_input_vertex_residual"
        config["target_definition"] = "delta_v_gt=V_clean-V_input"
        training["loss"] = "mse"
        training["prediction_loss_space"] = "output_representation"
        training["target_magnitude_weight_lambda"] = 0.0
        training["direct_vertex_runtime_diagnostics"] = True
        training["recovery_aware_geometry_loss"] = {
            "enabled": False,
            "lambda": 0.01,
            "beta": 0.0,
            "runtime_diagnostics": False,
        }
        config["recovery"] = {
            "mode": "direct_vertex_addition",
            "equation": "V_refined=V_input+delta_v_pred",
            "laplacian_operator_used": False,
            "sparse_integration": False,
            "pcg_or_lsmr": False,
            "lambda": None,
            "visibility_gate": False,
            "confidence_weighting": False,
            "robust_loss": None,
            "optimizer": None,
            "postprocessing": None,
        }
        config["experiment_metadata"] = common_metadata | {
            "arm": "E_direct_vertex_residual",
            "forward_equation": (
                "delta_v_pred=f_theta(images,cameras,V_input,mesh_features);"
                "V_refined=V_input+delta_v_pred"
            ),
            "loss": "mean_vertex_squared_l2",
            "checkpoint_selection": "established_arm_e_validation_vertex_mse",
            "no_laplacian_target_loss_or_recovery": True,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

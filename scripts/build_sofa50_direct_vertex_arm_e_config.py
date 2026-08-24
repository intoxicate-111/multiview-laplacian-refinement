#!/usr/bin/env python3
from __future__ import annotations

"""Build the controlled Sofa50-v2 Arm-E direct vertex residual config."""

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
    parser.add_argument("--distributed-world-size", type=int, default=8)
    parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument(
        "--training-gpu-model",
        default="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    )
    args = parser.parse_args()
    if args.effective_global_batch % args.distributed_world_size:
        raise ValueError("effective global batch must be divisible by world size")

    baseline = _read(args.baseline.resolve())
    config = json.loads(json.dumps(baseline))
    config["method"] = "sofa50_multitopology_direct_vertex_residual_baseline"
    config["prediction_semantics"] = "direct_vertex_displacement"
    config["target_semantics"] = "same_index_clean_minus_input_vertex_residual"
    config["target_definition"] = "delta_v_gt=V_clean-V_input"
    # These fields remain in the prepared dataset only as inert compatibility
    # data. They are not used in Arm E's target, loss, or inference path.
    config["target_scaling"]["applied_to_training_target"] = False
    config["target_scaling"]["clip_max_norm"] = None
    config["confidence"] = {
        "enabled": False,
        "quantile_bins": 5,
        "recovery_weight": "none",
    }
    training = config["training"]
    training["loss"] = "mse"
    training["prediction_loss_space"] = "output_representation"
    training["target_magnitude_weight_lambda"] = 0.0
    training["vertex_sampling"] = {"mode": "full"}
    training["direct_vertex_runtime_diagnostics"] = True
    training["recovery_aware_geometry_loss"] = {
        "enabled": False,
        "lambda": 1e-2,
        "beta": 0.0,
        "runtime_diagnostics": False,
    }
    multi = config["multi_object_training"]
    multi["max_optimizer_steps"] = 20000
    multi["gradient_accumulation_meshes"] = (
        args.effective_global_batch // args.distributed_world_size
    )
    multi["checkpoint_optimizer_steps"] = [5000, 10000, 15000, 20000]
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
    config["experiment_metadata"] = {
        "experiment": "Sofa50MultiTopologyRawLap500_v2_direct_vertex_arm_e",
        "arm": "E_direct_vertex_residual",
        "initialization": "from_scratch",
        "capacity": "C2",
        "feature_resolution": "F2",
        "views": 28,
        "input_resolution": 960,
        "hf_feature": True,
        "distributed_world_size": args.distributed_world_size,
        "gradient_accumulation_meshes_per_rank": multi[
            "gradient_accumulation_meshes"
        ],
        "effective_global_batch_meshes": args.effective_global_batch,
        "training_gpu_model": args.training_gpu_model,
        "forward_equation": "delta_v_pred=f_theta(images,cameras,V_input,mesh_features);V_refined=V_input+delta_v_pred",
        "loss_equation": "mean_i(||delta_v_pred_i-(V_clean_i-V_input_i)||_2^2)",
        "no_laplacian_target_or_loss": True,
        "no_analytic_recovery": True,
        "no_visibility_confidence_huber_or_adam_recovery": True,
        "gradient_clip_note": "inherited unchanged from A-D training contract; no displacement clamp",
        "patch_size_8_contract": "not_applicable_baseline_has_no_patch_operator",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

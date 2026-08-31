#!/usr/bin/env python3
from __future__ import annotations

"""Build the Future2000 Arm-E config using the established Sofa50 Arm-E contract."""

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future2000-baseline", required=True, type=Path)
    parser.add_argument("--sofa50-arm-e-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distributed-world-size", type=int, default=4)
    parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument("--max-optimizer-steps", type=int, default=200_000)
    args = parser.parse_args()
    if args.effective_global_batch % args.distributed_world_size:
        raise ValueError("effective global batch must be divisible by world size")
    if args.max_optimizer_steps < 1:
        raise ValueError("max optimizer steps must be positive")

    baseline = read_json(args.future2000_baseline.resolve())
    reference = read_json(args.sofa50_arm_e_reference.resolve())
    config = json.loads(json.dumps(baseline))

    # Copy the complete specialist semantics from the established Sofa50 Arm-E.
    for key in (
        "confidence",
        "prediction_semantics",
        "recovery",
        "target_definition",
        "target_mode",
        "target_scaling",
        "target_semantics",
    ):
        config[key] = json.loads(json.dumps(reference[key]))
    config["method"] = "future2000_current28view_direct_vertex_arm_e_200k"

    # The backbone and 28-view HF observation path are already identical; copy
    # the Arm-E objective/runtime fields explicitly and retain Future2000 data.
    config["training"] = json.loads(json.dumps(reference["training"]))
    multi = config["multi_object_training"]
    multi["epochs"] = 2000
    multi["max_optimizer_steps"] = args.max_optimizer_steps
    multi["gradient_accumulation_meshes"] = (
        args.effective_global_batch // args.distributed_world_size
    )
    multi["checkpoint_optimizer_steps"] = list(
        range(10_000, args.max_optimizer_steps + 1, 10_000)
    )
    if multi["checkpoint_optimizer_steps"][-1] != args.max_optimizer_steps:
        multi["checkpoint_optimizer_steps"].append(args.max_optimizer_steps)
    multi["checkpoint_epochs"] = []
    multi["checkpoint_every_epochs"] = 0
    multi["report_every_optimizer_steps"] = 2000
    multi["validation_every_epochs"] = 5
    multi["early_stopping"] = {
        "enabled": False,
        "min_delta": 0.0001,
        "patience_validations": 15,
    }

    config["experiment_metadata"] = {
        "experiment": "Future2000_current28view_Arm_E_direct_vertex",
        "arm": "E_direct_vertex_residual",
        "arm_e_architecture_reference": (
            "completed_Sofa50_Arm_E_direct_vertex_specialist"
        ),
        "architecture_and_objective_fields_copied_exactly": [
            "input_mode",
            "image_encoder",
            "model",
            "confidence",
            "recovery",
            "prediction_semantics",
            "target_mode",
            "target_scaling",
            "query_training",
            "local_query_jitter",
            "renderer_visibility",
            "training",
        ],
        "capacity": "C2",
        "feature_resolution": "F2",
        "views": 28,
        "input_resolution": 960,
        "query_graph": "gt_adaptive_sub2_represented_vertex_area",
        "distributed_world_size": args.distributed_world_size,
        "per_gpu_batch_meshes": 1,
        "gradient_accumulation_meshes_per_rank": multi[
            "gradient_accumulation_meshes"
        ],
        "effective_global_batch_meshes": args.effective_global_batch,
        "initialization": "from_scratch",
        "loss_contract": "mean_same_index_vertex_residual_squared_l2",
        "forward_equation": (
            "delta_v_pred=f_theta(images,cameras,V_input,mesh_features);"
            "V_E=V_input+delta_v_pred"
        ),
        "training_gpu_model": (
            "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        ),
        "max_optimizer_steps": args.max_optimizer_steps,
        "checkpoint_selection": "validation_direct_vertex_mse_only",
        "future_fusion_role": "frozen_vertex_anchor_V_E_for_B_plus_E",
        "ground_truth_never_model_input": True,
        "sealed_test_policy": (
            "test_unavailable_to_training_and_checkpoint_selection"
        ),
        "no_laplacian_target_loss_or_analytic_recovery": True,
    }

    expected = config["dataset"]["expected_split_counts"]
    if expected != {"train": 8000, "validation": 1000, "test": 1000}:
        raise ValueError(f"Unexpected Future2000 split contract: {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

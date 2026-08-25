#!/usr/bin/env python3
from __future__ import annotations

"""Build one audited Sofa50 recovery-aware ablation config from the v2 baseline."""

import argparse
import json
from pathlib import Path
from typing import Any


FROZEN_TOP_LEVEL = (
    "seed",
    "dataset",
    "input_mode",
    "target_mode",
    "target_semantics",
    "target_definition",
    "target_scaling",
    "query_training",
    "local_query_jitter",
    "renderer_visibility",
    "image_encoder",
    "model",
    "data_loading",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--arm", required=True, choices=("A", "B", "C", "D", "E", "F")
    )
    parser.add_argument("--lambda-value", required=True, type=float)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--max-optimizer-steps", type=int, default=20000)
    parser.add_argument("--run-kind", choices=("final", "beta_pilot"), default="final")
    parser.add_argument("--distributed-world-size", type=int, default=2)
    parser.add_argument("--effective-global-batch", type=int, default=8)
    parser.add_argument("--training-gpu-model", default="NVIDIA L40")
    parser.add_argument(
        "--initialization", choices=("from_scratch", "resume"), default="from_scratch"
    )
    args = parser.parse_args()
    if args.lambda_value <= 0:
        raise ValueError("lambda must be positive")
    if args.arm == "A" and args.beta != 0:
        raise ValueError("Arm A beta must be zero")
    if args.arm in {"B", "C", "D", "E", "F"} and args.beta <= 0:
        raise ValueError(f"Arm {args.arm} beta must be positive")
    if args.run_kind == "final" and args.max_optimizer_steps != 20000:
        raise ValueError("Final runs require exactly 20,000 optimizer steps")
    if args.distributed_world_size <= 0 or args.effective_global_batch <= 0:
        raise ValueError("world size and effective global batch must be positive")
    if args.effective_global_batch % args.distributed_world_size:
        raise ValueError("effective global batch must be divisible by world size")
    accumulation_per_rank = args.effective_global_batch // args.distributed_world_size

    baseline = _read(args.baseline.resolve())
    config = json.loads(json.dumps(baseline))
    config["method"] = "sofa50_multitopology_rawlap_regularized_sparse_ablation"
    # This explicit change is required by the requested ablation. The predictor
    # has no confidence head in either arm; A and B therefore have identical
    # architecture and parameter count.
    config["confidence"] = {
        "enabled": False,
        "quantile_bins": 5,
        "recovery_weight": "none",
    }
    config["training"]["recovery_aware_geometry_loss"] = {
        "enabled": args.arm in {"B", "C", "D", "E", "F"},
        "lambda": args.lambda_value,
        "beta": args.beta,
        "maximum_iterations": 2048 if args.arm in {"C", "D", "E", "F"} else 256,
        "tolerance": 1e-4,
        "compute_dtype": (
            "float64" if args.arm in {"C", "D", "E", "F"} else "float32"
        ),
        "differentiation": "implicit_custom_autograd",
        "clean_vertex_use": "training_loss_only_never_model_input",
        # C/D require observational runtime logging. This flag only exposes
        # audit values from the same PCG solve and retains gradients for norm
        # measurement; it does not alter the solve, loss, or optimizer update.
        "runtime_diagnostics": args.arm in {"C", "D", "E", "F"},
    }
    config["multi_object_training"]["max_optimizer_steps"] = args.max_optimizer_steps
    config["multi_object_training"]["gradient_accumulation_meshes"] = (
        accumulation_per_rank
    )
    config["multi_object_training"]["checkpoint_optimizer_steps"] = (
        [5000, 10000, 15000, 20000]
        if args.max_optimizer_steps == 20000
        else [args.max_optimizer_steps]
    )
    config["recovery"] = {
        "solver": "regularized_sparse_least_squares",
        "operator_type": "uniform_random_walk_current_graph",
        "lambda": args.lambda_value,
        "laplacian_equations": "all_vertices",
        "anchor": "lambda_times_input_vertex_l2",
        "visibility_gate": False,
        "confidence_weighting": False,
        "robust_loss": None,
        "optimizer": None,
        "standalone_implementation": "scipy_lsmr_augmented_system",
    }
    config["experiment_metadata"] = {
        "experiment": (
            "Sofa50MultiTopologyRawLap500_v2_recovery_aware_lambda_extension"
            if args.arm in {"C", "D", "E", "F"}
            else "Sofa50MultiTopologyRawLap500_v2_recovery_aware_two_arm"
        ),
        "arm": args.arm,
        "run_kind": args.run_kind,
        "beta": args.beta,
        "lambda": args.lambda_value,
        "initialization": args.initialization,
        "capacity": "C2",
        "feature_resolution": "F2",
        "views": 28,
        "input_resolution": 960,
        "hf_feature": True,
        "distributed_world_size": args.distributed_world_size,
        "gradient_accumulation_meshes_per_rank": accumulation_per_rank,
        "effective_global_batch_meshes": args.effective_global_batch,
        "training_gpu_model": args.training_gpu_model,
        "training_huber_is_laplacian_regression_only": True,
        "no_huber_in_recovery": True,
        "no_visibility_confidence_or_adam_recovery": True,
        "pcg_numerical_execution_difference": (
            "float64_with_maximum_iterations_2048_after_float32_smoke_stagnation; tolerance_unchanged_at_1e-4"
            if args.arm in {"C", "D", "E", "F"}
            else "none"
        ),
        # The completed strong_smooth_v2 pipeline has no patch sampler or
        # patch_size field. Record this instead of inventing an ignored value.
        "patch_size_8_contract": "not_applicable_baseline_has_no_patch_operator",
        "frozen_top_level_fields": list(FROZEN_TOP_LEVEL),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

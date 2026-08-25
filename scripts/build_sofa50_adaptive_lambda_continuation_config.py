#!/usr/bin/env python3
from __future__ import annotations

"""Build matched G/H continuations from the exact completed Arm-B run config."""

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
    parser.add_argument("--arm", required=True, choices=("G", "H"))
    parser.add_argument("--continuation-steps", type=int, default=5000)
    args = parser.parse_args()
    if args.continuation_steps != 5000:
        raise ValueError("This predeclared continuation control uses exactly 5,000 steps.")

    payload = _read(args.arm_b_run_config.resolve())
    config = payload.get("experiment_config", payload)
    if not isinstance(config, dict):
        raise ValueError("Arm-B run config has no experiment_config object.")
    # JSON round trip deliberately prevents mutation of the source object.
    config = json.loads(json.dumps(config))
    original_steps = int(config["multi_object_training"]["max_optimizer_steps"])
    if original_steps != 20000:
        raise ValueError(f"Expected completed Arm B at 20,000 steps, got {original_steps}.")
    if float(config["training"]["recovery_aware_geometry_loss"]["lambda"]) != 1e-2:
        raise ValueError("Continuation source is not fixed-lambda Arm B.")
    if float(config["training"]["recovery_aware_geometry_loss"]["beta"]) != 1e-2:
        raise ValueError("Continuation source does not use beta=1e-2.")

    adaptive = args.arm == "H"
    config["method"] = "sofa50_recovery_aware_adaptive_lambda_continuation"
    config.setdefault("model", {})["recovery_lambda_head"] = {
        "enabled": adaptive,
        "pooling": "mean_shared_graph_features",
        "hidden_dim": 16,
        "output": "one_scalar_per_mesh",
        "parameterization": "bounded_log10_sigmoid",
        "lambda_min": 1e-3,
        "lambda_max": 1e-1,
        "lambda_initial": 1e-2,
        "gt_inputs": False,
    }
    recovery_loss = config["training"]["recovery_aware_geometry_loss"]
    recovery_loss.update(
        {
            "lambda": 1e-2,
            "adaptive_lambda": adaptive,
            "beta": 1e-2,
            "compute_dtype": "float64",
            "maximum_iterations": 2048,
            "tolerance": 1e-4,
            "runtime_diagnostics": True,
        }
    )
    config["training"]["allow_resume_with_new_recovery_lambda_head"] = adaptive
    final_steps = original_steps + args.continuation_steps
    config["multi_object_training"]["max_optimizer_steps"] = final_steps
    config["multi_object_training"]["checkpoint_optimizer_steps"] = list(
        range(original_steps + 1000, final_steps + 1, 1000)
    )
    config["recovery"].update(
        {
            "lambda": "predicted_scalar" if adaptive else 1e-2,
            "lambda_min": 1e-3 if adaptive else None,
            "lambda_max": 1e-1 if adaptive else None,
            "visibility_gate": False,
            "confidence_weighting": False,
            "robust_loss": None,
            "optimizer": None,
        }
    )
    metadata = config.setdefault("experiment_metadata", {})
    metadata.update(
        {
            "experiment": "Sofa50_v2_adaptive_lambda_matched_continuation",
            "arm": "G_fixed_continue" if args.arm == "G" else "H_predicted_lambda",
            "source_arm": "B_lap_plus_refine",
            "source_optimizer_steps": original_steps,
            "additional_optimizer_steps": args.continuation_steps,
            "final_optimizer_steps": final_steps,
            "resume_model_optimizer_scheduler_scaler": True,
            "effective_global_batch_meshes": 8,
            "distributed_world_size": 8,
            "gradient_accumulation_meshes_per_rank": 1,
            "pcg_compute_dtype": "float64",
            "pcg_tolerance": 1e-4,
            "pcg_maximum_iterations": 2048,
            "adaptive_training_authorized_by": (
                "validation_only_oracle_gate" if adaptive else "matched_control"
            ),
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

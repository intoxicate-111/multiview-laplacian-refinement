#!/usr/bin/env python3
from __future__ import annotations

"""Read-only implementation and data-lineage audit for Sofa50 direct Arm E."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mlr.learned_laplacian.controlled_displacement import displacement_target
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model, _prepare_object_static


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--arm-e-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline_payload = _read(args.baseline_config.resolve())
    baseline = baseline_payload.get("experiment_config", baseline_payload)
    arm_e = _read(args.arm_e_config.resolve())
    split_counts: dict[str, int] = {}
    maximum_target_clean_error = 0.0
    model_input_leaks: list[dict[str, str]] = []
    connectivity_mismatches: list[str] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        split_counts[split] = len(dataset)
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            target = displacement_target(static)
            clean = static.get("clean_reference_vertices")
            clean_faces = static.get("clean_reference_faces")
            if not isinstance(clean, torch.Tensor):
                raise RuntimeError(f"{sample_id}: missing clean_reference_vertices")
            expected = clean - static["vertices"]
            maximum_target_clean_error = max(
                maximum_target_clean_error,
                float(torch.max(torch.abs(target - expected)).item()),
            )
            if not isinstance(clean_faces, torch.Tensor) or not torch.equal(
                clean_faces, static["faces"]
            ):
                connectivity_mismatches.append(sample_id)
            prepared = _prepare_object_static(static, arm_e)
            leaking = sorted(
                key
                for key in prepared.sample
                if "clean" in key.lower()
                or key.startswith("gt_")
                or key == "target_positions"
            )
            if leaking:
                model_input_leaks.extend(
                    {"sample_id": sample_id, "key": key} for key in leaking
                )

    baseline_parameters = sum(
        parameter.numel() for parameter in _build_model(baseline, None, False).parameters()
    )
    arm_e_parameters = sum(
        parameter.numel() for parameter in _build_model(arm_e, None, False).parameters()
    )
    frozen_equal = {
        key: baseline[key] == arm_e[key]
        for key in (
            "seed",
            "dataset",
            "input_mode",
            "query_training",
            "local_query_jitter",
            "renderer_visibility",
            "image_encoder",
            "model",
            "data_loading",
        )
    }
    optimizer_schedule_equal = all(
        baseline["training"][key] == arm_e["training"][key]
        for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "amp", "lr_scheduler")
    )
    cadence_equal = all(
        baseline["multi_object_training"][key] == arm_e["multi_object_training"][key]
        for key in (
            "max_optimizer_steps",
            "report_every_optimizer_steps",
            "checkpoint_optimizer_steps",
            "validation_every_epochs",
        )
    )
    passed = bool(
        split_counts == {"train": 400, "validation": 50, "test": 50}
        and maximum_target_clean_error == 0.0
        and not model_input_leaks
        and not connectivity_mismatches
        and baseline_parameters == arm_e_parameters == 826115
        and all(frozen_equal.values())
        and optimizer_schedule_equal
        and cadence_equal
        and arm_e["prediction_semantics"] == "direct_vertex_displacement"
        and arm_e["training"]["loss"] == "mse"
        and not arm_e["training"]["recovery_aware_geometry_loss"]["enabled"]
        and not arm_e["confidence"]["enabled"]
        and arm_e["recovery"]["mode"] == "direct_vertex_addition"
    )
    payload = {
        "implementation_audit": passed,
        "split_counts": split_counts,
        "maximum_abs_error_between_training_target_and_clean_minus_input": maximum_target_clean_error,
        "same_topology_connectivity_mismatches": connectivity_mismatches,
        "gt_or_clean_model_input_leaks": model_input_leaks,
        "baseline_parameter_count": baseline_parameters,
        "arm_e_parameter_count": arm_e_parameters,
        "frozen_contract_equal": frozen_equal,
        "optimizer_and_lr_schedule_equal": optimizer_schedule_equal,
        "validation_checkpoint_cadence_equal": cadence_equal,
        "forward_equation": "delta_v_pred=f_theta(I,C,V_input,F);V_refined=V_input+delta_v_pred",
        "loss_equation": "mean_i(||delta_v_pred_i-(V_clean_i-V_input_i)||_2^2)",
        "laplacian_or_analytic_recovery_in_primary_path": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

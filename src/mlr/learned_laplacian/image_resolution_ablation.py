from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .geometry_aware_sampling import _magnitude_masks
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import (
    _amp_settings,
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from .trainer import _seed_everything, load_checkpoint


ARM_LAYOUT = {
    "F0_240": "image_resolution_f0",
    "F1_480": "image_resolution_f1",
    "F2_960": "image_resolution_f2",
}

GROUPS = (
    "all",
    "smooth_bottom_90",
    "high_top_10",
    "high_top_1",
)


def analyze_image_resolution_ablation(
    output_root: str | Path,
    manifest_path: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()

    arm_dirs = {
        name: output_root / "arms" / arm
        for name, arm in ARM_LAYOUT.items()
    }

    resolved_device = torch.device(device)

    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    condition_metrics: dict[
        str,
        dict[str, dict[str, float | int]],
    ] = {}

    feature_shapes: dict[str, list[int]] = {}
    fixed_prediction_max_abs_difference: dict[str, float] = {}

    for arm, arm_dir in arm_dirs.items():
        config = _read_json(arm_dir / "config.json")

        dataset = PreparedMeshDataset.from_manifest(
            manifest_path,
            "validation",
        )

        result = _evaluate_original_and_zero(
            arm_dir,
            config,
            dataset,
            resolved_device,
        )

        condition_metrics[arm] = result["metrics"]
        feature_shapes[arm] = result["feature_map_shape"]

        fixed_prediction_max_abs_difference[arm] = result[
            "fixed_prediction_max_abs_difference"
        ]

    original = {
        arm: values["original_rgb"]
        for arm, values in condition_metrics.items()
    }

    zero = {
        arm: values["zero_rgb"]
        for arm, values in condition_metrics.items()
    }

    comparison = {
        "overall_endpoint": _arm_values(
            original,
            "all",
            "endpoint",
        ),
        "top10_endpoint": _arm_values(
            original,
            "high_top_10",
            "endpoint",
        ),
        "top1_endpoint": _arm_values(
            original,
            "high_top_1",
            "endpoint",
        ),
        "smooth90_endpoint": _arm_values(
            original,
            "smooth_bottom_90",
            "endpoint",
        ),
        "global_cosine": _arm_values(
            original,
            "all",
            "global_cosine",
        ),
        "prediction_to_gt_norm": _arm_values(
            original,
            "all",
            "prediction_to_gt_norm",
        ),
    }

    relative_changes = {
        "F1_vs_F0": _relative_changes(
            comparison,
            source="F0",
            target="F1",
        ),
        "F2_vs_F0": _relative_changes(
            comparison,
            source="F0",
            target="F2",
        ),
        "F2_vs_F1": _relative_changes(
            comparison,
            source="F1",
            target="F2",
        ),
    }

    rgb_gaps = {
        arm: {
            "overall_endpoint_zero_minus_original": (
                float(zero[arm]["all"]["endpoint"])
                - float(original[arm]["all"]["endpoint"])
            ),
            "top10_endpoint_zero_minus_original": (
                float(zero[arm]["high_top_10"]["endpoint"])
                - float(original[arm]["high_top_10"]["endpoint"])
            ),
            "top1_endpoint_zero_minus_original": (
                float(zero[arm]["high_top_1"]["endpoint"])
                - float(original[arm]["high_top_1"]["endpoint"])
            ),
            "smooth90_endpoint_zero_minus_original": (
                float(zero[arm]["smooth_bottom_90"]["endpoint"])
                - float(original[arm]["smooth_bottom_90"]["endpoint"])
            ),
            "global_cosine_original_minus_zero": (
                float(original[arm]["all"]["global_cosine"])
                - float(zero[arm]["all"]["global_cosine"])
            ),
            "prediction_to_gt_norm_original_minus_zero": (
                float(original[arm]["all"]["prediction_to_gt_norm"])
                - float(zero[arm]["all"]["prediction_to_gt_norm"])
            ),
        }
        for arm in ARM_LAYOUT
    }

    configs = {
        arm: _read_json(path / "config.json")
        for arm, path in arm_dirs.items()
    }

    summaries = {
        arm: _read_json(path / "screening_summary.json")
        for arm, path in arm_dirs.items()
    }

    summary = {
        "experiment": (
            "Sofa50 image feature resolution ablation, "
            "240 vs 480 vs 960"
        ),
        "comparison_original_rgb": comparison,
        "relative_changes": relative_changes,
        "original_vs_zero_rgb_gap": rgb_gaps,
        "condition_metrics": condition_metrics,
        "feature_map_shapes_vchw": feature_shapes,
        "contract_audit": {
            "same_contract_except_image_strides": (
                _same_contract_except_strides(configs)
            ),
            "same_initial_parameter_tensors": (
                _initialization_equal(configs)
            ),
            "same_seed": (
                len(
                    {
                        int(config["seed"])
                        for config in configs.values()
                    }
                )
                == 1
            ),
            "uniform_full_vertex_training": all(
                config["training"]["vertex_sampling"]["mode"]
                == "full"
                for config in configs.values()
            ),
            "fresh_start": all(
                config["screening"]["resume_checkpoint"] is None
                for config in configs.values()
            ),
            "optimizer_steps": {
                arm: int(value["optimizer_steps"])
                for arm, value in summaries.items()
            },
            "fixed_original_prediction_max_abs_difference": (
                fixed_prediction_max_abs_difference
            ),
        },
        "recovery": {
            "performed": False,
        },
    }

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    (analysis_dir / "REPORT.md").write_text(
        _report(summary),
        encoding="utf-8",
    )

    return summary


@torch.no_grad()
def _evaluate_original_and_zero(
    arm_dir: Path,
    config: Mapping[str, Any],
    dataset: PreparedMeshDataset,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(int(config.get("seed", 7)))

    model = _build_model(
        config,
        None,
        False,
    ).to(device)

    load_checkpoint(
        arm_dir / "best.pt",
        model,
        map_location=device,
    )

    model.eval()

    amp_enabled, amp_dtype = _amp_settings(
        config,
        device,
    )

    arrays: dict[
        str,
        dict[str, list[np.ndarray]],
    ] = {
        condition: {
            "prediction": [],
            "target": [],
            **{
                group: []
                for group in GROUPS
            },
        }
        for condition in (
            "original_rgb",
            "zero_rgb",
        )
    }

    prediction_dir = (
        arm_dir
        / "rgb_resolution_ablation_predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_map_shape: list[int] | None = None
    max_difference = 0.0

    exact_config = copy.deepcopy(dict(config))
    exact_config["query_training"][
        "apply_to_validation"
    ] = False

    for index in range(len(dataset)):
        prepared = _prepare_item_for_use(
            _prepare_object_static(
                dataset.load_static(index),
                exact_config,
            ),
            exact_config,
            device,
            cache_on_device=False,
            decode_images=True,
        )

        base = dict(prepared.sample)

        base["query_positions"] = base["vertices"]

        base["query_is_exact"] = torch.ones(
            len(base["vertices"]),
            dtype=torch.bool,
            device=device,
        )

        if feature_map_shape is None:
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                feature_map_shape = list(
                    model.image_encoder(
                        base["images"][:1]
                    ).shape
                )

        predictions: dict[str, np.ndarray] = {}

        for condition in (
            "original_rgb",
            "zero_rgb",
        ):
            sample = dict(base)

            if condition == "zero_rgb":
                sample["images"] = torch.zeros_like(
                    base["images"]
                )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                prediction = (
                    model(sample)
                    .predicted_laplacian
                    .float()
                )

            predictions[condition] = (
                prediction.cpu().numpy()
            )

        target = (
            prepared.training_target
            .float()
            .cpu()
            .numpy()
        )

        masks = _magnitude_masks(
            np.linalg.norm(
                target,
                axis=1,
            )
        )

        sample_id = str(
            base["sample_id"]
        )

        fixed = np.load(
            arm_dir
            / "fixed_query_predictions"
            / f"{sample_id}__exact.npz"
        )["prediction"]

        max_difference = max(
            max_difference,
            float(
                np.max(
                    np.abs(
                        fixed
                        - predictions["original_rgb"]
                    )
                )
            ),
        )

        np.savez_compressed(
            prediction_dir / f"{sample_id}.npz",
            target=target,
            original_rgb=predictions[
                "original_rgb"
            ],
            zero_rgb=predictions[
                "zero_rgb"
            ],
        )

        for condition, prediction in predictions.items():
            arrays[condition][
                "prediction"
            ].append(prediction)

            arrays[condition][
                "target"
            ].append(target)

            for group in GROUPS:
                arrays[condition][
                    group
                ].append(
                    masks[group]
                )

        del prepared, base

        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = {
        condition: _condition_metrics(values)
        for condition, values in arrays.items()
    }

    if feature_map_shape is None:
        raise RuntimeError(
            "Validation dataset was empty."
        )

    return {
        "metrics": metrics,
        "feature_map_shape": feature_map_shape,
        "fixed_prediction_max_abs_difference": (
            max_difference
        ),
    }


def _condition_metrics(
    values: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    prediction = np.concatenate(
        values["prediction"]
    )

    target = np.concatenate(
        values["target"]
    )

    result: dict[str, Any] = {}

    for group in GROUPS:
        mask = np.concatenate(
            values[group]
        ).astype(bool)

        pred = prediction[
            mask
        ].astype(np.float64)

        gt = target[
            mask
        ].astype(np.float64)

        endpoint = np.linalg.norm(
            pred - gt,
            axis=1,
        )

        result[group] = {
            "vertex_count": int(
                mask.sum()
            ),
            "endpoint": float(
                endpoint.mean()
            ),
            "global_cosine": float(
                np.dot(
                    pred.reshape(-1),
                    gt.reshape(-1),
                )
                / max(
                    np.linalg.norm(pred)
                    * np.linalg.norm(gt),
                    1e-12,
                )
            ),
            "prediction_to_gt_norm": float(
                np.linalg.norm(pred)
                / max(
                    np.linalg.norm(gt),
                    1e-12,
                )
            ),
        }

    return result


def _arm_values(
    metrics: Mapping[
        str,
        Mapping[
            str,
            Mapping[str, float | int],
        ],
    ],
    group: str,
    field: str,
) -> dict[str, float]:
    return {
        "F0": float(
            metrics["F0_240"][group][field]
        ),
        "F1": float(
            metrics["F1_480"][group][field]
        ),
        "F2": float(
            metrics["F2_960"][group][field]
        ),
    }


def _relative_changes(
    comparison: Mapping[
        str,
        Mapping[str, float],
    ],
    *,
    source: str,
    target: str,
) -> dict[str, float]:
    return {
        "top10_improvement": (
            comparison["top10_endpoint"][source]
            - comparison["top10_endpoint"][target]
        )
        / max(
            abs(
                comparison[
                    "top10_endpoint"
                ][source]
            ),
            1e-12,
        ),
        "top1_improvement": (
            comparison["top1_endpoint"][source]
            - comparison["top1_endpoint"][target]
        )
        / max(
            abs(
                comparison[
                    "top1_endpoint"
                ][source]
            ),
            1e-12,
        ),
        "smooth90_improvement": (
            comparison["smooth90_endpoint"][source]
            - comparison["smooth90_endpoint"][target]
        )
        / max(
            abs(
                comparison[
                    "smooth90_endpoint"
                ][source]
            ),
            1e-12,
        ),
        "overall_improvement": (
            comparison["overall_endpoint"][source]
            - comparison["overall_endpoint"][target]
        )
        / max(
            abs(
                comparison[
                    "overall_endpoint"
                ][source]
            ),
            1e-12,
        ),
        "global_cosine_change": (
            comparison["global_cosine"][target]
            - comparison["global_cosine"][source]
        ),
        "prediction_to_gt_norm_change": (
            comparison[
                "prediction_to_gt_norm"
            ][target]
            - comparison[
                "prediction_to_gt_norm"
            ][source]
        ),
    }


def _same_contract_except_strides(
    configs: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> bool:
    normalized = []

    for arm in (
        "F0_240",
        "F1_480",
        "F2_960",
    ):
        config = copy.deepcopy(
            dict(configs[arm])
        )

        image_encoder = config.setdefault(
            "image_encoder",
            {},
        )

        image_encoder[
            "first_stride"
        ] = "resolution_arm"

        image_encoder[
            "second_stride"
        ] = "resolution_arm"

        config["screening"][
            "arm"
        ] = "resolution_arm"

        normalized.append(config)

    return (
        normalized[0]
        == normalized[1]
        == normalized[2]
    )


def _initialization_equal(
    configs: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> bool:
    states = []

    for arm in (
        "F0_240",
        "F1_480",
        "F2_960",
    ):
        seed = int(
            configs[arm]["seed"]
        )

        _seed_everything(seed)

        state = _build_model(
            configs[arm],
            None,
            False,
        ).state_dict()

        states.append(state)

    reference = states[0]

    for state in states[1:]:
        if reference.keys() != state.keys():
            return False

        for key in reference:
            if not torch.equal(
                reference[key],
                state[key],
            ):
                return False

    return True


def _read_json(
    path: Path,
) -> dict[str, Any]:
    result = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return result


def _report(
    summary: Mapping[str, Any],
) -> str:
    comparison = summary[
        "comparison_original_rgb"
    ]

    lines = [
        "# Sofa50 image feature resolution ablation",
        "",
        "| metric | F0 240x240 | F1 480x480 | F2 960x960 |",
        "|---|---:|---:|---:|",
    ]

    for name, values in comparison.items():
        lines.append(
            f"| {name} | "
            f"{values['F0']:.6f} | "
            f"{values['F1']:.6f} | "
            f"{values['F2']:.6f} |"
        )

    for label in (
        "F1_vs_F0",
        "F2_vs_F0",
        "F2_vs_F1",
    ):
        lines.extend(
            (
                "",
                f"## {label.replace('_', ' ')}",
                "",
                "```json",
                json.dumps(
                    summary[
                        "relative_changes"
                    ][label],
                    indent=2,
                ),
                "```",
            )
        )

    lines.extend(
        (
            "",
            "## Original versus zero RGB",
            "",
            "```json",
            json.dumps(
                summary[
                    "original_vs_zero_rgb_gap"
                ],
                indent=2,
            ),
            "```",
            "",
            "## Feature map shapes",
            "",
            "```json",
            json.dumps(
                summary[
                    "feature_map_shapes_vchw"
                ],
                indent=2,
            ),
            "```",
            "",
            "## Contract audit",
            "",
            "```json",
            json.dumps(
                summary[
                    "contract_audit"
                ],
                indent=2,
            ),
            "```",
            "",
        )
    )

    return "\n".join(lines)
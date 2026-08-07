from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .geometry_aware_sampling import _magnitude_masks
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _amp_settings, _build_model, _prepare_item_for_use, _prepare_object_static
from .trainer import _seed_everything, load_checkpoint


ARM_LAYOUT = {"F0_240": "image_resolution_f0", "F1_480": "image_resolution_f1"}
GROUPS = ("all", "smooth_bottom_90", "high_top_10", "high_top_1")


def analyze_image_resolution_ablation(
    output_root: str | Path, manifest_path: str | Path, *, device: str = "cuda"
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    arm_dirs = {name: output_root / "arms" / arm for name, arm in ARM_LAYOUT.items()}
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    condition_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    feature_shapes: dict[str, list[int]] = {}
    fixed_prediction_max_abs_difference: dict[str, float] = {}
    for arm, arm_dir in arm_dirs.items():
        config = _read_json(arm_dir / "config.json")
        dataset = PreparedMeshDataset.from_manifest(
            manifest_path, "validation"
        )
        result = _evaluate_original_and_zero(
            arm_dir, config, dataset, resolved_device
        )
        condition_metrics[arm] = result["metrics"]
        feature_shapes[arm] = result["feature_map_shape"]
        fixed_prediction_max_abs_difference[arm] = result[
            "fixed_prediction_max_abs_difference"
        ]

    original = {
        arm: values["original_rgb"] for arm, values in condition_metrics.items()
    }
    zero = {arm: values["zero_rgb"] for arm, values in condition_metrics.items()}
    comparison = {
        "overall_endpoint": _arm_pair(original, "all", "endpoint"),
        "top10_endpoint": _arm_pair(original, "high_top_10", "endpoint"),
        "top1_endpoint": _arm_pair(original, "high_top_1", "endpoint"),
        "smooth90_endpoint": _arm_pair(original, "smooth_bottom_90", "endpoint"),
        "global_cosine": _arm_pair(original, "all", "global_cosine"),
        "prediction_to_gt_norm": _arm_pair(original, "all", "prediction_to_gt_norm"),
    }
    changes = {
        "top10_improvement": _endpoint_improvement(comparison["top10_endpoint"]),
        "top1_improvement": _endpoint_improvement(comparison["top1_endpoint"]),
        "smooth90_degradation": -_endpoint_improvement(
            comparison["smooth90_endpoint"]
        ),
        "overall_degradation": -_endpoint_improvement(comparison["overall_endpoint"]),
        "global_cosine_change": (
            comparison["global_cosine"]["F1"] - comparison["global_cosine"]["F0"]
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
    verdict = (
        "Supported"
        if changes["top10_improvement"] >= 0.05
        and changes["top1_improvement"] >= 0.05
        and changes["smooth90_degradation"] <= 0.02
        and changes["overall_degradation"] <= 0.02
        else "Not supported"
        if changes["top10_improvement"] <= 0.0
        and changes["top1_improvement"] <= 0.0
        else "Inconclusive"
    )
    configs = {
        arm: _read_json(path / "config.json") for arm, path in arm_dirs.items()
    }
    summaries = {
        arm: _read_json(path / "screening_summary.json")
        for arm, path in arm_dirs.items()
    }
    summary = {
        "experiment": "Sofa50 image feature resolution ablation, 1000 optimizer steps",
        "verdict": verdict,
        "comparison_original_rgb": comparison,
        "relative_changes_f1_vs_f0": changes,
        "original_vs_zero_rgb_gap": rgb_gaps,
        "condition_metrics": condition_metrics,
        "feature_map_shapes_vchw": feature_shapes,
        "contract_audit": {
            "same_contract_except_second_stride": _same_contract_except_stride(configs),
            "same_initial_parameter_tensors": _initialization_equal(configs),
            "same_seed": len({int(config["seed"]) for config in configs.values()}) == 1,
            "uniform_full_vertex_training": all(
                config["training"]["vertex_sampling"]["mode"] == "full"
                for config in configs.values()
            ),
            "fresh_start": all(
                config["screening"]["resume_checkpoint"] is None
                for config in configs.values()
            ),
            "optimizer_steps": {
                arm: int(value["optimizer_steps"]) for arm, value in summaries.items()
            },
            "fixed_original_prediction_max_abs_difference": fixed_prediction_max_abs_difference,
        },
        "recovery": {"performed": False},
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


@torch.no_grad()
def _evaluate_original_and_zero(
    arm_dir: Path,
    config: Mapping[str, Any],
    dataset: PreparedMeshDataset,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False).to(device)
    load_checkpoint(arm_dir / "best.pt", model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    arrays: dict[str, dict[str, list[np.ndarray]]] = {
        condition: {
            "prediction": [],
            "target": [],
            **{group: [] for group in GROUPS},
        }
        for condition in ("original_rgb", "zero_rgb")
    }
    prediction_dir = arm_dir / "rgb_resolution_ablation_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    feature_map_shape: list[int] | None = None
    max_difference = 0.0
    exact_config = copy.deepcopy(dict(config))
    exact_config["query_training"]["apply_to_validation"] = False
    for index in range(len(dataset)):
        prepared = _prepare_item_for_use(
            _prepare_object_static(dataset.load_static(index), exact_config),
            exact_config,
            device,
            cache_on_device=False,
            decode_images=True,
        )
        base = dict(prepared.sample)
        base["query_positions"] = base["vertices"]
        base["query_is_exact"] = torch.ones(
            len(base["vertices"]), dtype=torch.bool, device=device
        )
        if feature_map_shape is None:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                feature_map_shape = list(model.image_encoder(base["images"][:1]).shape)
        predictions: dict[str, np.ndarray] = {}
        for condition in ("original_rgb", "zero_rgb"):
            sample = dict(base)
            if condition == "zero_rgb":
                sample["images"] = torch.zeros_like(base["images"])
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                prediction = model(sample).predicted_laplacian.float()
            predictions[condition] = prediction.cpu().numpy()
        target = prepared.training_target.float().cpu().numpy()
        masks = _magnitude_masks(np.linalg.norm(target, axis=1))
        sample_id = str(base["sample_id"])
        fixed = np.load(
            arm_dir / "fixed_query_predictions" / f"{sample_id}__exact.npz"
        )["prediction"]
        max_difference = max(
            max_difference, float(np.max(np.abs(fixed - predictions["original_rgb"])))
        )
        np.savez_compressed(
            prediction_dir / f"{sample_id}.npz",
            target=target,
            original_rgb=predictions["original_rgb"],
            zero_rgb=predictions["zero_rgb"],
        )
        for condition, prediction in predictions.items():
            arrays[condition]["prediction"].append(prediction)
            arrays[condition]["target"].append(target)
            for group in GROUPS:
                arrays[condition][group].append(masks[group])
        del prepared, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = {
        condition: _condition_metrics(values) for condition, values in arrays.items()
    }
    if feature_map_shape is None:
        raise RuntimeError("Validation dataset was empty.")
    return {
        "metrics": metrics,
        "feature_map_shape": feature_map_shape,
        "fixed_prediction_max_abs_difference": max_difference,
    }


def _condition_metrics(values: Mapping[str, list[np.ndarray]]) -> dict[str, Any]:
    prediction = np.concatenate(values["prediction"])
    target = np.concatenate(values["target"])
    result: dict[str, Any] = {}
    for group in GROUPS:
        mask = np.concatenate(values[group]).astype(bool)
        pred = prediction[mask].astype(np.float64)
        gt = target[mask].astype(np.float64)
        endpoint = np.linalg.norm(pred - gt, axis=1)
        result[group] = {
            "vertex_count": int(mask.sum()),
            "endpoint": float(endpoint.mean()),
            "global_cosine": float(
                np.dot(pred.reshape(-1), gt.reshape(-1))
                / max(np.linalg.norm(pred) * np.linalg.norm(gt), 1e-12)
            ),
            "prediction_to_gt_norm": float(
                np.linalg.norm(pred) / max(np.linalg.norm(gt), 1e-12)
            ),
        }
    return result


def _arm_pair(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    group: str,
    field: str,
) -> dict[str, float]:
    return {
        "F0": float(metrics["F0_240"][group][field]),
        "F1": float(metrics["F1_480"][group][field]),
    }


def _endpoint_improvement(pair: Mapping[str, float]) -> float:
    return (pair["F0"] - pair["F1"]) / max(abs(pair["F0"]), 1e-12)


def _same_contract_except_stride(configs: Mapping[str, Mapping[str, Any]]) -> bool:
    left = copy.deepcopy(dict(configs["F0_240"]))
    right = copy.deepcopy(dict(configs["F1_480"]))
    left["image_encoder"]["second_stride"] = "paired"
    right["image_encoder"]["second_stride"] = "paired"
    left["screening"]["arm"] = "paired"
    right["screening"]["arm"] = "paired"
    return left == right


def _initialization_equal(configs: Mapping[str, Mapping[str, Any]]) -> bool:
    seed = int(configs["F0_240"]["seed"])
    _seed_everything(seed)
    f0 = _build_model(configs["F0_240"], None, False).state_dict()
    _seed_everything(seed)
    f1 = _build_model(configs["F1_480"], None, False).state_dict()
    return f0.keys() == f1.keys() and all(torch.equal(f0[key], f1[key]) for key in f0)


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def _report(summary: Mapping[str, Any]) -> str:
    comparison = summary["comparison_original_rgb"]
    lines = [
        "# Sofa50 image feature resolution ablation",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "| metric | F0 240x240 | F1 480x480 |",
        "|---|---:|---:|",
    ]
    for name, pair in comparison.items():
        lines.append(f"| {name} | {pair['F0']:.6f} | {pair['F1']:.6f} |")
    lines.extend(
        (
            "",
            "## F1 versus F0",
            "",
            "```json",
            json.dumps(summary["relative_changes_f1_vs_f0"], indent=2),
            "```",
            "",
            "## Original versus zero RGB",
            "",
            "```json",
            json.dumps(summary["original_vs_zero_rgb_gap"], indent=2),
            "```",
            "",
        )
    )
    return "\n".join(lines)

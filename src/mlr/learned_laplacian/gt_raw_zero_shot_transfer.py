from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.laplacian import unique_edges
from mlr.refinement import RefinementConfig

from .canonical_experiment import _exact_query_sample, _first_camera, _load_device_item
from .diagnostics import _amp_settings
from .evaluation import (
    _chamfer_distance,
    _normal_consistency,
    _point_to_surface_stats,
    _reconstruct,
)
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .synthetic_current_comparison import _topology_change
from .target_scaling import EDGE_SCALE_NORMALIZED_LAPLACIAN, RAW_LAPLACIAN
from .trainer import load_checkpoint
from .visibility_recovery import confidence_aware_recovery_weight
from .visualization import render_mesh_comparison_grid


GT_CONDITIONS = ("correct_rgb", "zero_rgb", "shuffled_view_rgb")
PRIMARY_ARM = "B_gt_query_direct_raw_hf"
NORMALIZED_ARM = "A_previous_gt_query_h2_normalized"
CURRENT_ARM = "C_current_query_direct_raw_hf"


def run_gt_raw_zero_shot_transfer(
    new_run: str | Path,
    gt_manifest: str | Path,
    current_manifest: str | Path,
    normalized_gt_run: str | Path,
    current_hf_run: str | Path,
    current_hf_analysis: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    seed: int = 7,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The transfer evaluator requires CUDA.")

    gt_path = Path(gt_manifest).resolve()
    current_path = Path(current_manifest).resolve()
    gt_datasets = {
        split: PreparedMeshDataset.from_manifest(gt_path, split)
        for split in ("train", "validation", "test")
    }
    current_datasets = {
        split: PreparedMeshDataset.from_manifest(current_path, split)
        for split in ("train", "validation", "test")
    }
    validate_disjoint_splits(*gt_datasets.values())
    validate_disjoint_splits(*current_datasets.values())

    new_spec = _load_spec(Path(new_run).resolve(), resolved)
    normalized_spec = _load_spec(Path(normalized_gt_run).resolve(), resolved, load_model=False)
    current_spec = _load_spec(Path(current_hf_run).resolve(), resolved, load_model=False)
    preflight = _preflight(
        gt_path,
        current_path,
        gt_datasets,
        current_datasets,
        new_spec,
        normalized_spec,
        current_spec,
        Path(current_hf_analysis).resolve(),
    )
    _write_json(output / "contract_audit_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("GT raw transfer preflight failed.")

    gt_rows: list[dict[str, Any]] = []
    gt_arrays: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gt_prediction_root = output / "raw_predictions" / "gt_query"
    for split in ("validation", "test"):
        dataset = gt_datasets[split]
        for index in range(len(dataset)):
            values = _infer_gt_conditions(
                new_spec, dataset, index, resolved, seed=seed
            )
            sample_id = str(values["sample_id"])
            target = values["target"]
            valid = values["valid"]
            payload: dict[str, np.ndarray] = {
                "delta_gt_raw": target.numpy(),
                "valid_mask": valid.numpy(),
            }
            for condition in GT_CONDITIONS:
                prediction = values[condition]["prediction"]
                confidence = values[condition]["confidence"]
                weight = values[condition]["weight"]
                metrics = _raw_metrics_by_gt_magnitude(
                    prediction, target, weight, valid
                )
                gt_rows.append(
                    {
                        "split": split,
                        "arm": PRIMARY_ARM,
                        "condition": condition,
                        "sample_id": sample_id,
                        "vertex_count": int(len(target)),
                        "valid_vertex_count": int(valid.sum().item()),
                        "mean_confidence": float(confidence.mean().item()),
                        **metrics,
                    }
                )
                group = gt_arrays[(split, condition)]
                group["prediction"].append(prediction[valid].numpy())
                group["target"].append(target[valid].numpy())
                group["weight"].append(weight[valid].numpy())
                payload[f"delta_pred_raw__{condition}"] = prediction.numpy()
                payload[f"confidence__{condition}"] = confidence.numpy()
                payload[f"recovery_weight__{condition}"] = weight.numpy()
            path = gt_prediction_root / split / f"{sample_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **payload)
            print(f"GT {split} {sample_id} evaluated", flush=True)
            torch.cuda.empty_cache()
    gt_aggregate = _aggregate_gt_arrays(gt_arrays)

    gt_validation_reference = np.concatenate(
        gt_arrays[("validation", "correct_rgb")]["prediction"]
    )
    reference_magnitude = np.linalg.norm(gt_validation_reference, axis=1)
    reference_stats = _distribution(reference_magnitude)

    current_test = current_datasets["test"]
    current_rows: list[dict[str, Any]] = []
    safe_input_audits: list[dict[str, Any]] = []
    for index in range(len(current_test)):
        static = current_test.load_static(index)
        sample_id = str(static["sample_id"])
        safe = _inference_only_sample(static)
        safe_input_audits.append(_safe_input_audit(safe))
        values = _infer_safe_conditions(
            new_spec, safe, resolved, seed=seed + index * 104729
        )
        payload: dict[str, np.ndarray] = {
            "vertices_current": static["vertices"].cpu().numpy(),
            "faces_current": static["faces"].cpu().numpy(),
            "local_edge_length": static["local_edge_length"].cpu().numpy(),
        }
        for condition in GT_CONDITIONS:
            prediction = values[condition]["prediction"]
            confidence = values[condition]["confidence"]
            weight = values[condition]["weight"]
            prediction_stats = _coarse_prediction_stats(
                prediction,
                static["local_edge_length"],
                reference_stats,
            )
            recovery_dir = output / "recovered_meshes" / PRIMARY_ARM / condition / sample_id
            recovery = _recover_raw_only(
                static,
                prediction,
                confidence,
                weight,
                recovery_dir,
                new_spec["config"],
                seed,
            )
            current_rows.append(
                {
                    "arm": PRIMARY_ARM,
                    "condition": condition,
                    "sample_id": sample_id,
                    "object_id": static.get("metadata", {}).get("object_id"),
                    "variant_index": static.get("metadata", {}).get("variant_index"),
                    **prediction_stats,
                    **recovery,
                }
            )
            payload[f"delta_pred_raw__{condition}"] = prediction.numpy()
            payload[f"confidence__{condition}"] = confidence.numpy()
            payload[f"recovery_weight__{condition}"] = weight.numpy()
        prediction_path = output / "raw_predictions" / "current_query" / f"{sample_id}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, **payload)
        print(f"current {sample_id} B controls recovered", flush=True)
        torch.cuda.empty_cache()

    del new_spec["model"]
    torch.cuda.empty_cache()
    normalized_spec = _load_spec(Path(normalized_gt_run).resolve(), resolved)
    for index in range(len(current_test)):
        static = current_test.load_static(index)
        sample_id = str(static["sample_id"])
        safe = _inference_only_sample(static)
        values = _infer_safe_conditions(
            normalized_spec,
            safe,
            resolved,
            seed=seed + index * 104729,
            conditions=("correct_rgb",),
        )
        output_prediction = values["correct_rgb"]["prediction"]
        h = static["local_edge_length"].float().cpu()
        prediction_raw = output_prediction * (
            h.square() + float(normalized_spec["config"].get("target_scaling", {}).get("epsilon", 1e-12))
        ).unsqueeze(-1)
        confidence = values["correct_rgb"]["confidence"]
        weight = values["correct_rgb"]["weight"]
        recovery = _recover_raw_only(
            static,
            prediction_raw,
            confidence,
            weight,
            output / "recovered_meshes" / NORMALIZED_ARM / "correct_rgb" / sample_id,
            normalized_spec["config"],
            seed,
        )
        current_rows.append(
            {
                "arm": NORMALIZED_ARM,
                "condition": "correct_rgb",
                "sample_id": sample_id,
                "object_id": static.get("metadata", {}).get("object_id"),
                "variant_index": static.get("metadata", {}).get("variant_index"),
                **_coarse_prediction_stats(prediction_raw, h, reference_stats),
                **recovery,
            }
        )
        prediction_path = (
            output / "raw_predictions" / "normalized_gt_baseline" / f"{sample_id}.npz"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            delta_hat_prediction=output_prediction.numpy(),
            delta_pred_raw=prediction_raw.numpy(),
            confidence=confidence.numpy(),
            recovery_weight=weight.numpy(),
        )
        print(f"current {sample_id} A recovered", flush=True)
        torch.cuda.empty_cache()
    del normalized_spec["model"]
    torch.cuda.empty_cache()

    current_hf_rows = _load_current_hf_rows(Path(current_hf_analysis).resolve())
    current_hf_prediction_context = _current_hf_prediction_context(
        Path(current_hf_analysis).resolve()
    )
    current_rows.extend(current_hf_rows)
    current_aggregate = _aggregate_current(current_rows)
    graph_diagnostics = {
        "gt_training_graphs": _graph_distribution(gt_datasets["train"]),
        "current_test_graphs": _graph_distribution(current_test),
    }
    _write_json(output / "graph_distribution_diagnostics.json", graph_diagnostics)
    visual_failures = _render_visuals(
        current_test,
        current_rows,
        output,
        Path(current_hf_analysis).resolve(),
    )
    comparison = _comparison(current_aggregate)
    final_audit = {
        **preflight,
        "safe_inference_samples_all_targets_zero_and_gt_removed": all(
            row["passed"] for row in safe_input_audits
        ),
        "safe_inference_sample_audits": safe_input_audits,
        "new_raw_output_used_without_h2_conversion": True,
        "normalized_baseline_h2_conversion_isolated_to_arm_a": True,
        "coarse_vertexwise_raw_error_metrics_emitted": False,
        "gt_used_after_prediction_only_for_surface_geometry_evaluation": True,
        "passed": bool(
            preflight["passed"]
            and all(row["passed"] for row in safe_input_audits)
            and len([row for row in current_rows if row["arm"] == PRIMARY_ARM]) == 75
            and len([row for row in current_rows if row["arm"] == NORMALIZED_ARM]) == 25
            and len([row for row in current_rows if row["arm"] == CURRENT_ARM]) == 25
        ),
    }
    _write_json(output / "contract_audit.json", final_audit)
    if not final_audit["passed"]:
        raise RuntimeError("Final transfer contract audit failed.")

    _write_csv(output / "gt_query_prediction_per_sample.csv", gt_rows)
    _write_csv(output / "gt_query_prediction_aggregate.csv", gt_aggregate)
    _write_csv(output / "current_zero_shot_per_sample.csv", current_rows)
    _write_csv(output / "current_zero_shot_aggregate.csv", current_aggregate)
    summary = {
        "experiment": "GT direct-raw Laplacian training to current-graph zero-shot transfer",
        "target_formula": "delta_gt_raw = L_gt @ V_gt",
        "h2_normalization_used_by_new_model": False,
        "training": _training_summary(new_spec),
        "execution_hardware_difference": (
            "The new controlled run used 2x RTX PRO 6000 Blackwell because the "
            "2x L40 scheduler estimate was 2026-08-30. The optimizer/world-size/"
            "global-batch contract is unchanged, but hardware and PyTorch environment "
            "are recorded execution differences from the historical L40 comparators."
        ),
        "gt_query_prediction": gt_aggregate,
        "current_zero_shot_geometry": current_aggregate,
        "current_query_direct_raw_existing_prediction_context": current_hf_prediction_context,
        "graph_distribution_diagnostics": graph_diagnostics,
        "comparison": comparison,
        "contract_audit": final_audit,
        "visualization_failures": visual_failures,
        "historical_comparator_limitations": [
            "Arm A is the existing pre-HF normalized GT-query run, so A versus B also differs in image feature construction and effective global batch.",
            "Arm C is trained on current graphs; its stored raw prediction scores use the legitimate synthetic P_proxy target but are not relabelled as zero-shot coarse raw EPE here.",
        ],
    }
    _write_json(output / "aggregate_summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _load_spec(
    run_dir: Path, device: torch.device, *, load_model: bool = True
) -> dict[str, Any]:
    run_config = _read_json(run_dir / "run_config.json")
    config = run_config.get("experiment_config", run_config)
    checkpoint = _checkpoint(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    spec: dict[str, Any] = {
        "run_dir": run_dir,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": metrics,
        "run_config": run_config,
    }
    if load_model:
        inference_config = copy.deepcopy(config)
        inference_config.setdefault("query_training", {})["enabled"] = False
        inference_config.setdefault("local_query_jitter", {})["enabled"] = False
        model = _build_model(inference_config, None, False).to(device)
        payload = load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        amp_enabled, amp_dtype = _amp_settings(inference_config, device)
        spec.update(
            {
                "config": inference_config,
                "model": model,
                "checkpoint_payload": payload,
                "amp_enabled": amp_enabled,
                "amp_dtype": amp_dtype,
            }
        )
    return spec


def _checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt", "checkpoint_latest.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _preflight(
    gt_manifest: Path,
    current_manifest: Path,
    gt_datasets: Mapping[str, PreparedMeshDataset],
    current_datasets: Mapping[str, PreparedMeshDataset],
    new_spec: Mapping[str, Any],
    normalized_spec: Mapping[str, Any],
    current_spec: Mapping[str, Any],
    current_analysis: Path,
) -> dict[str, Any]:
    config = new_spec["config"]
    split_counts = {
        "gt": {key: len(value) for key, value in gt_datasets.items()},
        "current": {key: len(value) for key, value in current_datasets.items()},
    }
    new_ids = _run_manifest_ids(new_spec["run_config"])
    expected_gt_ids = {key: list(value.sample_ids) for key, value in gt_datasets.items()}
    current_rows = _load_current_hf_rows(current_analysis)
    current_ids = sorted(current_datasets["test"].sample_ids)
    recovery_equal = config.get("recovery") == current_spec["config"].get("recovery")
    fixed = bool(
        config.get("target_mode") == RAW_LAPLACIAN
        and config.get("target_definition") == "delta_gt_raw=L_gt@V_gt"
        and config.get("training", {}).get("prediction_loss_space") == "output_representation"
        and config.get("training", {}).get("loss") == "huber"
        and float(config.get("training", {}).get("huber_delta")) == 0.01
        and not config.get("query_training", {}).get("enabled", False)
        and not config.get("local_query_jitter", {}).get("enabled", False)
        and config.get("image_encoder", {}).get("feature_construction", {}).get("mode")
        == "original_plus_high_frequency"
        and int(config.get("image_encoder", {}).get("feature_dim")) == 64
        and int(config.get("model", {}).get("hidden_dim")) == 256
        and int(config.get("model", {}).get("num_graph_layers")) == 3
    )
    optimizer_steps = int(new_spec["metrics"].get("optimizer_steps", -1))
    current_row_ids = sorted(row["sample_id"] for row in current_rows)
    passed = bool(
        split_counts["gt"] == {"train": 40, "validation": 5, "test": 5}
        and split_counts["current"] == {"train": 200, "validation": 25, "test": 25}
        and new_ids == expected_gt_ids
        and current_row_ids == current_ids
        and len(current_rows) == 25
        and fixed
        and optimizer_steps == 20_000
        and int(new_spec["metrics"].get("global_batch_meshes", -1)) == 2
        and recovery_equal
        and normalized_spec["config"].get("target_mode")
        == EDGE_SCALE_NORMALIZED_LAPLACIAN
        and current_spec["config"].get("target_mode") == RAW_LAPLACIAN
    )
    return {
        "passed": passed,
        "gt_manifest": str(gt_manifest),
        "gt_manifest_sha256": _sha256(gt_manifest),
        "current_manifest": str(current_manifest),
        "current_manifest_sha256": _sha256(current_manifest),
        "split_counts": split_counts,
        "new_run_sample_ids_match_gt_manifest": new_ids == expected_gt_ids,
        "same_25_current_test_samples_as_existing_hf_report": current_row_ids == current_ids,
        "fixed_c2f2_28view_960_hf_raw_contract": fixed,
        "optimizer_steps": optimizer_steps,
        "effective_global_batch_meshes": new_spec["metrics"].get("global_batch_meshes"),
        "same_recovery_config_as_current_hf": recovery_equal,
        "new_checkpoint": str(new_spec["checkpoint"]),
        "new_checkpoint_sha256": new_spec["checkpoint_sha256"],
        "same_checkpoint_for_gt_and_current_inference": True,
        "no_finetuning_or_calibration": True,
    }


def _run_manifest_ids(run_config: Mapping[str, Any]) -> dict[str, list[str]]:
    value = run_config.get("source_manifest", {})
    if isinstance(value, str):
        value = _read_json(Path(value))
    output = {split: [] for split in ("train", "validation", "test")}
    for item in value.get("samples", []):
        split = str(item.get("split"))
        if split in output:
            output[split].append(str(item.get("sample_id")))
    return output


@torch.no_grad()
def _infer_gt_conditions(
    spec: Mapping[str, Any],
    dataset: PreparedMeshDataset,
    index: int,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    prepared = _load_device_item(dataset, index, spec["config"], device)
    base = _exact_query_sample(prepared.sample, device)
    target = prepared.raw_target.float().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().cpu()
    result: dict[str, Any] = {
        "sample_id": str(base["sample_id"]),
        "target": target,
        "valid": valid,
    }
    permutation = torch.randperm(
        int(base["images"].shape[0]), generator=torch.Generator().manual_seed(seed + index)
    ).to(device)
    for condition in GT_CONDITIONS:
        sample = _condition(base, condition, permutation)
        with torch.autocast(
            device_type=device.type,
            dtype=spec["amp_dtype"],
            enabled=bool(spec["amp_enabled"]),
        ):
            model_output = spec["model"](sample)
        prediction = model_output.predicted_laplacian.float().cpu()
        confidence = _required_confidence(model_output).cpu()
        weight = confidence_aware_recovery_weight(
            prepared.sample["visibility"].cpu(),
            confidence,
            num_vertices=len(prediction),
        ).cpu()
        result[condition] = {
            "prediction": prediction,
            "confidence": confidence,
            "weight": weight,
        }
    return result


def _inference_only_sample(static: Mapping[str, Any]) -> dict[str, Any]:
    sample = dict(static)
    vertices = torch.as_tensor(sample["vertices"])
    zeros = torch.zeros_like(vertices)
    sample["laplacian_target"] = zeros
    sample["raw_laplacian_target"] = zeros
    sample["normalized_laplacian_target"] = zeros
    sample["target_positions"] = vertices.clone()
    sample["target_confidence"] = torch.ones(len(vertices), dtype=torch.float32)
    sample.pop("gt_vertices", None)
    sample.pop("gt_faces", None)
    metadata = dict(sample.get("metadata", {}))
    for key in list(metadata):
        if "gt" in key.lower() or "target" in key.lower() or "proxy" in key.lower():
            metadata.pop(key)
    metadata["inference_only_target_fields_zeroed"] = True
    sample["metadata"] = metadata
    return sample


def _safe_input_audit(sample: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("laplacian_target", "raw_laplacian_target", "normalized_laplacian_target")
    zero = all(torch.count_nonzero(torch.as_tensor(sample[name])).item() == 0 for name in fields)
    passed = bool(
        zero
        and "gt_vertices" not in sample
        and "gt_faces" not in sample
        and torch.equal(sample["target_positions"], sample["vertices"])
    )
    return {"sample_id": str(sample["sample_id"]), "passed": passed}


@torch.no_grad()
def _infer_safe_conditions(
    spec: Mapping[str, Any],
    safe: Mapping[str, Any],
    device: torch.device,
    *,
    seed: int,
    conditions: Sequence[str] = GT_CONDITIONS,
) -> dict[str, dict[str, torch.Tensor]]:
    prepared = _prepare_item_for_use(
        _prepare_object_static(safe, spec["config"]),
        spec["config"],
        device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )
    base = _exact_query_sample(prepared.sample, device)
    permutation = torch.randperm(
        int(base["images"].shape[0]), generator=torch.Generator().manual_seed(seed)
    ).to(device)
    output: dict[str, dict[str, torch.Tensor]] = {}
    for condition in conditions:
        sample = _condition(base, condition, permutation)
        with torch.autocast(
            device_type=device.type,
            dtype=spec["amp_dtype"],
            enabled=bool(spec["amp_enabled"]),
        ):
            model_output = spec["model"](sample)
        prediction = model_output.predicted_laplacian.float().cpu()
        confidence = _required_confidence(model_output).cpu()
        weight = confidence_aware_recovery_weight(
            prepared.sample["visibility"].cpu(),
            confidence,
            num_vertices=len(prediction),
        ).cpu()
        output[condition] = {
            "prediction": prediction,
            "confidence": confidence,
            "weight": weight,
        }
    return output


def _condition(
    base: Mapping[str, Any], condition: str, permutation: torch.Tensor
) -> dict[str, Any]:
    sample = dict(base)
    if condition == "correct_rgb":
        return sample
    if condition == "zero_rgb":
        sample["images"] = torch.zeros_like(base["images"])
        return sample
    if condition == "shuffled_view_rgb":
        sample["images"] = base["images"].index_select(0, permutation)
        return sample
    raise ValueError(f"Unknown image condition: {condition}")


def _required_confidence(model_output: Any) -> torch.Tensor:
    confidence = model_output.confidence_prediction
    if confidence is None:
        raise RuntimeError("The controlled model requires a confidence head.")
    return confidence.float()


def _raw_metrics_by_gt_magnitude(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recovery_weight: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction[valid].double()
    target = target[valid].double()
    weight = recovery_weight[valid].double().clamp_min(0.0)
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    magnitude = torch.linalg.vector_norm(target, dim=-1)
    order = torch.argsort(magnitude, descending=True, stable=True)
    top10_count = max(1, int(math.ceil(0.10 * len(order))))
    top1_count = max(1, int(math.ceil(0.01 * len(order))))
    top10 = order[:top10_count]
    top1 = order[:top1_count]
    bottom90 = order[top10_count:]
    cosine = F.cosine_similarity(
        prediction.reshape(1, -1), target.reshape(1, -1), dim=-1, eps=1e-12
    )
    return {
        "raw_epe": float(error.mean().item()),
        "raw_rms": float(torch.sqrt(error.square().mean()).item()),
        "recovery_weighted_raw_rms": float(
            torch.sqrt(
                (weight * error.square()).sum() / weight.sum().clamp_min(1e-12)
            ).item()
        ),
        "raw_max": float(error.max().item()),
        "raw_cosine": float(cosine.item()),
        "prediction_to_gt_raw_magnitude_ratio": float(
            (
                torch.linalg.vector_norm(prediction)
                / torch.linalg.vector_norm(target).clamp_min(1e-12)
            ).item()
        ),
        "bottom_90_raw_epe": float(error[bottom90].mean().item()),
        "top_10_raw_epe": float(error[top10].mean().item()),
        "top_1_raw_epe": float(error[top1].mean().item()),
    }


def _aggregate_gt_arrays(
    arrays: Mapping[tuple[str, str], Mapping[str, Sequence[np.ndarray]]]
) -> list[dict[str, Any]]:
    rows = []
    for (split, condition), fields in sorted(arrays.items()):
        prediction = torch.from_numpy(np.concatenate(fields["prediction"]))
        target = torch.from_numpy(np.concatenate(fields["target"]))
        weight = torch.from_numpy(np.concatenate(fields["weight"]))
        rows.append(
            {
                "split": split,
                "arm": PRIMARY_ARM,
                "condition": condition,
                "vertex_count": len(prediction),
                **_raw_metrics_by_gt_magnitude(
                    prediction,
                    target,
                    weight,
                    torch.ones(len(prediction), dtype=torch.bool),
                ),
            }
        )
    return rows


def _coarse_prediction_stats(
    prediction: torch.Tensor,
    local_h: torch.Tensor,
    reference: Mapping[str, float],
) -> dict[str, float]:
    magnitude = torch.linalg.vector_norm(prediction.double(), dim=-1).numpy()
    h = torch.as_tensor(local_h).double().numpy()
    relative = magnitude / np.maximum(h, 1e-12)
    stats = _distribution(magnitude)
    threshold = float(reference["p99"])
    return {
        "delta_pred_raw_mean_magnitude": stats["mean"],
        "delta_pred_raw_median_magnitude": stats["median"],
        "delta_pred_raw_p90_magnitude": stats["p90"],
        "delta_pred_raw_p99_magnitude": stats["p99"],
        "delta_pred_raw_max_magnitude": stats["max"],
        "delta_pred_raw_over_h_mean": float(np.mean(relative)),
        "delta_pred_raw_over_h_p99": float(np.quantile(relative, 0.99)),
        "mean_magnitude_over_gt_validation_prediction_mean": stats["mean"]
        / max(float(reference["mean"]), 1e-12),
        "p99_magnitude_over_gt_validation_prediction_p99": stats["p99"]
        / max(float(reference["p99"]), 1e-12),
        "fraction_above_gt_validation_prediction_p99": float(np.mean(magnitude > threshold)),
    }


def _recover_raw_only(
    static: Mapping[str, Any],
    prediction_raw: torch.Tensor,
    confidence: torch.Tensor,
    weight: torch.Tensor,
    output_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = torch.as_tensor(static["vertices"]).cpu().numpy().astype(np.float64)
    faces = torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64)
    prediction = prediction_raw.cpu().numpy().astype(np.float64)
    recovery = config.get("recovery", {})
    refinement = RefinementConfig(
        operator_type=str(recovery.get("operator_type", "uniform")),
        lambda_lap=float(recovery.get("lambda_lap", 1.0)),
        lambda_anchor=float(recovery.get("lambda_anchor", 0.01)),
        lambda_edge=float(recovery.get("lambda_edge", 0.0)),
        lambda_unseen_anchor=float(recovery.get("unseen_anchor_weight", 0.0)),
        num_iters=int(recovery.get("num_iters", 200)),
        learning_rate=float(recovery.get("learning_rate", 0.01)),
        robust_loss=str(recovery.get("robust_loss", "huber")),
        huber_delta=float(recovery.get("huber_delta", 0.01)),
    )
    coarse = Mesh(vertices, faces).ensure_normals()
    result, solver = _reconstruct(
        coarse,
        prediction,
        np.ones(len(vertices), dtype=np.float64),
        refinement,
        dense_vertex_limit=5000,
        laplacian_weight=weight.cpu().numpy().astype(np.float64),
    )
    gt = Mesh(
        torch.as_tensor(static["gt_vertices"]).cpu().numpy(),
        torch.as_tensor(static["gt_faces"]).cpu().numpy().astype(np.int64),
    ).ensure_normals()
    coarse_forward = _point_to_surface_stats(coarse.vertices, gt)
    coarse_reverse = _point_to_surface_stats(gt.vertices, coarse)
    refined_forward = _point_to_surface_stats(result.mesh.vertices, gt)
    refined_reverse = _point_to_surface_stats(gt.vertices, result.mesh)
    initial_chamfer = float(_chamfer_distance(coarse, gt, samples=3000, seed=seed))
    refined_chamfer = float(_chamfer_distance(result.mesh, gt, samples=3000, seed=seed))
    initial_p2s = 0.5 * (float(coarse_forward["mean"]) + float(coarse_reverse["mean"]))
    refined_p2s = 0.5 * (float(refined_forward["mean"]) + float(refined_reverse["mean"]))
    displacement = np.linalg.norm(result.vertices - vertices, axis=1)
    h = torch.as_tensor(static["local_edge_length"]).cpu().numpy().astype(np.float64)
    displacement_over_h = displacement / np.maximum(h, 1e-12)
    topology = _topology_change(vertices, result.vertices, faces)
    np.save(output_dir / "delta_pred_raw.npy", prediction)
    np.save(output_dir / "confidence_prediction.npy", confidence.cpu().numpy())
    np.save(output_dir / "recovery_weight.npy", weight.cpu().numpy())
    np.save(output_dir / "displacement.npy", displacement)
    save_mesh(coarse, output_dir / "current.obj")
    save_mesh(result.mesh, output_dir / "refined.obj")
    _write_json(
        output_dir / "recovery_config.json",
        {
            **dict(recovery),
            "solver": solver,
            "solver_input": "delta_pred_raw_direct",
            "h2_normalization_or_denormalization_before_solver": (
                config.get("target_mode") == EDGE_SCALE_NORMALIZED_LAPLACIAN
            ),
        },
    )
    return {
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "initial_chamfer": initial_chamfer,
        "refined_chamfer": refined_chamfer,
        "initial_point_to_surface": initial_p2s,
        "refined_point_to_surface": refined_p2s,
        "initial_normal_consistency": float(_normal_consistency(coarse, gt)),
        "refined_normal_consistency": float(_normal_consistency(result.mesh, gt)),
        "introduced_flipped_faces": int(topology["introduced_flips"]),
        "new_degenerate_faces": int(topology["new_degeneracies"]),
        "mean_displacement": float(np.mean(displacement)),
        "median_displacement": float(np.median(displacement)),
        "p95_displacement": float(np.quantile(displacement, 0.95)),
        "max_displacement": float(np.max(displacement)),
        "mean_displacement_over_local_h": float(np.mean(displacement_over_h)),
        "p95_displacement_over_local_h": float(np.quantile(displacement_over_h, 0.95)),
        "improved_over_initial": refined_chamfer < initial_chamfer,
        "percentage_chamfer_improvement": 100.0 * (initial_chamfer - refined_chamfer)
        / max(initial_chamfer, 1e-12),
        "mean_confidence": float(confidence.mean().item()),
        "visible_weight_fraction": float((weight > 0).float().mean().item()),
    }


def _load_current_hf_rows(analysis: Path) -> list[dict[str, Any]]:
    source = analysis / "recovery_per_sample.csv"
    rows = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] != "C_original_plus_high_frequency":
                continue
            rows.append(
                {
                    "arm": CURRENT_ARM,
                    "condition": "correct_rgb",
                    "sample_id": row["sample_id"],
                    "object_id": row.get("object_id"),
                    "variant_index": int(row["variant_index"]),
                    "initial_chamfer": float(row["initial_chamfer"]),
                    "refined_chamfer": float(row["reconstruction_chamfer"]),
                    "initial_point_to_surface": float(row["initial_point_to_surface"]),
                    "refined_point_to_surface": float(row["reconstruction_point_to_surface"]),
                    "initial_normal_consistency": float(row["initial_normal_consistency"]),
                    "refined_normal_consistency": float(row["reconstruction_normal_consistency"]),
                    "introduced_flipped_faces": int(row["introduced_flipped_faces"]),
                    "new_degenerate_faces": int(row["new_degenerate_faces"]),
                    "improved_over_initial": row["improved_over_initial"].lower() == "true",
                    "percentage_chamfer_improvement": 100.0
                    * (float(row["initial_chamfer"]) - float(row["reconstruction_chamfer"]))
                    / max(float(row["initial_chamfer"]), 1e-12),
                    "mean_confidence": float(row["mean_confidence"]),
                    "visible_weight_fraction": float(row["visible_vertex_fraction"]),
                    "source": "existing audited current-HF evaluation",
                }
            )
    return rows


def _current_hf_prediction_context(analysis: Path) -> dict[str, Any]:
    summary = _read_json(analysis / "image_feature_ablation_summary.json")
    aggregates = [
        row
        for row in summary["prediction_aggregate"]
        if row["arm"] == "C_original_plus_high_frequency"
    ]
    magnitude_groups = [
        row
        for row in summary["gt_raw_laplacian_magnitude_groups"]
        if row["arm"] == "C_original_plus_high_frequency"
    ]
    return {
        "source": str(analysis / "image_feature_ablation_summary.json"),
        "scope": (
            "Existing current-query supervised prediction context only; these scores "
            "are not reported as coarse zero-shot Raw EPE for the GT-trained model."
        ),
        "prediction_aggregate": aggregates,
        "gt_magnitude_groups": magnitude_groups,
    }


def _aggregate_current(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(str(row["arm"]), str(row["condition"])) for row in rows})
    geometry = (
        "initial_chamfer",
        "refined_chamfer",
        "initial_point_to_surface",
        "refined_point_to_surface",
        "initial_normal_consistency",
        "refined_normal_consistency",
    )
    optional = (
        "mean_displacement",
        "median_displacement",
        "p95_displacement",
        "max_displacement",
        "mean_displacement_over_local_h",
        "p95_displacement_over_local_h",
    )
    for arm, condition in keys:
        selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition]
        item: dict[str, Any] = {
            "arm": arm,
            "condition": condition,
            "sample_count": len(selected),
            **{field: float(np.mean([float(row[field]) for row in selected])) for field in geometry},
            "introduced_flipped_faces": sum(int(row["introduced_flipped_faces"]) for row in selected),
            "improved_over_initial_count": sum(bool(row["improved_over_initial"]) for row in selected),
            "worsened_count": sum(not bool(row["improved_over_initial"]) for row in selected),
            "mean_percentage_chamfer_improvement": float(
                np.mean([float(row["percentage_chamfer_improvement"]) for row in selected])
            ),
            "median_percentage_chamfer_improvement": float(
                np.median([float(row["percentage_chamfer_improvement"]) for row in selected])
            ),
        }
        for field in optional:
            available = [float(row[field]) for row in selected if field in row]
            item[field] = float(np.mean(available)) if available else None
        output.append(item)
    return output


def _graph_distribution(dataset: PreparedMeshDataset) -> dict[str, Any]:
    vertices_all: list[np.ndarray] = []
    h_all: list[np.ndarray] = []
    valence_all: list[np.ndarray] = []
    vertex_counts = []
    edge_counts = []
    bbox_diagonals = []
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        vertices = sample["vertices"].cpu().numpy().astype(np.float64)
        faces = sample["faces"].cpu().numpy().astype(np.int64)
        edges = unique_edges(faces)
        valence = np.zeros(len(vertices), dtype=np.int64)
        np.add.at(valence, edges[:, 0], 1)
        np.add.at(valence, edges[:, 1], 1)
        vertices_all.append(vertices)
        h_all.append(sample["local_edge_length"].cpu().numpy().astype(np.float64))
        valence_all.append(valence)
        vertex_counts.append(len(vertices))
        edge_counts.append(len(edges))
        bbox_diagonals.append(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))))
    xyz = np.concatenate(vertices_all)
    h = np.concatenate(h_all)
    valence = np.concatenate(valence_all)
    center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
    radius = np.linalg.norm(xyz - center, axis=1)
    return {
        "sample_count": len(dataset),
        "vertex_count": _distribution(np.asarray(vertex_counts)),
        "edge_count": _distribution(np.asarray(edge_counts)),
        "local_edge_length_h": _distribution(h),
        "vertex_valence": _distribution(valence),
        "bbox_diagonal": _distribution(np.asarray(bbox_diagonals)),
        "query_position": {
            axis: _distribution(xyz[:, index]) for index, axis in enumerate(("x", "y", "z"))
        },
        "query_radius_from_global_bbox_center": _distribution(radius),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p50": float(np.quantile(values, 0.50)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _render_visuals(
    dataset: PreparedMeshDataset,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    current_analysis: Path,
) -> list[dict[str, str]]:
    failures = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        try:
            current = Mesh(
                static["vertices"].cpu().numpy(), static["faces"].cpu().numpy()
            ).ensure_normals()
            gt = Mesh(
                static["gt_vertices"].cpu().numpy(), static["gt_faces"].cpu().numpy()
            ).ensure_normals()
            c_mesh = current_analysis / "reconstruction" / "C_original_plus_high_frequency" / sample_id / "predicted_refined.obj"
            entries = [
                ("GT", gt),
                ("Current", current),
                ("A normalized GT", load_mesh(output / "recovered_meshes" / NORMALIZED_ARM / "correct_rgb" / sample_id / "refined.obj")),
                ("B raw GT", load_mesh(output / "recovered_meshes" / PRIMARY_ARM / "correct_rgb" / sample_id / "refined.obj")),
                ("B zero RGB", load_mesh(output / "recovered_meshes" / PRIMARY_ARM / "zero_rgb" / sample_id / "refined.obj")),
                ("B shuffled RGB", load_mesh(output / "recovered_meshes" / PRIMARY_ARM / "shuffled_view_rgb" / sample_id / "refined.obj")),
                ("C current raw HF", load_mesh(c_mesh)),
            ]
            render_mesh_comparison_grid(
                entries,
                _first_camera(static, 320),
                output / "fixed_camera_visualizations" / f"{sample_id}.png",
                image_size=320,
                columns=4,
            )
        except Exception as error:
            failures.append({"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"})
    return failures


def _comparison(aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keyed = {(row["arm"], row["condition"]): row for row in aggregate}
    a = keyed[(NORMALIZED_ARM, "correct_rgb")]
    b = keyed[(PRIMARY_ARM, "correct_rgb")]
    c = keyed[(CURRENT_ARM, "correct_rgb")]
    return {
        "B_minus_A": {
            "refined_chamfer": b["refined_chamfer"] - a["refined_chamfer"],
            "refined_point_to_surface": b["refined_point_to_surface"] - a["refined_point_to_surface"],
            "refined_normal_consistency": b["refined_normal_consistency"] - a["refined_normal_consistency"],
            "improved_count": b["improved_over_initial_count"] - a["improved_over_initial_count"],
        },
        "B_minus_C": {
            "refined_chamfer": b["refined_chamfer"] - c["refined_chamfer"],
            "refined_point_to_surface": b["refined_point_to_surface"] - c["refined_point_to_surface"],
            "refined_normal_consistency": b["refined_normal_consistency"] - c["refined_normal_consistency"],
            "improved_count": b["improved_over_initial_count"] - c["improved_over_initial_count"],
        },
    }


def _training_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    metrics = spec["metrics"]
    run = spec["run_config"]
    inventories = sorted(Path(spec["run_dir"]).glob("gpu_inventory_*.csv"))
    return {
        "run_dir": str(spec["run_dir"]),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "optimizer_steps": metrics.get("optimizer_steps"),
        "best_epoch": metrics.get("best_epoch"),
        "best_selection_loss": metrics.get("best_selection_loss"),
        "final_validation_loss": metrics.get("final_validation_loss"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "distributed_world_size": metrics.get("distributed_world_size"),
        "effective_global_batch_meshes": metrics.get("global_batch_meshes"),
        "seed": spec["config"].get("seed"),
        "git_commit": run.get("git_commit", run.get("git", {}).get("commit")),
        "gpu_inventory": (
            inventories[-1].read_text(encoding="utf-8").strip()
            if inventories
            else None
        ),
    }


def _report(summary: Mapping[str, Any]) -> str:
    gt = {
        (row["split"], row["condition"]): row for row in summary["gt_query_prediction"]
    }
    geometry = {
        (row["arm"], row["condition"]): row for row in summary["current_zero_shot_geometry"]
    }
    c_prediction = {
        row["split"]: row
        for row in summary["current_query_direct_raw_existing_prediction_context"][
            "prediction_aggregate"
        ]
    }
    c_groups = {
        row["split"]: row
        for row in summary["current_query_direct_raw_existing_prediction_context"][
            "gt_magnitude_groups"
        ]
    }
    lines = [
        "# Sofa50 GT-query direct-raw zero-shot transfer",
        "",
        "## Contract",
        "",
        "The trained target is exactly `delta_gt_raw = L_gt @ V_gt`. The new model uses no h² normalization, no normalized-space loss, and no inference-time denormalization. Coarse/current inference uses target-scrubbed model inputs; GT is introduced only after prediction for surface evaluation.",
        "",
        f"- Contract audit: `{summary['contract_audit']['passed']}`",
        f"- Optimizer steps: `{summary['training']['optimizer_steps']}`",
            f"- Runtime: `{summary['training']['runtime_seconds']}` seconds",
            f"- GPU inventory: `{summary['training']['gpu_inventory']}`",
        "",
        "## Held-out GT-query prediction",
        "",
        "| Split | RGB | Raw EPE | Raw RMS | RW RMS | Bottom90 | Top10 | Top1 | Cosine | Magnitude ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for condition in GT_CONDITIONS:
            row = gt[(split, condition)]
            lines.append(
                f"| {split} | {condition} | {row['raw_epe']:.8g} | {row['raw_rms']:.8g} | {row['recovery_weighted_raw_rms']:.8g} | {row['bottom_90_raw_epe']:.8g} | {row['top_10_raw_epe']:.8g} | {row['top_1_raw_epe']:.8g} | {row['raw_cosine']:.8g} | {row['prediction_to_gt_raw_magnitude_ratio']:.8g} |"
            )
    c_test = c_prediction["test"]
    c_test_groups = c_groups["test"]
    lines.extend(
        [
            "",
            "### Existing current-query direct-raw HF context",
            "",
            "These are existing supervised current-query scores against `L_current @ P_proxy`; they are context for Arm C and are not assigned to the GT-trained model's zero-shot coarse inference.",
            "",
            f"Test Raw EPE `{c_test['raw_epe']:.8g}`, Raw RMS `{c_test['raw_residual_rms']:.8g}`, RW RMS `{c_test['recovery_weighted_raw_residual_rms']:.8g}`, Bottom90 `{c_test_groups['bottom_90_percent_mean_raw_error_epe']:.8g}`, Top10 `{c_test_groups['top_10_percent_mean_raw_error_epe']:.8g}`, Top1 `{c_test_groups['top_1_percent_mean_raw_error_epe']:.8g}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Current-mesh zero-shot recovery",
            "",
            "No coarse vertexwise Raw EPE is reported. The table contains only common-surface geometry and topology metrics.",
            "",
            "| Arm | RGB | Chamfer | P2S | Normal | Flips | Improved/25 | Mean % improvement |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = [
        (NORMALIZED_ARM, "correct_rgb"),
        (PRIMARY_ARM, "correct_rgb"),
        (PRIMARY_ARM, "zero_rgb"),
        (PRIMARY_ARM, "shuffled_view_rgb"),
        (CURRENT_ARM, "correct_rgb"),
    ]
    for key in order:
        row = geometry[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {row['refined_chamfer']:.8g} | {row['refined_point_to_surface']:.8g} | {row['refined_normal_consistency']:.8g} | {row['introduced_flipped_faces']} | {row['improved_over_initial_count']}/25 | {row['mean_percentage_chamfer_improvement']:.4g}% |"
        )
    a = geometry[(NORMALIZED_ARM, "correct_rgb")]
    b = geometry[(PRIMARY_ARM, "correct_rgb")]
    c = geometry[(CURRENT_ARM, "correct_rgb")]
    materially_better = b["refined_chamfer"] < a["refined_chamfer"] and b[
        "refined_point_to_surface"
    ] < a["refined_point_to_surface"]
    approaches_c = abs(b["refined_chamfer"] - c["refined_chamfer"]) <= 0.1 * abs(
        a["refined_chamfer"] - c["refined_chamfer"]
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Removing normalization materially improves over the historical normalized GT-query comparator: **{'yes' if materially_better else 'no'}**.",
            f"- The GT-trained raw model approaches the current-query raw-HF result within 10% of the historical-to-current Chamfer gap: **{'yes' if approaches_c else 'no'}**.",
            "- These are recovery conclusions; GT-query prediction quality and RGB controls are reported separately above.",
            "- Arm A is historical and predates the HF feature construction, so A→B is informative but not a strict one-variable causal estimate of normalization alone.",
            f"- Execution note: {summary['execution_hardware_difference']}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

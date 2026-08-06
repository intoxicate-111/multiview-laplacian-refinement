from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .evaluation import reconstruct_and_evaluate
from .losses import weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from .target_scaling import (
    denormalize_laplacian_by_edge_scale,
    normalize_laplacian_by_edge_scale,
)
from .trainer import load_checkpoint


PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def run_laplacian_diagnostics(
    run_dir: str | Path,
    *,
    split: str = "validation",
    output_dir: str | Path | None = None,
    device: str = "cuda",
    seed: int = 7,
    overwrite: bool = False,
    skip_reconstruction: bool = False,
) -> dict[str, Any]:
    """Diagnose one completed multi-mesh run without mutating its artifacts."""

    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir or run_dir / "diagnostics").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Diagnostics directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = _read_json(run_dir / "config.json")
    run_metrics = _read_json(run_dir / "metrics.json")
    manifest = run_dir / "dataset_manifest.json"
    checkpoint = run_dir / "best.pt"
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Run directory requires dataset_manifest.json and best.pt.")
    if split != "validation":
        raise ValueError("The complete diagnostic currently requires split='validation'.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    train_dataset = PreparedMeshDataset.from_manifest(manifest, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest, split)
    model = _build_model(config, None, False).to(resolved_device)
    checkpoint_payload = load_checkpoint(
        checkpoint, model, map_location=resolved_device
    )
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)
    loss_kwargs = _loss_kwargs(config)

    print("Computing exact-query train predictions and training-target mean...", flush=True)
    train_records = _predict_split(
        model,
        train_dataset,
        config,
        resolved_device,
        loss_kwargs,
        amp_enabled,
        amp_dtype,
        retain_arrays=False,
    )
    target_sum = np.sum(
        [np.asarray(record["target_sum"], dtype=np.float64) for record in train_records],
        axis=0,
    )
    target_count = int(sum(int(record["valid_vertex_count"]) for record in train_records))
    if target_count < 1:
        raise ValueError("Training set has no valid target vertices.")
    global_mean = target_sum / target_count

    print("Computing exact-query validation predictions...", flush=True)
    validation_records = _predict_split(
        model,
        validation_dataset,
        config,
        resolved_device,
        loss_kwargs,
        amp_enabled,
        amp_dtype,
        retain_arrays=True,
    )
    model_validation_loss = float(np.mean([r["model_loss"] for r in validation_records]))
    train_set_eval_loss = float(np.mean([r["model_loss"] for r in train_records]))

    baseline = _baseline_comparison(
        validation_records, global_mean, model_validation_loss, loss_kwargs
    )
    _write_json(output_dir / "baseline_comparison.json", baseline)
    _write_baseline_csv(output_dir / "baseline_comparison.csv", baseline)

    target_all, prediction_all, confidence_all, _ = _concatenate_valid(
        validation_records
    )
    magnitude = _magnitude_statistics(
        validation_records, target_all, prediction_all
    )
    _write_json(output_dir / "magnitude_statistics.json", magnitude)
    _write_magnitude_csv(output_dir / "magnitude_statistics.csv", magnitude)
    _plot_magnitudes(output_dir, target_all, prediction_all, magnitude["ratio_threshold"])

    error_bins = _error_by_magnitude(
        validation_records,
        target_all,
        prediction_all,
        confidence_all,
        loss_kwargs,
        magnitude["ratio_threshold"],
    )
    _write_json(output_dir / "error_by_magnitude_bin.json", error_bins)
    _write_error_bins_csv(output_dir / "error_by_magnitude_bin.csv", error_bins)
    _plot_error_bins(output_dir / "error_by_magnitude_bin.png", error_bins["overall"])

    visual_metadata = _write_validation_mesh_visualizations(
        output_dir / "meshes", validation_records, target_all, prediction_all,
        magnitude["ratio_threshold"]
    )

    round_trip = _normalization_round_trip(validation_dataset, config)
    _write_json(output_dir / "normalization_round_trip.json", round_trip)

    comparability = _train_validation_comparability(
        config,
        run_metrics,
        train_records,
        validation_records,
        train_set_eval_loss,
        model_validation_loss,
    )
    _write_json(output_dir / "train_validation_comparability.json", comparability)

    reconstruction: dict[str, Any]
    if skip_reconstruction:
        reconstruction = {"skipped": True, "per_mesh": []}
    else:
        reconstruction = _run_reconstruction(
            output_dir / "reconstruction",
            validation_dataset,
            validation_records,
            config,
        )
    _write_json(output_dir / "reconstruction_summary.json", reconstruction)

    report = _diagnostic_report(
        run_dir=run_dir,
        checkpoint_payload=checkpoint_payload,
        run_metrics=run_metrics,
        config=config,
        baseline=baseline,
        magnitude=magnitude,
        error_bins=error_bins,
        reconstruction=reconstruction,
        round_trip=round_trip,
        comparability=comparability,
        visual_metadata=visual_metadata,
    )
    (output_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", checkpoint_payload.get("step", -1))),
        "baseline": baseline["overall"],
        "magnitude_ratio_global": magnitude["overall"]["magnitude_ratio_global"],
        "top_10": error_bins["overall"]["top_10"],
        "top_1": error_bins["overall"]["top_1"],
        "normalization_round_trip": round_trip["overall"],
        "reconstruction": reconstruction.get("overall", {"skipped": True}),
    }
    _write_json(output_dir / "diagnostic_summary.json", summary)
    return summary


@torch.no_grad()
def _predict_split(
    model: torch.nn.Module,
    dataset: PreparedMeshDataset,
    config: Mapping[str, Any],
    device: torch.device,
    loss_kwargs: Mapping[str, Any],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    *,
    retain_arrays: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        prepared = _prepare_object_static(static, config)
        prepared = _prepare_item_for_use(
            prepared,
            config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        sample = dict(prepared.sample)
        sample["query_positions"] = sample["vertices"]
        sample["query_is_exact"] = torch.ones(
            sample["vertices"].shape[0], dtype=torch.bool, device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            prediction = model(sample).predicted_laplacian
        prediction = prediction.float()
        target = prepared.training_target.float()
        confidence = sample["target_confidence"].float()
        valid = sample["valid_scale_mask"].bool() & (confidence > 0)
        if not bool(valid.any()):
            raise ValueError(f"Sample {sample['sample_id']!r} has no valid vertices.")
        loss = weighted_robust_laplacian_loss(
            prediction, target, confidence, **loss_kwargs
        )
        valid_target = target[valid].detach().cpu().numpy().astype(np.float64)
        record: dict[str, Any] = {
            "sample_id": str(sample["sample_id"]),
            "vertex_count": int(target.shape[0]),
            "valid_vertex_count": int(valid.sum().item()),
            "model_loss": float(loss.item()),
            "target_sum": valid_target.sum(axis=0).tolist(),
        }
        if retain_arrays:
            record.update(
                target=target.detach().cpu().numpy(),
                prediction=prediction.detach().cpu().numpy(),
                confidence=confidence.detach().cpu().numpy(),
                valid_mask=valid.detach().cpu().numpy(),
                static_sample={
                    "vertices": static["vertices"].detach().cpu().numpy(),
                    "faces": static["faces"].detach().cpu().numpy(),
                },
            )
        records.append(record)
        print(
            f"  {sample['sample_id']}: vertices={target.shape[0]} loss={loss.item():.8f}",
            flush=True,
        )
        del prepared, sample, prediction, target, confidence
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def _baseline_comparison(
    records: Sequence[Mapping[str, Any]],
    global_mean: np.ndarray,
    model_validation_loss: float,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    per_mesh = []
    for record in records:
        target = torch.from_numpy(np.asarray(record["target"]))
        confidence = torch.from_numpy(np.asarray(record["confidence"]))
        zero = torch.zeros_like(target)
        mean_prediction = torch.as_tensor(global_mean, dtype=target.dtype).expand_as(target)
        zero_loss = float(
            weighted_robust_laplacian_loss(zero, target, confidence, **loss_kwargs).item()
        )
        mean_loss = float(
            weighted_robust_laplacian_loss(
                mean_prediction, target, confidence, **loss_kwargs
            ).item()
        )
        model_loss = float(record["model_loss"])
        per_mesh.append(
            {
                "sample_id": record["sample_id"],
                "valid_vertex_count": int(record["valid_vertex_count"]),
                "zero_baseline_loss": zero_loss,
                "global_mean_baseline_loss": mean_loss,
                "model_validation_loss": model_loss,
                "relative_improvement_vs_zero": _relative_improvement(zero_loss, model_loss),
                "relative_improvement_vs_global_mean": _relative_improvement(mean_loss, model_loss),
            }
        )
    zero_loss = float(np.mean([item["zero_baseline_loss"] for item in per_mesh]))
    mean_loss = float(np.mean([item["global_mean_baseline_loss"] for item in per_mesh]))
    overall = {
        "zero_baseline_loss": zero_loss,
        "global_mean_baseline_loss": mean_loss,
        "model_validation_loss": model_validation_loss,
        "relative_improvement_vs_zero": _relative_improvement(zero_loss, model_validation_loss),
        "relative_improvement_vs_global_mean": _relative_improvement(
            mean_loss, model_validation_loss
        ),
    }
    return {
        **overall,
        "overall": overall,
        "global_mean_target": {
            "mean_x": float(global_mean[0]),
            "mean_y": float(global_mean[1]),
            "mean_z": float(global_mean[2]),
            "source": "training valid target vertices only",
        },
        "reduction": "weighted Huber within each mesh, arithmetic mean across meshes",
        "per_mesh": per_mesh,
    }


def _magnitude_statistics(
    records: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    target_magnitude = np.linalg.norm(target, axis=1)
    prediction_magnitude = np.linalg.norm(prediction, axis=1)
    positive = target_magnitude[target_magnitude > 0]
    positive_median = float(np.median(positive)) if positive.size else 0.0
    ratio_threshold = max(1e-8, 1e-3 * positive_median)
    overall = _magnitude_scope(target, prediction, ratio_threshold)
    per_mesh = []
    for record in records:
        valid = np.asarray(record["valid_mask"], dtype=bool)
        target_values = np.asarray(record["target"])[valid]
        prediction_values = np.asarray(record["prediction"])[valid]
        per_mesh.append(
            {
                "sample_id": record["sample_id"],
                **_magnitude_scope(target_values, prediction_values, ratio_threshold),
            }
        )
    return {
        "ratio_threshold": ratio_threshold,
        "ratio_threshold_definition": "max(1e-8, 1e-3 * median positive validation target magnitude)",
        "overall": overall,
        "per_mesh": per_mesh,
    }


def _magnitude_scope(
    target: np.ndarray,
    prediction: np.ndarray,
    ratio_threshold: float,
) -> dict[str, Any]:
    target_magnitude = np.linalg.norm(target, axis=1)
    prediction_magnitude = np.linalg.norm(prediction, axis=1)
    error = np.linalg.norm(prediction - target, axis=1)
    stable = target_magnitude > ratio_threshold
    ratios = prediction_magnitude[stable] / target_magnitude[stable]
    return {
        "vertex_count": int(target_magnitude.size),
        "stable_ratio_vertex_count": int(stable.sum()),
        "target_magnitude": _distribution(target_magnitude),
        "prediction_magnitude": _distribution(prediction_magnitude),
        "error_magnitude": _distribution(error),
        "magnitude_ratio_global": _safe_div(
            float(prediction_magnitude.mean()), float(target_magnitude.mean())
        ),
        "mean_vertex_ratio": float(ratios.mean()) if ratios.size else 0.0,
        "median_vertex_ratio": float(np.median(ratios)) if ratios.size else 0.0,
    }


def _error_by_magnitude(
    records: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    loss_kwargs: Mapping[str, Any],
    cosine_threshold: float,
) -> dict[str, Any]:
    magnitudes = np.linalg.norm(target, axis=1)
    thresholds = {
        "p50": float(np.quantile(magnitudes, 0.50)),
        "p90": float(np.quantile(magnitudes, 0.90)),
        "p95": float(np.quantile(magnitudes, 0.95)),
        "p99": float(np.quantile(magnitudes, 0.99)),
    }
    definitions = _bin_masks(magnitudes, thresholds)
    overall = {
        name: _bin_metrics(
            target, prediction, confidence, mask, loss_kwargs, cosine_threshold
        )
        for name, mask in definitions.items()
    }
    per_mesh = []
    for record in records:
        valid = np.asarray(record["valid_mask"], dtype=bool)
        target_mesh = np.asarray(record["target"])[valid]
        prediction_mesh = np.asarray(record["prediction"])[valid]
        confidence_mesh = np.asarray(record["confidence"])[valid]
        mesh_magnitudes = np.linalg.norm(target_mesh, axis=1)
        masks = _bin_masks(mesh_magnitudes, thresholds)
        per_mesh.append(
            {
                "sample_id": record["sample_id"],
                "bins": {
                    name: _bin_metrics(
                        target_mesh,
                        prediction_mesh,
                        confidence_mesh,
                        mask,
                        loss_kwargs,
                        cosine_threshold,
                    )
                    for name, mask in masks.items()
                },
            }
        )
    return {
        "percentile_thresholds": thresholds,
        "bin_definitions": {
            "low": "[0, p50]",
            "medium": "(p50, p90)",
            "high": "[p90, max]",
            "top_10": "[p90, max]",
            "top_5": "[p95, max]",
            "top_1": "[p99, max]",
        },
        "cosine_target_magnitude_threshold": cosine_threshold,
        "overall": overall,
        "per_mesh": per_mesh,
    }


def _bin_masks(values: np.ndarray, thresholds: Mapping[str, float]) -> dict[str, np.ndarray]:
    return {
        "low": values <= thresholds["p50"],
        "medium": (values > thresholds["p50"]) & (values < thresholds["p90"]),
        "high": values >= thresholds["p90"],
        "top_10": values >= thresholds["p90"],
        "top_5": values >= thresholds["p95"],
        "top_1": values >= thresholds["p99"],
    }


def _bin_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    loss_kwargs: Mapping[str, Any],
    cosine_threshold: float,
) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {
            "vertex_count": 0,
            "fraction_of_vertices": 0.0,
            "mean_target_magnitude": 0.0,
            "mean_prediction_magnitude": 0.0,
            "mean_absolute_vector_error": 0.0,
            "mean_squared_error": 0.0,
            "training_loss": 0.0,
            "cosine_similarity": None,
            "cosine_vertex_count": 0,
        }
    target_bin = target[mask]
    prediction_bin = prediction[mask]
    confidence_bin = confidence[mask]
    target_magnitude = np.linalg.norm(target_bin, axis=1)
    prediction_magnitude = np.linalg.norm(prediction_bin, axis=1)
    residual = prediction_bin - target_bin
    stable = target_magnitude > cosine_threshold
    cosine = None
    if stable.any():
        cosine = float(
            np.mean(
                np.sum(prediction_bin[stable] * target_bin[stable], axis=1)
                / np.maximum(
                    prediction_magnitude[stable] * target_magnitude[stable], 1e-12
                )
            )
        )
    loss = weighted_robust_laplacian_loss(
        torch.from_numpy(prediction_bin),
        torch.from_numpy(target_bin),
        torch.from_numpy(confidence_bin),
        **loss_kwargs,
    )
    return {
        "vertex_count": count,
        "fraction_of_vertices": float(count / len(target)),
        "mean_target_magnitude": float(target_magnitude.mean()),
        "mean_prediction_magnitude": float(prediction_magnitude.mean()),
        "mean_absolute_vector_error": float(np.linalg.norm(residual, axis=1).mean()),
        "mean_squared_error": float(np.mean(residual**2)),
        "training_loss": float(loss.item()),
        "cosine_similarity": cosine,
        "cosine_vertex_count": int(stable.sum()),
    }


def _normalization_round_trip(
    dataset: PreparedMeshDataset, config: Mapping[str, Any]
) -> dict[str, Any]:
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    per_mesh = []
    absolute_errors = []
    raw_values = []
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        raw = sample["raw_laplacian_target"].float()
        local_h = sample["local_edge_length"].float()
        valid = sample["valid_scale_mask"].bool()
        normalized = normalize_laplacian_by_edge_scale(
            raw, local_h, eps=epsilon, valid_scale_mask=valid
        )
        recovered = denormalize_laplacian_by_edge_scale(normalized, local_h)
        error = (recovered - raw)[valid].abs()
        raw_valid = raw[valid].abs()
        absolute_errors.append(error.reshape(-1))
        raw_values.append(raw_valid.reshape(-1))
        per_mesh.append(
            {
                "sample_id": sample["sample_id"],
                "max_abs_error": float(error.max().item()) if error.numel() else 0.0,
                "mean_abs_error": float(error.mean().item()) if error.numel() else 0.0,
                "relative_error": _safe_div(
                    float(error.mean().item()), float(raw_valid.mean().item())
                ),
            }
        )
    all_error = torch.cat(absolute_errors)
    all_raw = torch.cat(raw_values)
    overall = {
        "max_abs_error": float(all_error.max().item()),
        "mean_abs_error": float(all_error.mean().item()),
        "relative_error": _safe_div(
            float(all_error.mean().item()), float(all_raw.mean().item())
        ),
    }
    return {
        "target_mode": config.get("target_mode"),
        "definition": "normalized = raw / (h^2 + epsilon); recovered = normalized * h^2",
        "local_scale_definition": "mean unique incident one-ring edge length per vertex",
        "epsilon": epsilon,
        "overall": overall,
        "per_mesh": per_mesh,
    }


def _train_validation_comparability(
    config: Mapping[str, Any],
    run_metrics: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    train_set_eval_loss: float,
    validation_set_eval_loss: float,
) -> dict[str, Any]:
    query = config.get("query_training", {})
    loading = config.get("data_loading", {})
    model = config.get("model", {})
    return {
        "train_set_eval_loss_exact_query": train_set_eval_loss,
        "validation_set_eval_loss_exact_query": validation_set_eval_loss,
        "eval_generalization_gap": validation_set_eval_loss - train_set_eval_loss,
        "recorded_final_train_loss_augmented": run_metrics.get("final_train_loss"),
        "recorded_final_validation_loss_augmented": run_metrics.get("final_validation_loss"),
        "same_loss_function": True,
        "loss": {
            "type": config.get("training", {}).get("loss", "huber"),
            "huber_delta": config.get("training", {}).get("huber_delta", 0.01),
            "reduction": "confidence-weighted vertex mean per mesh, then mesh mean",
        },
        "train_augmentation_enabled": bool(query.get("enabled", False)),
        "validation_augmentation_enabled_during_training": bool(
            query.get("enabled", False) and query.get("apply_to_validation", True)
        ),
        "diagnostic_augmentation_enabled": False,
        "train_views_per_sample": loading.get("train_views_per_sample"),
        "validation_views_per_sample": loading.get("validation_views_per_sample"),
        "dropout": float(model.get("dropout", 0.0)),
        "batch_norm_present": False,
        "model_eval_used": True,
        "training_epoch_loss_semantics": (
            "average of mesh losses evaluated while weights change during the epoch"
        ),
        "final_eval_checkpoint": "best.pt",
        "train": _split_difficulty_summary(train_records),
        "validation": _split_difficulty_summary(validation_records),
    }


def _split_difficulty_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vertices = np.asarray([record["valid_vertex_count"] for record in records], dtype=np.float64)
    losses = np.asarray([record["model_loss"] for record in records], dtype=np.float64)
    return {
        "mesh_count": len(records),
        "valid_vertex_count": int(vertices.sum()),
        "mean_vertices_per_mesh": float(vertices.mean()),
        "mean_mesh_loss": float(losses.mean()),
        "median_mesh_loss": float(np.median(losses)),
        "min_mesh_loss": float(losses.min()),
        "max_mesh_loss": float(losses.max()),
        "per_mesh": [
            {
                "sample_id": record["sample_id"],
                "valid_vertex_count": int(record["valid_vertex_count"]),
                "loss": float(record["model_loss"]),
            }
            for record in records
        ],
    }


def _write_validation_mesh_visualizations(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    target_all: np.ndarray,
    prediction_all: np.ndarray,
    ratio_threshold: float,
) -> dict[str, Any]:
    from matplotlib import colormaps

    target_all_magnitude = np.linalg.norm(target_all, axis=1)
    prediction_all_magnitude = np.linalg.norm(prediction_all, axis=1)
    error_all = np.linalg.norm(prediction_all - target_all, axis=1)
    magnitude_max = float(
        np.quantile(np.concatenate([target_all_magnitude, prediction_all_magnitude]), 0.99)
    )
    error_max = float(np.quantile(error_all, 0.99))
    stable = target_all_magnitude > ratio_threshold
    ratios = prediction_all_magnitude[stable] / target_all_magnitude[stable]
    ratio_max = float(np.quantile(ratios, 0.99)) if ratios.size else 1.0
    global_metadata = {
        "magnitude_range": [0.0, magnitude_max],
        "magnitude_clipping": "global validation p99 of combined target/prediction magnitudes",
        "error_range": [0.0, error_max],
        "error_clipping": "global validation p99 absolute vector error",
        "direction_error_range": [0.0, 2.0],
        "magnitude_ratio_range": [0.0, ratio_max],
        "magnitude_ratio_clipping": "global validation p99 on stable target vertices",
        "stable_target_threshold": ratio_threshold,
        "near_zero_direction_color": [128, 128, 128],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "global_color_metadata.json", global_metadata)
    for record in records:
        sample_dir = output_root / _safe_name(str(record["sample_id"]))
        sample_dir.mkdir(parents=True, exist_ok=True)
        valid = np.asarray(record["valid_mask"], dtype=bool)
        target = np.asarray(record["target"])
        prediction = np.asarray(record["prediction"])
        target_magnitude = np.linalg.norm(target, axis=1)
        prediction_magnitude = np.linalg.norm(prediction, axis=1)
        error = np.linalg.norm(prediction - target, axis=1)
        stable_mesh = valid & (target_magnitude > ratio_threshold)
        cosine = np.zeros(len(target), dtype=np.float64)
        cosine[stable_mesh] = np.sum(
            target[stable_mesh] * prediction[stable_mesh], axis=1
        ) / np.maximum(
            target_magnitude[stable_mesh] * prediction_magnitude[stable_mesh], 1e-12
        )
        direction_error = 1.0 - cosine
        ratio = np.zeros(len(target), dtype=np.float64)
        ratio[stable_mesh] = prediction_magnitude[stable_mesh] / target_magnitude[stable_mesh]
        sample = record["static_sample"]
        vertices = np.asarray(sample["vertices"])
        faces = np.asarray(sample["faces"])
        entries = (
            ("target_magnitude.ply", target_magnitude, 0.0, magnitude_max, "viridis", None),
            ("prediction_magnitude.ply", prediction_magnitude, 0.0, magnitude_max, "viridis", None),
            ("absolute_error.ply", error, 0.0, error_max, "magma", None),
            (
                "direction_error.ply",
                direction_error,
                0.0,
                2.0,
                "coolwarm",
                ~stable_mesh,
            ),
            ("magnitude_ratio.ply", ratio, 0.0, ratio_max, "viridis", ~stable_mesh),
        )
        for filename, values, minimum, maximum, cmap_name, invalid in entries:
            colors = _colors(values, minimum, maximum, colormaps[cmap_name])
            if invalid is not None:
                colors[invalid] = np.asarray([128, 128, 128], dtype=np.uint8)
            _write_colored_ply(sample_dir / filename, vertices, faces, colors)
        _write_json(
            sample_dir / "metadata.json",
            {"sample_id": record["sample_id"], **global_metadata},
        )
    return global_metadata


def _run_reconstruction(
    output_root: Path,
    dataset: PreparedMeshDataset,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction_config = dict(config.get("reconstruction", {}))
    if not reconstruction_config:
        reconstruction_config = {
            "operator_type": "uniform",
            "lambda_lap": 1.0,
            "lambda_anchor": 0.01,
            "lambda_edge": 0.0,
            "num_iters": 500,
            "learning_rate": 0.01,
            "robust_loss": "huber",
            "huber_delta": 0.01,
            "dense_vertex_limit": 5000,
            "chamfer_samples": 1000,
            "metric_seed": 7,
        }
    record_by_id = {str(record["sample_id"]): record for record in records}
    per_mesh = []
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        sample_id = str(sample["sample_id"])
        record = record_by_id[sample_id]
        target_prediction = torch.from_numpy(np.asarray(record["prediction"]))
        raw_prediction = denormalize_laplacian_by_edge_scale(
            target_prediction, sample["local_edge_length"]
        )
        mesh_output = output_root / _safe_name(sample_id)
        print(f"Reconstructing {sample_id}...", flush=True)
        metrics = reconstruct_and_evaluate(
            sample,
            raw_prediction,
            mesh_output,
            reconstruction_config,
            normalized_prediction=target_prediction,
            edge_scale_epsilon=float(
                config.get("target_scaling", {}).get("epsilon", 1e-12)
            ),
        )
        for source, destination in (
            ("coarse.obj", "initial_mesh.obj"),
            ("oracle_refined.obj", "gt_delta_reconstruction.obj"),
            ("predicted_refined.obj", "predicted_delta_reconstruction.obj"),
        ):
            shutil.copyfile(mesh_output / source, mesh_output / destination)
        metrics["sample_id"] = sample_id
        metrics["input_is_gt_query_mesh"] = bool(
            torch.equal(sample["vertices"], sample["gt_vertices"])
        )
        metrics["reconstruction_config"] = reconstruction_config
        _write_json(mesh_output / "reconstruction_metrics.json", metrics)
        per_mesh.append(metrics)
    overall = _reconstruction_overall(per_mesh)
    return {
        "skipped": False,
        "reconstruction_config": reconstruction_config,
        "important_context": (
            "GT-query validation vertices equal GT vertices; initial_mesh is not a coarse mesh."
        ),
        "overall": overall,
        "per_mesh": per_mesh,
    }


def _reconstruction_overall(per_mesh: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("coarse", "oracle", "predicted")
    result: dict[str, Any] = {}
    for name in names:
        p2s = [item["geometry"][name].get("point_to_surface_mean") for item in per_mesh]
        rmse = [item["geometry"][name].get("target_position_rmse") for item in per_mesh]
        result[name] = {
            "mean_point_to_surface": _mean_optional(p2s),
            "mean_target_position_rmse": _mean_optional(rmse),
            "collapsed_or_exploded_meshes": int(
                sum(bool(item["geometry"][name]["collapsed_or_exploded"]) for item in per_mesh)
            ),
        }
    result["predicted_improves_over_initial_mesh_count"] = int(
        sum(bool(item["predicted_improves_over_coarse"]) for item in per_mesh)
    )
    return result


def _diagnostic_report(
    *,
    run_dir: Path,
    checkpoint_payload: Mapping[str, Any],
    run_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
    magnitude: Mapping[str, Any],
    error_bins: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    round_trip: Mapping[str, Any],
    comparability: Mapping[str, Any],
    visual_metadata: Mapping[str, Any],
) -> str:
    overall_baseline = baseline["overall"]
    overall_magnitude = magnitude["overall"]
    top10 = error_bins["overall"]["top_10"]
    top5 = error_bins["overall"]["top_5"]
    top1 = error_bins["overall"]["top_1"]
    loss_contributions = _partition_loss_contributions(error_bins["overall"])
    reconstruction_overall = reconstruction.get("overall", {})
    conclusions = _conclusions(
        overall_baseline, overall_magnitude, top10, top1, reconstruction_overall
    )
    target_stats = overall_magnitude["target_magnitude"]
    prediction_stats = overall_magnitude["prediction_magnitude"]
    checkpoint_epoch = checkpoint_payload.get("epoch", checkpoint_payload.get("step", "unknown"))
    reconstruction_config = reconstruction.get("reconstruction_config", {})
    lines = [
        "# Laplacian prediction diagnostic report",
        "",
        "## 1. Experiment overview",
        "",
        f"- Run: `{run_dir}`",
        f"- Checkpoint: `best.pt` (epoch {checkpoint_epoch})",
        f"- Training best epoch: {run_metrics.get('best_epoch')}",
        f"- Target mode: `{config.get('target_mode')}`",
        f"- Train/validation meshes: {len(comparability['train']['per_mesh'])} / "
        f"{len(comparability['validation']['per_mesh'])}",
        f"- Loss: `{comparability['loss']['type']}`, Huber delta "
        f"{comparability['loss']['huber_delta']}",
        f"- Reduction: {comparability['loss']['reduction']}",
        f"- Reconstruction config: `{json.dumps(reconstruction_config, sort_keys=True)}`",
        "- Critical context: GT-query validation vertices equal their GT vertices. The saved "
        "`initial_mesh.obj` files are GT-query meshes, not deployment coarse/expanded meshes.",
        "",
        "## 2. Baseline comparison",
        "",
        "| Predictor | Validation loss | Relative improvement |",
        "|---|---:|---:|",
        f"| Zero | {overall_baseline['zero_baseline_loss']:.8g} | — |",
        f"| Global train mean | {overall_baseline['global_mean_baseline_loss']:.8g} | — |",
        f"| Model | {overall_baseline['model_validation_loss']:.8g} | "
        f"{100.0 * overall_baseline['relative_improvement_vs_zero']:.3f}% vs zero; "
        f"{100.0 * overall_baseline['relative_improvement_vs_global_mean']:.3f}% vs mean |",
        "",
        "The global mean vector was computed only from valid training targets: "
        f"`[{baseline['global_mean_target']['mean_x']:.8g}, "
        f"{baseline['global_mean_target']['mean_y']:.8g}, "
        f"{baseline['global_mean_target']['mean_z']:.8g}]`.",
        "",
        "## 3. Magnitude statistics",
        "",
        "| Statistic | Target | Prediction |",
        "|---|---:|---:|",
    ]
    for name in ("mean", "median", "p90", "p95", "p99", "max"):
        lines.append(
            f"| {name} | {target_stats[name]:.8g} | {prediction_stats[name]:.8g} |"
        )
    lines.extend(
        [
            "",
            f"- Global magnitude ratio: {overall_magnitude['magnitude_ratio_global']:.8g}",
            f"- Mean stable per-vertex ratio: {overall_magnitude['mean_vertex_ratio']:.8g}",
            f"- Median stable per-vertex ratio: {overall_magnitude['median_vertex_ratio']:.8g}",
            f"- Ratio/cosine stability threshold: {magnitude['ratio_threshold']:.8g}",
            "",
            "## 4. Error by target magnitude",
            "",
            "| Region | Vertices | Mean target | Mean prediction | Vector error | Training loss | Cosine |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in (("top 10%", top10), ("top 5%", top5), ("top 1%", top1)):
        cosine = values["cosine_similarity"]
        cosine_text = "n/a" if cosine is None else f"{cosine:.6g}"
        lines.append(
            f"| {name} | {values['vertex_count']} | "
            f"{values['mean_target_magnitude']:.6g} | "
            f"{values['mean_prediction_magnitude']:.6g} | "
            f"{values['mean_absolute_vector_error']:.6g} | "
            f"{values['training_loss']:.6g} | {cosine_text} |"
        )
    lines.extend(
        [
            "",
            "Approximate contribution to the vertex-global Huber loss from the disjoint "
            f"low/medium/high partitions is {100 * loss_contributions['low']:.2f}% / "
            f"{100 * loss_contributions['medium']:.2f}% / "
            f"{100 * loss_contributions['high']:.2f}%. Thus flat/low vertices are numerous, "
            "but they do not dominate the loss mass.",
            "The Huber delta is 0.01, so large residuals lie in its linear region and the "
            "reported scalar loss is roughly 0.01 times a component-wise absolute error; its "
            "apparently small numerical value should not be interpreted as small geometric error.",
        ]
    )
    lines.extend(["", "## 5. Reconstruction comparison", ""])
    if reconstruction.get("skipped"):
        lines.append("Reconstruction was skipped.")
    else:
        lines.extend(
            [
                "| Input delta | Mean point-to-surface | Mean target-position RMSE | Collapsed/exploded |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, label in (
            ("coarse", "Initial GT-query mesh"),
            ("oracle", "GT delta"),
            ("predicted", "Predicted delta"),
        ):
            values = reconstruction_overall.get(name, {})
            lines.append(
                f"| {label} | {_format_optional(values.get('mean_point_to_surface'))} | "
                f"{_format_optional(values.get('mean_target_position_rmse'))} | "
                f"{values.get('collapsed_or_exploded_meshes', 0)} |"
            )
        lines.append(
            f"\nPredicted reconstruction improved over initial on "
            f"{reconstruction_overall.get('predicted_improves_over_initial_mesh_count', 0)} "
            "of the validation meshes. Because initial vertices already equal GT here, this is a "
            "solver sanity check rather than a deployment coarse-mesh improvement measurement."
        )
    lines.extend(
        [
            "",
            "## 6. Normalization and train/validation comparability",
            "",
            f"- Round-trip max absolute error: {round_trip['overall']['max_abs_error']:.8g}",
            f"- Round-trip mean absolute error: {round_trip['overall']['mean_abs_error']:.8g}",
            f"- Round-trip relative error: {round_trip['overall']['relative_error']:.8g}",
            f"- Exact-query train-set eval loss: {comparability['train_set_eval_loss_exact_query']:.8g}",
            f"- Exact-query validation-set eval loss: "
            f"{comparability['validation_set_eval_loss_exact_query']:.8g}",
            f"- Exact-query generalization gap (validation - train): "
            f"{comparability['eval_generalization_gap']:.8g}",
            "- Training used query perturbation; this diagnostic disabled perturbation for both "
            "splits. Both splits use all views and model.eval(). Dropout is zero and the model has "
            "no batch normalization.",
            "",
            "## 7. Conclusions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(
        [
            "",
            "## 8. Prioritized next steps",
            "",
            "1. Treat the GT-query/coarse-expanded deployment graph mismatch as the first "
            "correctness/domain-gap check: run the same diagnostics on actual expanded inference "
            "queries with measurable GT correspondence before changing the network.",
            "2. If the baseline and magnitude-bin results show flat-region dominance, test a "
            "magnitude-stratified or curvature-weighted loss as one isolated experiment; retain the "
            "current run as the control.",
            "3. If direction is reasonable but magnitude is contracted, test an explicit magnitude "
            "calibration/weighting experiment before increasing epochs.",
            "4. If GT-delta reconstruction is exact but predicted reconstruction degrades GT-query "
            "meshes, focus on prediction and correspondence. If GT reconstruction fails, fix the "
            "solver/normalization/indexing path first.",
            "5. The validation set has only five meshes; after correctness checks, expand validation "
            "before making architecture decisions.",
            "",
            "## Visualization normalization",
            "",
            f"`{json.dumps(dict(visual_metadata), sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _conclusions(
    baseline: Mapping[str, Any],
    magnitude: Mapping[str, Any],
    top10: Mapping[str, Any],
    top1: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> list[str]:
    conclusions = []
    zero_improvement = float(baseline["relative_improvement_vs_zero"])
    mean_improvement = float(baseline["relative_improvement_vs_global_mean"])
    ratio = float(magnitude["magnitude_ratio_global"])
    if zero_improvement < 0.05:
        conclusions.append("The model is close to the zero predictor under the training loss.")
    else:
        conclusions.append(
            f"The model improves on zero by {100 * zero_improvement:.2f}% under the exact validation reduction."
        )
    if mean_improvement < 0.05:
        conclusions.append("The model is only marginally better than the training-global-mean predictor.")
    else:
        conclusions.append(
            f"The model improves on the training-global-mean predictor by {100 * mean_improvement:.2f}%."
        )
    if ratio < 0.8:
        conclusions.append(
            f"Prediction magnitude is contracted: global prediction/target magnitude ratio is {ratio:.3f}."
        )
    elif ratio > 1.2:
        conclusions.append(
            f"Prediction magnitude is amplified: global prediction/target magnitude ratio is {ratio:.3f}."
        )
    else:
        conclusions.append(f"Global prediction magnitude is near target scale (ratio {ratio:.3f}).")
    if top1["training_loss"] > top10["training_loss"]:
        conclusions.append("The top 1% magnitude vertices are harder than the broader top 10% region.")
    if top10["cosine_similarity"] is not None:
        if top10["cosine_similarity"] < 0.3:
            conclusions.append("High-magnitude predictions have substantial direction error, not only scale error.")
        elif ratio < 0.8:
            conclusions.append("High-magnitude direction has signal, but magnitude remains under-predicted.")
    if reconstruction:
        pred = reconstruction.get("predicted", {})
        oracle = reconstruction.get("oracle", {})
        if (
            pred.get("mean_target_position_rmse") is not None
            and oracle.get("mean_target_position_rmse") is not None
            and pred["mean_target_position_rmse"] > oracle["mean_target_position_rmse"]
        ):
            conclusions.append("Predicted-delta reconstruction is worse than GT-delta reconstruction.")
    conclusions.append(
        "Validation meshes are GT-query graphs and the initial vertices already equal GT; this run alone does not validate coarse/expanded inference correspondence."
    )
    return conclusions


def _partition_loss_contributions(
    bins: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    masses = {
        name: float(bins[name]["fraction_of_vertices"])
        * float(bins[name]["training_loss"])
        for name in ("low", "medium", "high")
    }
    total = sum(masses.values())
    return {name: _safe_div(value, total) for name, value in masses.items()}


def _concatenate_valid(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = []
    predictions = []
    confidences = []
    mesh_ids = []
    for mesh_index, record in enumerate(records):
        valid = np.asarray(record["valid_mask"], dtype=bool)
        targets.append(np.asarray(record["target"])[valid])
        predictions.append(np.asarray(record["prediction"])[valid])
        confidences.append(np.asarray(record["confidence"])[valid])
        mesh_ids.append(np.full(int(valid.sum()), mesh_index, dtype=np.int32))
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(confidences),
        np.concatenate(mesh_ids),
    )


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Distribution requires non-empty finite values.")
    quantiles = np.quantile(values, PERCENTILES)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(quantiles[0]),
        "p75": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(values.max()),
    }


def _plot_magnitudes(
    output_dir: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_magnitude = np.linalg.norm(target, axis=1)
    prediction_magnitude = np.linalg.norm(prediction, axis=1)
    combined = np.concatenate([target_magnitude, prediction_magnitude])
    upper = float(np.quantile(combined, 0.995))
    linear_bins = np.linspace(0.0, max(upper, 1e-12), 100)
    positive = combined[combined > 0]
    lower_log = max(float(positive.min()) if positive.size else threshold, threshold * 0.1)
    upper_log = max(float(np.quantile(positive, 0.995)) if positive.size else 1.0, lower_log * 10)
    log_bins = np.geomspace(lower_log, upper_log, 100)
    for name, values, color in (
        ("target_magnitude_histogram.png", target_magnitude, "tab:blue"),
        ("prediction_magnitude_histogram.png", prediction_magnitude, "tab:orange"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(np.clip(values, 0, upper), bins=linear_bins, color=color)
        axes[0].set_title("Linear magnitude (clipped at validation p99.5)")
        axes[0].set_xlabel("magnitude")
        axes[0].set_ylabel("vertex count")
        axes[1].hist(values[values > 0], bins=log_bins, color=color)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Log magnitude / log count")
        axes[1].set_xlabel("magnitude")
        axes[1].set_ylabel("vertex count")
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)

    stable = (target_magnitude > threshold) & (prediction_magnitude > 0)
    x = target_magnitude[stable]
    y = prediction_magnitude[stable]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    if x.size:
        ax.hexbin(x, y, gridsize=80, xscale="log", yscale="log", bins="log", mincnt=1)
        low = max(min(float(x.min()), float(y.min())), threshold)
        high = max(float(np.quantile(x, 0.999)), float(np.quantile(y, 0.999)))
        ax.plot([low, high], [low, high], "r--", linewidth=1.2, label="y = x")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.legend()
    ax.set_xlabel("target magnitude")
    ax.set_ylabel("prediction magnitude")
    ax.set_title("Target vs prediction magnitude")
    fig.tight_layout()
    fig.savefig(output_dir / "target_vs_prediction_magnitude.png", dpi=180)
    plt.close(fig)


def _plot_error_bins(path: Path, bins: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["low", "medium", "top_10", "top_5", "top_1"]
    x = np.arange(len(names))
    target = [bins[name]["mean_target_magnitude"] for name in names]
    prediction = [bins[name]["mean_prediction_magnitude"] for name in names]
    error = [bins[name]["mean_absolute_vector_error"] for name in names]
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, target, width, label="target magnitude")
    ax.bar(x, prediction, width, label="prediction magnitude")
    ax.bar(x + width, error, width, label="vector error")
    ax.set_xticks(x, names)
    ax.set_yscale("log")
    ax.set_ylabel("mean value (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _colors(values: np.ndarray, minimum: float, maximum: float, colormap: Any) -> np.ndarray:
    denominator = max(maximum - minimum, 1e-12)
    normalized = np.clip((np.asarray(values) - minimum) / denominator, 0.0, 1.0)
    return np.asarray(np.rint(colormap(normalized)[:, :3] * 255.0), dtype=np.uint8)


def _write_colored_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.shape != vertices.shape:
        raise ValueError("Vertex colors must have shape [N, 3].")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            handle.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _write_baseline_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = [
        "scope",
        "sample_id",
        "valid_vertex_count",
        "zero_baseline_loss",
        "global_mean_baseline_loss",
        "model_validation_loss",
        "relative_improvement_vs_zero",
        "relative_improvement_vs_global_mean",
    ]
    rows = [{"scope": "overall", "sample_id": "overall", **payload["overall"]}]
    rows.extend({"scope": "per_mesh", **item} for item in payload["per_mesh"])
    _write_csv(path, fields, rows)


def _write_magnitude_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fields = ["scope", "sample_id", "vertex_count", "stable_ratio_vertex_count"]
    for prefix in ("target", "prediction", "error"):
        fields.extend(f"{prefix}_{name}" for name in ("mean", "std", "min", "median", "p75", "p90", "p95", "p99", "max"))
    fields.extend(("magnitude_ratio_global", "mean_vertex_ratio", "median_vertex_ratio"))
    rows = [_flatten_magnitude("overall", "overall", payload["overall"])]
    rows.extend(_flatten_magnitude("per_mesh", item["sample_id"], item) for item in payload["per_mesh"])
    _write_csv(path, fields, rows)


def _flatten_magnitude(scope: str, sample_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "scope": scope,
        "sample_id": sample_id,
        "vertex_count": item["vertex_count"],
        "stable_ratio_vertex_count": item["stable_ratio_vertex_count"],
        "magnitude_ratio_global": item["magnitude_ratio_global"],
        "mean_vertex_ratio": item["mean_vertex_ratio"],
        "median_vertex_ratio": item["median_vertex_ratio"],
    }
    for source, prefix in (
        ("target_magnitude", "target"),
        ("prediction_magnitude", "prediction"),
        ("error_magnitude", "error"),
    ):
        row.update({f"{prefix}_{name}": value for name, value in item[source].items()})
    return row


def _write_error_bins_csv(path: Path, payload: Mapping[str, Any]) -> None:
    metric_fields = [
        "vertex_count",
        "fraction_of_vertices",
        "mean_target_magnitude",
        "mean_prediction_magnitude",
        "mean_absolute_vector_error",
        "mean_squared_error",
        "training_loss",
        "cosine_similarity",
        "cosine_vertex_count",
    ]
    fields = ["scope", "sample_id", "bin", *metric_fields]
    rows = [
        {"scope": "overall", "sample_id": "overall", "bin": name, **values}
        for name, values in payload["overall"].items()
    ]
    for item in payload["per_mesh"]:
        rows.extend(
            {
                "scope": "per_mesh",
                "sample_id": item["sample_id"],
                "bin": name,
                **values,
            }
            for name, values in item["bins"].items()
        )
    _write_csv(path, fields, rows)


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _amp_settings(
    config: Mapping[str, Any], device: torch.device
) -> tuple[bool, torch.dtype]:
    raw = config.get("training", {}).get("amp", {})
    enabled = bool(raw.get("enabled", False)) and device.type == "cuda"
    dtype_name = str(raw.get("dtype", "float16"))
    if dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("AMP dtype must be float16 or bfloat16.")
    return enabled, torch.float16 if dtype_name == "float16" else torch.bfloat16


def _loss_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    return {
        "loss_type": str(training.get("loss", "huber")),
        "huber_delta": float(training.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training.get("charbonnier_epsilon", 1e-3)),
        "target_magnitude_weight_lambda": float(
            training.get("target_magnitude_weight_lambda", 0.0)
        ),
    }


def _relative_improvement(baseline: float, model: float) -> float:
    return (baseline - model) / baseline if baseline > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.8g}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_finite_json(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite JSON value at {path}: {value}")

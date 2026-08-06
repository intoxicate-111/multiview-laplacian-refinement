from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .diagnostics import _amp_settings, _loss_kwargs
from .evaluation import reconstruct_and_evaluate
from .losses import weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .target_scaling import denormalize_laplacian_by_edge_scale
from .trainer import load_checkpoint


IMAGE_CONDITIONS = (
    "original_rgb",
    "zero_rgb",
    "shuffled_images",
    "cross_object_rgb",
    "shuffled_view_order",
)


def run_image_ablation(
    run_dir: str | Path,
    coarse_manifest: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    seed: int = 7,
    overwrite: bool = False,
    skip_reconstruction: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint while changing only its multi-view image input."""

    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir or run_dir / "image_ablation").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Image-ablation directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(run_dir / "config.json")
    checkpoint = run_dir / "best.pt"
    gt_manifest = run_dir / "dataset_manifest.json"
    if not checkpoint.is_file() or not gt_manifest.is_file():
        raise FileNotFoundError("Run directory requires best.pt and dataset_manifest.json.")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(config, None, False).to(resolved_device)
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=resolved_device)
    model.eval()

    datasets = {
        "gt_query_validation": PreparedMeshDataset.from_manifest(gt_manifest, "validation"),
        "expanded_query_validation": PreparedMeshDataset.from_manifest(
            Path(coarse_manifest).resolve(), "validation"
        ),
    }
    query_config = copy.deepcopy(config)
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True
    loss_kwargs = _loss_kwargs(config)
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)

    query_results: dict[str, Any] = {}
    for query_name, dataset in datasets.items():
        query_dir = output_dir / query_name
        query_dir.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating {query_name} ({len(dataset)} meshes)...", flush=True)
        records = _predict_conditions(
            model,
            dataset,
            query_config,
            resolved_device,
            amp_enabled,
            amp_dtype,
            loss_kwargs,
            seed,
            query_dir / "predictions",
        )
        metrics = summarize_image_ablation(records, loss_kwargs)
        _write_json(query_dir / "prediction_metrics.json", metrics)
        _write_condition_csv(query_dir / "prediction_metrics.csv", metrics)
        _plot_condition_metrics(query_dir / "prediction_metrics.png", metrics)
        if skip_reconstruction:
            reconstruction = {"skipped": True}
        else:
            reconstruction = _run_reconstruction_ablation(
                dataset, records, config, query_dir / "reconstruction"
            )
        _write_json(query_dir / "reconstruction_summary.json", reconstruction)
        query_results[query_name] = {
            "prediction": metrics,
            "reconstruction": reconstruction,
        }

    modalities = _optional_modality_report(datasets["gt_query_validation"])
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(
            checkpoint_payload.get("epoch", checkpoint_payload.get("step", -1))
        ),
        "seed": seed,
        "conditions": list(IMAGE_CONDITIONS),
        "fixed_controls": [
            "checkpoint",
            "query positions",
            "graph topology",
            "camera matrices except in the consistent view-order control",
            "target and confidence",
        ],
        "optional_modalities": modalities,
        "expanded_query_target_note": (
            "Uses the existing prepared closest-surface supervision from the coarse/expanded "
            "manifest; this ablation does not interpolate GT raw Laplacians."
        ),
        "queries": query_results,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


@torch.no_grad()
def _predict_conditions(
    model: torch.nn.Module,
    dataset: PreparedMeshDataset,
    config: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    loss_kwargs: Mapping[str, Any],
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        prepared = _prepare_item_for_use(
            _prepare_object_static(static, config),
            config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        donor_static = dataset.load_static((index + 1) % len(dataset))
        donor = _prepare_item_for_use(
            _prepare_object_static(donor_static, config),
            config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        base = dict(prepared.sample)
        base["query_positions"] = base["vertices"]
        base["query_is_exact"] = torch.ones(
            base["vertices"].shape[0], dtype=torch.bool, device=device
        )
        target = prepared.training_target.float()
        confidence = base["target_confidence"].float()
        valid = base["valid_scale_mask"].bool() & (confidence > 0)
        permutation = torch.randperm(
            int(base["images"].shape[0]),
            generator=torch.Generator().manual_seed(seed + index * 104729),
        ).to(device)
        predictions: dict[str, np.ndarray] = {}
        losses: dict[str, float] = {}
        features: dict[str, float] = {}
        for condition in IMAGE_CONDITIONS:
            sample = _condition_sample(base, donor.sample["images"], permutation, condition)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(sample)
            prediction = output.predicted_laplacian.float()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Non-finite prediction for {condition}.")
            losses[condition] = float(
                weighted_robust_laplacian_loss(
                    prediction, target, confidence, **loss_kwargs
                ).item()
            )
            predictions[condition] = prediction.detach().cpu().numpy()
            features[condition] = float(
                torch.linalg.vector_norm(
                    output.aggregated_image_features.float(), dim=-1
                ).mean().item()
            )
        zero_predictor_loss = float(
            weighted_robust_laplacian_loss(
                torch.zeros_like(target), target, confidence, **loss_kwargs
            ).item()
        )
        sample_id = str(base["sample_id"])
        prediction_path = output_dir / f"{_safe_name(sample_id)}.npz"
        np.savez_compressed(
            prediction_path,
            target=target.detach().cpu().numpy(),
            confidence=confidence.detach().cpu().numpy(),
            valid_mask=valid.detach().cpu().numpy(),
            **predictions,
        )
        record = {
            "sample_id": sample_id,
            "prediction_path": str(prediction_path),
            "vertex_count": int(target.shape[0]),
            "valid_vertex_count": int(valid.sum().item()),
            "donor_sample_id": str(donor.sample["sample_id"]),
            "view_permutation": permutation.detach().cpu().tolist(),
            "losses": losses,
            "zero_predictor_loss": zero_predictor_loss,
            "mean_aggregated_image_feature_norm": features,
        }
        records.append(record)
        print(
            f"  {sample_id}: original={losses['original_rgb']:.8f} "
            f"zero={losses['zero_rgb']:.8f} shuffle={losses['shuffled_images']:.8f}",
            flush=True,
        )
        del prepared, donor, base, target, confidence
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def _condition_sample(
    base: Mapping[str, Any],
    donor_images: torch.Tensor,
    permutation: torch.Tensor,
    condition: str,
) -> dict[str, Any]:
    sample = dict(base)
    if condition == "original_rgb":
        return sample
    if condition == "zero_rgb":
        sample["images"] = torch.zeros_like(base["images"])
        return sample
    if condition == "shuffled_images":
        sample["images"] = base["images"].index_select(0, permutation)
        return sample
    if condition == "cross_object_rgb":
        if tuple(donor_images.shape) != tuple(base["images"].shape):
            raise ValueError("Cross-object RGB requires matching view count and image shape.")
        sample["images"] = donor_images
        return sample
    if condition == "shuffled_view_order":
        for name in ("images", "intrinsics", "extrinsics", "visibility"):
            value = sample.get(name)
            if isinstance(value, torch.Tensor):
                sample[name] = value.index_select(0, permutation)
        return sample
    raise ValueError(f"Unknown image condition: {condition}")


def summarize_image_ablation(
    records: Sequence[Mapping[str, Any]], loss_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    arrays = [_load_npz(Path(str(record["prediction_path"]))) for record in records]
    valid_targets = [a["target"][a["valid_mask"].astype(bool)] for a in arrays]
    target = np.concatenate(valid_targets, axis=0)
    magnitude = np.linalg.norm(target, axis=1)
    thresholds = {
        "p50": float(np.quantile(magnitude, 0.50)),
        "p90": float(np.quantile(magnitude, 0.90)),
        "p95": float(np.quantile(magnitude, 0.95)),
        "p99": float(np.quantile(magnitude, 0.99)),
    }
    original = np.concatenate(
        [a["original_rgb"][a["valid_mask"].astype(bool)] for a in arrays], axis=0
    )
    zero_loss = float(np.mean([float(r["zero_predictor_loss"]) for r in records]))
    conditions: dict[str, Any] = {}
    for condition in IMAGE_CONDITIONS:
        prediction = np.concatenate(
            [a[condition][a["valid_mask"].astype(bool)] for a in arrays], axis=0
        )
        pred_mag = np.linalg.norm(prediction, axis=1)
        difference = np.linalg.norm(prediction - original, axis=1)
        condition_loss = float(np.mean([float(r["losses"][condition]) for r in records]))
        conditions[condition] = {
            "validation_loss": condition_loss,
            "zero_predictor_loss": zero_loss,
            "relative_improvement_vs_zero_predictor": _relative_improvement(
                zero_loss, condition_loss
            ),
            "mean_target_magnitude": float(magnitude.mean()),
            "mean_prediction_magnitude": float(pred_mag.mean()),
            "mean_prediction_to_target_magnitude_ratio": _safe_div(
                float(pred_mag.mean()), float(magnitude.mean())
            ),
            "mean_prediction_change_vs_original": float(difference.mean()),
            "relative_prediction_change_vs_original": _safe_div(
                float(difference.mean()), float(np.linalg.norm(original, axis=1).mean())
            ),
            "magnitude_bins": _magnitude_bins(
                target, prediction, thresholds, loss_kwargs
            ),
            "per_mesh_loss": [
                {
                    "sample_id": str(record["sample_id"]),
                    "loss": float(record["losses"][condition]),
                }
                for record in records
            ],
        }
    return {
        "loss_reduction": "mesh mean, matching validation training semantics",
        "zero_predictor_is_not_zero_image": True,
        "percentile_thresholds": thresholds,
        "conditions": conditions,
        "view_order_invariance_max_abs_prediction_difference": float(
            max(
                np.max(np.abs(a["shuffled_view_order"] - a["original_rgb"]))
                for a in arrays
            )
        ),
        "records": [dict(record) for record in records],
    }


def _magnitude_bins(
    target: np.ndarray,
    prediction: np.ndarray,
    thresholds: Mapping[str, float],
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    magnitude = np.linalg.norm(target, axis=1)
    masks = {
        "low_0_p50": magnitude <= thresholds["p50"],
        "medium_p50_p90": (magnitude > thresholds["p50"]) & (magnitude < thresholds["p90"]),
        "high_top10": magnitude >= thresholds["p90"],
        "top5": magnitude >= thresholds["p95"],
        "top1": magnitude >= thresholds["p99"],
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        t = target[mask]
        p = prediction[mask]
        t_mag = np.linalg.norm(t, axis=1)
        p_mag = np.linalg.norm(p, axis=1)
        cosine = np.sum(t * p, axis=1) / np.maximum(t_mag * p_mag, 1e-12)
        confidence = torch.ones(len(t), dtype=torch.float32)
        loss = weighted_robust_laplacian_loss(
            torch.from_numpy(p), torch.from_numpy(t), confidence, **loss_kwargs
        )
        result[name] = {
            "vertex_count": int(mask.sum()),
            "mean_target_magnitude": float(t_mag.mean()),
            "mean_prediction_magnitude": float(p_mag.mean()),
            "mean_vector_error": float(np.linalg.norm(p - t, axis=1).mean()),
            "cosine_similarity": float(cosine.mean()),
            "training_loss": float(loss.item()),
        }
    return result


def _run_reconstruction_ablation(
    dataset: PreparedMeshDataset,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    reconstruction_config = dict(config.get("reconstruction", {})) or {
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
    per_condition: dict[str, list[dict[str, Any]]] = {
        name: [] for name in IMAGE_CONDITIONS
    }
    for index, record in enumerate(records):
        static = dataset.load_static(index)
        arrays = _load_npz(Path(str(record["prediction_path"])))
        for condition in IMAGE_CONDITIONS:
            if condition == "shuffled_view_order":
                # Mean view aggregation is permutation invariant. Avoid an identical solver run.
                source = output_dir / _safe_name(str(record["sample_id"])) / "original_rgb"
                destination = (
                    output_dir
                    / _safe_name(str(record["sample_id"]))
                    / "shuffled_view_order"
                )
                destination.mkdir(parents=True, exist_ok=True)
                metrics = _read_json(source / "metrics.json")
                metrics["reused_from"] = "original_rgb"
                _write_json(destination / "metrics.json", metrics)
                per_condition[condition].append(metrics)
                continue
            normalized_prediction = torch.from_numpy(arrays[condition])
            raw_prediction = denormalize_laplacian_by_edge_scale(
                normalized_prediction, static["local_edge_length"]
            )
            condition_dir = (
                output_dir / _safe_name(str(record["sample_id"])) / condition
            )
            print(f"  reconstruct {record['sample_id']} / {condition}", flush=True)
            metrics = reconstruct_and_evaluate(
                static,
                raw_prediction,
                condition_dir,
                reconstruction_config,
                normalized_prediction=normalized_prediction,
                edge_scale_epsilon=float(
                    config.get("target_scaling", {}).get("epsilon", 1e-12)
                ),
            )
            metrics["sample_id"] = str(record["sample_id"])
            metrics["condition"] = condition
            metrics = _sanitize(metrics)
            _write_json(condition_dir / "metrics.json", metrics)
            per_condition[condition].append(metrics)
    overall: dict[str, Any] = {}
    for condition, rows in per_condition.items():
        predicted = [row["geometry"]["predicted"] for row in rows]
        overall[condition] = {
            "mean_chamfer": _mean_optional([row.get("chamfer") for row in predicted]),
            "mean_point_to_surface": _mean_optional(
                [row.get("point_to_surface_mean") for row in predicted]
            ),
            "mean_normal_consistency": _mean_optional(
                [row.get("normal_consistency") for row in predicted]
            ),
            "mean_target_position_rmse": _mean_optional(
                [row.get("target_position_rmse") for row in predicted]
            ),
            "improves_over_initial_count": int(
                sum(bool(row["predicted_improves_over_coarse"]) for row in rows)
            ),
        }
    return {
        "skipped": False,
        "config": reconstruction_config,
        "overall": overall,
        "per_condition": per_condition,
        "visualization": "Each condition directory contains OBJ meshes and error arrays.",
    }


def _optional_modality_report(dataset: PreparedMeshDataset) -> dict[str, Any]:
    sample = dataset.load_static(0)
    root = Path(str(sample.get("_dataset_root", dataset.records[0].dataset_root)))
    image_paths = [str(value).lower() for value in sample.get("image_paths", [])]
    normal = any("normal" in value for value in image_paths)
    relit = any(any(token in value for token in ("relight", "uniform_light")) for value in image_paths)
    return {
        "normal_maps": {"available": normal, "reason": None if normal else "not listed in prepared samples"},
        "relit_or_uniform_lighting_rgb": {
            "available": relit,
            "reason": None if relit else "not listed in prepared samples",
        },
        "inspected_dataset_root": str(root),
    }


def _write_condition_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "validation_loss",
                "zero_predictor_loss",
                "relative_improvement_vs_zero_predictor",
                "mean_prediction_to_target_magnitude_ratio",
                "high_top10_cosine_similarity",
                "relative_prediction_change_vs_original",
            ),
        )
        writer.writeheader()
        for condition, values in metrics["conditions"].items():
            writer.writerow(
                {
                    "condition": condition,
                    "validation_loss": values["validation_loss"],
                    "zero_predictor_loss": values["zero_predictor_loss"],
                    "relative_improvement_vs_zero_predictor": values[
                        "relative_improvement_vs_zero_predictor"
                    ],
                    "mean_prediction_to_target_magnitude_ratio": values[
                        "mean_prediction_to_target_magnitude_ratio"
                    ],
                    "high_top10_cosine_similarity": values["magnitude_bins"][
                        "high_top10"
                    ]["cosine_similarity"],
                    "relative_prediction_change_vs_original": values[
                        "relative_prediction_change_vs_original"
                    ],
                }
            )


def _plot_condition_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    names = list(metrics["conditions"])
    losses = [metrics["conditions"][name]["validation_loss"] for name in names]
    baseline = metrics["conditions"][names[0]]["zero_predictor_loss"]
    ratios = [
        metrics["conditions"][name]["mean_prediction_to_target_magnitude_ratio"]
        for name in names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, losses)
    axes[0].axhline(baseline, color="black", linestyle="--", label="zero predictor")
    axes[0].set_ylabel("validation loss")
    axes[0].legend()
    axes[1].bar(names, ratios)
    axes[1].set_ylabel("mean |pred| / mean |GT|")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Image ablation",
        "",
        f"Checkpoint: `{summary['checkpoint']}` (epoch {summary['checkpoint_epoch']})",
        "",
    ]
    for query_name, result in summary["queries"].items():
        lines.extend((f"## {query_name}", "", "| condition | loss | vs zero predictor | |pred|/|GT| | top-10% cosine |", "|---|---:|---:|---:|---:|"))
        for condition, values in result["prediction"]["conditions"].items():
            lines.append(
                f"| {condition} | {values['validation_loss']:.8g} | "
                f"{values['relative_improvement_vs_zero_predictor']:.3%} | "
                f"{values['mean_prediction_to_target_magnitude_ratio']:.3%} | "
                f"{values['magnitude_bins']['high_top10']['cosine_similarity']:.4f} |"
            )
        lines.append("")
    lines.extend(
        (
            "## Optional modalities",
            "",
            f"Normal maps available: {summary['optional_modalities']['normal_maps']['available']}.",
            f"Relit/uniform-light RGB available: {summary['optional_modalities']['relit_or_uniform_lighting_rgb']['available']}.",
            "",
            summary["expanded_query_target_note"],
            "",
        )
    )
    return "\n".join(lines)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name] for name in payload.files}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _relative_improvement(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)

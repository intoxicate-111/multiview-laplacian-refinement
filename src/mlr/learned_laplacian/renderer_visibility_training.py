from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .diagnostics import _amp_settings, _loss_kwargs
from .image_ablation import _predict_conditions, summarize_image_ablation
from .multi_dataset import PreparedMeshDataset, PreparedMeshRecord, validate_disjoint_splits
from .multi_trainer import train_multi_object
from .projection import project_vertices


SHORT_TRAINING_CONDITIONS = ("frustum_only", "backface_and_occlusion")


def run_renderer_visibility_short_training(
    manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    mesh_counts: Sequence[int] = (1, 4, 16),
    conditions: Sequence[str] = SHORT_TRAINING_CONDITIONS,
    optimizer_steps: int = 100,
    device: str = "cuda",
    seed: int = 7,
) -> dict[str, Any]:
    """Run paired, fixed-budget visibility training diagnostics.

    Every run is initialized from the same seed and uses the same Adam settings,
    targets, selected views, and optimizer-step budget.  Only the selected train
    mesh prefix and renderer visibility condition differ.
    """

    manifest = Path(manifest).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if optimizer_steps < 1:
        raise ValueError("optimizer_steps must be positive.")
    counts = tuple(int(value) for value in mesh_counts)
    if not counts or min(counts) < 1:
        raise ValueError("mesh_counts must contain positive integers.")
    allowed = {"frustum_only", "backface_and_occlusion"}
    unknown = set(conditions) - allowed
    if unknown:
        raise ValueError(f"Unsupported short-training visibility conditions: {sorted(unknown)}")

    source_config = _read_json(config_path)
    source_train = PreparedMeshDataset.from_manifest(manifest, "train")
    validation = PreparedMeshDataset.from_manifest(manifest, "validation")
    validate_disjoint_splits(source_train, validation)
    maximum = max(counts)
    if maximum > len(source_train):
        raise ValueError(
            f"Requested {maximum} train meshes but manifest contains {len(source_train)}."
        )

    selected_ids = list(source_train.sample_ids[:maximum])
    results: dict[str, Any] = {}
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    for mesh_count in counts:
        train = PreparedMeshDataset(source_train.records[:mesh_count])
        validate_disjoint_splits(train, validation)
        for condition in conditions:
            run_name = f"mesh_count_{mesh_count:02d}__{condition}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            config = build_short_training_config(
                source_config,
                condition=condition,
                mesh_count=mesh_count,
                validation_mesh_count=len(validation),
                optimizer_steps=optimizer_steps,
                seed=seed,
            )
            _write_run_inputs(run_dir, manifest, train, validation, config)
            print(
                f"Short training: meshes={mesh_count} visibility={condition} "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
            result = train_multi_object(
                train,
                validation,
                config,
                output_dir=run_dir,
                device_override=device,
                progress=True,
            )
            image_metrics = _run_matching_image_ablation(
                result.model,
                validation,
                config,
                resolved_device,
                seed,
                run_dir / "image_ablation",
            )
            visibility_stats = _zero_view_statistics(validation, config)
            metrics = {
                "mesh_count": mesh_count,
                "visibility_condition": condition,
                "optimizer_steps": result.optimizer_steps,
                "training_loss": result.final_train_loss,
                "validation_loss": result.final_validation_loss,
                "best_validation_loss": result.best_selection_loss,
                "runtime_seconds": result.runtime_seconds,
                "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                "peak_cpu_memory_mb": result.peak_cpu_memory_mb,
                "zero_visible_vertex_ratio": visibility_stats[
                    "zero_visible_vertex_ratio"
                ],
                "mean_visible_views_per_vertex": visibility_stats[
                    "mean_visible_views_per_vertex"
                ],
                "image_ablation": image_metrics,
            }
            _write_json(run_dir / "short_training_metrics.json", metrics)
            results[run_name] = metrics
            del result
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "source_manifest": str(manifest),
        "source_config": str(config_path),
        "seed": seed,
        "mesh_counts": list(counts),
        "conditions": list(conditions),
        "optimizer_steps_per_run": optimizer_steps,
        "controlled_fields": [
            "model initialization seed",
            "Adam optimizer and learning rate",
            "optimizer-step count",
            "selected view count and view order",
            "query augmentation, graph, and target",
        ],
        "changed_fields": ["train mesh prefix", "renderer visibility condition"],
        "results": results,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", results)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def build_short_training_config(
    source: Mapping[str, Any],
    *,
    condition: str,
    mesh_count: int,
    validation_mesh_count: int,
    optimizer_steps: int,
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(source))
    config["seed"] = int(seed)
    config["renderer_visibility"] = {
        "condition": condition,
        "source": "precomputed_depth_tested_face_id_buffer",
        "neighborhood_radius": 1,
        "front_face_winding": "ccw",
        "depth_image_used": False,
    }
    config["dataset"] = {
        "expected_split_counts": {
            "train": int(mesh_count),
            "validation": int(validation_mesh_count),
            "test": 0,
        }
    }
    training = config.setdefault("training", {})
    training["loss"] = "huber"
    training["huber_delta"] = 0.01
    training["target_magnitude_weight_lambda"] = 0.0
    # Scheduler state would otherwise depend on how many epochs are required to
    # reach the same step budget at each mesh count.
    training["lr_scheduler"] = {"type": "none"}
    multi = config.setdefault("multi_object_training", {})
    accumulation = int(multi.get("gradient_accumulation_meshes", 4))
    steps_per_epoch = max(1, math.ceil(int(mesh_count) / accumulation))
    multi["epochs"] = int(optimizer_steps)
    multi["max_optimizer_steps"] = int(optimizer_steps)
    multi["validation_every_epochs"] = max(1, math.ceil(10 / steps_per_epoch))
    multi["checkpoint_every_epochs"] = 0
    multi["early_stopping"] = {"enabled": False}
    return config


def _run_matching_image_ablation(
    model: torch.nn.Module,
    validation: PreparedMeshDataset,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_config = copy.deepcopy(dict(config))
    exact_config.setdefault("query_training", {})["enabled"] = False
    amp_enabled, amp_dtype = _amp_settings(exact_config, device)
    records = _predict_conditions(
        model,
        validation,
        exact_config,
        device,
        amp_enabled,
        amp_dtype,
        _loss_kwargs(exact_config),
        seed,
        output_dir / "arrays",
    )
    metrics = summarize_image_ablation(records, _loss_kwargs(exact_config))
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def _zero_view_statistics(
    dataset: PreparedMeshDataset, config: Mapping[str, Any]
) -> dict[str, float]:
    condition = str(config["renderer_visibility"]["condition"])
    field = {
        "backface_and_occlusion": "visibility_backface_and_occlusion"
    }.get(condition)
    zero_ratios = []
    means = []
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        projection = project_vertices(
            sample["vertices"].float(),
            sample["intrinsics"].float(),
            sample["extrinsics"].float(),
            (int(sample["prepared_image_size"]), int(sample["prepared_image_size"])),
        )
        valid = projection.frustum_valid
        if field is not None:
            valid = valid & sample[field].bool()
        counts = valid.sum(dim=0).float()
        zero_ratios.append(float((counts == 0).float().mean().item()))
        means.append(float(counts.mean().item()))
    return {
        "zero_visible_vertex_ratio": float(np.mean(zero_ratios)),
        "mean_visible_views_per_vertex": float(np.mean(means)),
    }


def _write_run_inputs(
    run_dir: Path,
    source_manifest: Path,
    train: PreparedMeshDataset,
    validation: PreparedMeshDataset,
    config: Mapping[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    for dataset in (train, validation):
        for record in dataset.records:
            records.append(
                {
                    "path": str(record.path),
                    "split": record.split,
                    "sample_id": record.sample_id,
                }
            )
    manifest = {"samples": records}
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "dataset_manifest.json", manifest)
    _write_json(
        run_dir / "run_config.json",
        {
            "experiment_config": config,
            "manifest_path": "dataset_manifest.json",
            "source_manifest": str(source_manifest),
        },
    )


def _write_summary_csv(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "mesh_count",
        "visibility_condition",
        "optimizer_steps",
        "training_loss",
        "validation_loss",
        "best_validation_loss",
        "original_loss",
        "zero_rgb_loss",
        "shuffled_images_loss",
        "cross_object_rgb_loss",
        "original_vs_zero_predictor",
        "prediction_target_magnitude_ratio",
        "high_10_cosine",
        "zero_visible_vertex_ratio",
        "runtime_seconds",
        "peak_gpu_memory_mb",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metrics in results.values():
            conditions = metrics["image_ablation"]["conditions"]
            original = conditions["original_rgb"]
            writer.writerow(
                {
                    **{key: metrics.get(key) for key in fields},
                    "original_loss": original["validation_loss"],
                    "zero_rgb_loss": conditions["zero_rgb"]["validation_loss"],
                    "shuffled_images_loss": conditions["shuffled_images"][
                        "validation_loss"
                    ],
                    "cross_object_rgb_loss": conditions["cross_object_rgb"][
                        "validation_loss"
                    ],
                    "original_vs_zero_predictor": original[
                        "relative_improvement_vs_zero_predictor"
                    ],
                    "prediction_target_magnitude_ratio": original[
                        "mean_prediction_to_target_magnitude_ratio"
                    ],
                    "high_10_cosine": original["magnitude_bins"]["high_top10"][
                        "cosine_similarity"
                    ],
                }
            )


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Renderer visibility short-training comparison",
        "",
        f"All runs use {summary['optimizer_steps_per_run']} optimizer steps and seed "
        f"{summary['seed']}. No depth image is loaded or compared.",
        "",
        "| meshes | visibility | train loss | validation loss | original loss | zero RGB | shuffled RGB | cross RGB | |pred|/|GT| | High-10% cosine | zero-view |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in summary["results"].values():
        conditions = metrics["image_ablation"]["conditions"]
        original = conditions["original_rgb"]
        lines.append(
            f"| {metrics['mesh_count']} | {metrics['visibility_condition']} | "
            f"{metrics['training_loss']:.6g} | {metrics['validation_loss']:.6g} | "
            f"{original['validation_loss']:.6g} | "
            f"{conditions['zero_rgb']['validation_loss']:.6g} | "
            f"{conditions['shuffled_images']['validation_loss']:.6g} | "
            f"{conditions['cross_object_rgb']['validation_loss']:.6g} | "
            f"{original['mean_prediction_to_target_magnitude_ratio']:.3f} | "
            f"{original['magnitude_bins']['high_top10']['cosine_similarity']:.3f} | "
            f"{metrics['zero_visible_vertex_ratio']:.3%} |"
        )
    lines.extend(
        (
            "",
            "Training validation loss uses the configured perturbed GT-query validation. "
            "Image-ablation loss uses exact GT-query vertices and keeps graph, target, cameras, "
            "and visibility fixed.",
            "",
        )
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value

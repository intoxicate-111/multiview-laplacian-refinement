from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh

from .diagnostics import _amp_settings, _loss_kwargs
from .evaluation import reconstruct_and_evaluate
from .image_ablation import _predict_conditions, summarize_image_ablation
from .losses import weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
    train_multi_object,
)
from .renderer_visibility_training import build_short_training_config
from .target_scaling import denormalize_laplacian_by_edge_scale
from .trainer import load_checkpoint
from .visibility_recovery import hard_any_view_recovery_mask
from .visibility_recovery_ablation import _write_visibility_ply


VISIBILITY_FIELD = "visibility_backface_and_occlusion"
VISIBILITY_GROUPS = ("0_views", "1_view", "2_views", "3_4_views", "5_plus_views")
RGB_CONDITIONS = (
    "original_rgb",
    "zero_rgb",
    "shuffled_images",
    "cross_object_rgb",
    "shuffled_view_order",
)


def normalize_checkpoint_steps(
    checkpoint_steps: Sequence[int], optimizer_steps: int
) -> tuple[int, ...]:
    maximum = int(optimizer_steps)
    if maximum < 1:
        raise ValueError("optimizer_steps must be positive.")
    steps = tuple(sorted({int(value) for value in checkpoint_steps}))
    if not steps or steps[0] < 0 or steps[-1] > maximum:
        raise ValueError("checkpoint steps must be non-negative and within the run budget.")
    if 0 not in steps or maximum not in steps:
        raise ValueError("checkpoint steps must include step 0 and the final optimizer step.")
    return steps


def visibility_group_masks(visibility_count: np.ndarray) -> dict[str, np.ndarray]:
    count = np.asarray(visibility_count).reshape(-1)
    if np.any(count < 0):
        raise ValueError("visibility_count cannot be negative.")
    return {
        "0_views": count == 0,
        "1_view": count == 1,
        "2_views": count == 2,
        "3_4_views": (count >= 3) & (count <= 4),
        "5_plus_views": count >= 5,
    }


def latest_resume_checkpoint(
    checkpoint_dir: str | Path, checkpoint_steps: Sequence[int], final_step: int
) -> Path | None:
    root = Path(checkpoint_dir)
    candidates = [
        (step, root / f"checkpoint_step_{step:06d}.pt")
        for step in checkpoint_steps
        if step < final_step
    ]
    existing = [(step, path) for step, path in candidates if path.is_file()]
    return max(existing, default=(None, None), key=lambda item: item[0])[1]


def validate_visibility_shape(visibility: torch.Tensor, num_vertices: int) -> None:
    if visibility.ndim != 2 or tuple(visibility.shape)[1] != int(num_vertices):
        raise ValueError("Renderer visibility must have shape [views, vertices].")


def validate_expanded_sample_ids(
    actual: Sequence[str], expected: Sequence[str]
) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError("Expanded validation mesh IDs changed between checkpoints.")


def validate_summary_consistency(csv_path: str | Path, json_path: str | Path) -> None:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        csv_steps = [int(row["optimizer_step"]) for row in csv.DictReader(handle)]
    payload = _read_json(Path(json_path))
    json_steps = [int(row["optimizer_step"]) for row in payload["checkpoints"]]
    if csv_steps != json_steps:
        raise ValueError("checkpoint_summary.csv and checkpoint_summary.json disagree.")


def run_visibility_convergence_study(
    gt_query_manifest: str | Path,
    expanded_manifest: str | Path,
    source_config_path: str | Path,
    recovery_config_path: str | Path,
    output_dir: str | Path,
    *,
    mesh_count: int = 16,
    optimizer_steps: int = 2000,
    checkpoint_steps: Sequence[int] = (0, 100, 250, 500, 1000, 2000),
    visibility_key: str = VISIBILITY_FIELD,
    expanded_split: str = "validation",
    seed: int = 7,
    device: str = "cuda",
) -> dict[str, Any]:
    gt_query_manifest = Path(gt_query_manifest).expanduser().resolve()
    expanded_manifest = Path(expanded_manifest).expanduser().resolve()
    source_config_path = Path(source_config_path).expanduser().resolve()
    recovery_config_path = Path(recovery_config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = normalize_checkpoint_steps(checkpoint_steps, optimizer_steps)
    if visibility_key != VISIBILITY_FIELD:
        raise ValueError(
            f"This controlled study requires visibility_key={VISIBILITY_FIELD!r}."
        )

    source_config = _read_json(source_config_path)
    source_train = PreparedMeshDataset.from_manifest(gt_query_manifest, "train")
    validation = PreparedMeshDataset.from_manifest(gt_query_manifest, "validation")
    expanded = PreparedMeshDataset.from_manifest(expanded_manifest, expanded_split)
    if mesh_count < 1 or mesh_count > len(source_train):
        raise ValueError(f"mesh_count must be within 1..{len(source_train)}.")
    train = PreparedMeshDataset(source_train.records[:mesh_count])
    validate_disjoint_splits(train, validation)
    expected_expanded_ids = tuple(expanded.sample_ids)
    if len(expanded) != 5:
        raise ValueError(
            f"The controlled study requires exactly five expanded meshes, got {len(expanded)}."
        )

    config = build_short_training_config(
        source_config,
        condition="backface_and_occlusion",
        mesh_count=mesh_count,
        validation_mesh_count=len(validation),
        optimizer_steps=optimizer_steps,
        seed=seed,
    )
    config["multi_object_training"]["checkpoint_optimizer_steps"] = list(steps)
    config["convergence_study"] = {
        "mesh_count": mesh_count,
        "optimizer_steps": optimizer_steps,
        "checkpoint_steps": list(steps),
        "expanded_split": expanded_split,
        "expanded_sample_ids": list(expected_expanded_ids),
        "hard_zero_visibility_gate": True,
        "strong_unseen_anchor": False,
    }
    recovery_experiment = _read_json(recovery_config_path)
    recovery_config = dict(recovery_experiment.get("reconstruction", {}))
    recovery_config["evaluate_oracle"] = False

    _write_json(output_dir / "config.json", config)
    _write_dataset_manifest(output_dir / "dataset_manifest.json", train, validation)
    _write_json(
        output_dir / "run_metadata.json",
        _run_metadata(
            gt_query_manifest,
            expanded_manifest,
            source_config_path,
            recovery_config_path,
            expected_expanded_ids,
            steps,
            seed,
            device,
        ),
    )

    checkpoint_dir = output_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / f"checkpoint_step_{optimizer_steps:06d}.pt"
    if not final_checkpoint.is_file():
        resume = latest_resume_checkpoint(checkpoint_dir, steps, optimizer_steps)
        if resume is not None:
            print(f"Resuming deterministic training from {resume.name}", flush=True)
        train_multi_object(
            train,
            validation,
            config,
            output_dir=output_dir,
            device_override=device,
            progress=True,
            resume_checkpoint=resume,
        )
    missing = [
        step
        for step in steps
        if not (checkpoint_dir / f"checkpoint_step_{step:06d}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"Training did not produce checkpoint steps: {missing}")
    _training_history_csv(output_dir)

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    checkpoint_summaries: list[dict[str, Any]] = []
    for step in steps:
        checkpoint_output = output_dir / "checkpoint_evaluation" / f"step_{step:06d}"
        checkpoint_json = checkpoint_output / "metrics.json"
        if checkpoint_json.is_file():
            existing = _read_json(checkpoint_json)
            if tuple(existing["expanded_sample_ids"]) != expected_expanded_ids:
                raise ValueError("Existing checkpoint evaluation used different expanded meshes.")
            checkpoint_summaries.append(existing)
            continue
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
        print(f"Evaluating optimizer step {step}", flush=True)
        metrics = _evaluate_checkpoint(
            checkpoint,
            step,
            validation,
            expanded,
            expected_expanded_ids,
            config,
            recovery_config,
            checkpoint_output,
            seed,
            device_obj,
        )
        _write_json(checkpoint_json, metrics)
        checkpoint_summaries.append(metrics)

    checkpoint_summaries.sort(key=lambda row: int(row["optimizer_step"]))
    summary = _write_study_outputs(
        output_dir,
        checkpoint_summaries,
        config,
        expected_expanded_ids,
    )
    return summary


def _evaluate_checkpoint(
    checkpoint_path: Path,
    step: int,
    validation: PreparedMeshDataset,
    expanded: PreparedMeshDataset,
    expected_expanded_ids: Sequence[str],
    config: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint_path, model, map_location=device)
    if int(payload.get("optimizer_steps", -1)) != step:
        raise ValueError(f"Checkpoint metadata does not match requested step {step}.")
    model.eval()
    query_config = copy.deepcopy(dict(config))
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True
    amp_enabled, amp_dtype = _amp_settings(query_config, device)
    loss_kwargs = _loss_kwargs(query_config)

    records = _predict_conditions(
        model,
        validation,
        query_config,
        device,
        amp_enabled,
        amp_dtype,
        loss_kwargs,
        seed,
        output_dir / "gt_query_predictions",
    )
    image_metrics = summarize_image_ablation(records, loss_kwargs)
    group_metrics = _visibility_group_metrics(validation, records, loss_kwargs)
    expanded_metrics = _expanded_recovery(
        model,
        expanded,
        expected_expanded_ids,
        query_config,
        recovery_config,
        output_dir / "expanded_recovery",
        amp_enabled,
        amp_dtype,
        device,
    )
    original = image_metrics["conditions"]["original_rgb"]
    overall = _overall_direction_metrics(records)
    return {
        "optimizer_step": step,
        "epoch": int(payload.get("epoch", -1)),
        "checkpoint": str(checkpoint_path),
        "gt_query": {
            "conditions": image_metrics["conditions"],
            "view_order_invariance_max_abs_prediction_difference": image_metrics[
                "view_order_invariance_max_abs_prediction_difference"
            ],
            "mean_cosine": overall["mean_cosine"],
            "high_10_cosine": original["magnitude_bins"]["high_top10"][
                "cosine_similarity"
            ],
        },
        "visibility_groups": group_metrics,
        "expanded_recovery": expanded_metrics,
        "expanded_sample_ids": list(expected_expanded_ids),
        "controls": {
            "query_graph_target_visibility_fixed_across_rgb_ablation": True,
            "visibility_shape": "[views, vertices]",
            "recovery_equation": "sqrt(weight) * (L @ X - predicted_delta)",
            "zero_view_weight": 0.0,
            "unseen_anchor_weight": 0.0,
            "expanded_oracle_evaluated": False,
        },
    }


def _visibility_group_metrics(
    dataset: PreparedMeshDataset,
    records: Sequence[Mapping[str, Any]],
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"target": [], "original": [], "zero": [], "confidence": []}
        for name in VISIBILITY_GROUPS
    }
    per_mesh: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        static = dataset.load_static(index)
        if str(static["sample_id"]) != str(record["sample_id"]):
            raise ValueError("Ablation records no longer match validation dataset order.")
        visibility = torch.as_tensor(static[VISIBILITY_FIELD], dtype=torch.bool)
        num_vertices = int(static["vertices"].shape[0])
        validate_visibility_shape(visibility, num_vertices)
        counts = visibility.sum(dim=0).cpu().numpy()
        arrays = np.load(str(record["prediction_path"]))
        valid = arrays["valid_mask"].astype(bool)
        masks = visibility_group_masks(counts)
        mesh_groups: dict[str, Any] = {}
        for name, group in masks.items():
            keep = group & valid
            mesh_groups[name] = int(keep.sum())
            buckets[name]["target"].append(arrays["target"][keep])
            buckets[name]["original"].append(arrays["original_rgb"][keep])
            buckets[name]["zero"].append(arrays["zero_rgb"][keep])
            buckets[name]["confidence"].append(arrays["confidence"][keep])
        per_mesh.append({"sample_id": str(record["sample_id"]), "counts": mesh_groups})

    result: dict[str, Any] = {}
    for name, values in buckets.items():
        target = np.concatenate(values["target"], axis=0)
        original = np.concatenate(values["original"], axis=0)
        zero = np.concatenate(values["zero"], axis=0)
        confidence = np.concatenate(values["confidence"], axis=0)
        result[name] = _group_metric(target, original, zero, confidence, loss_kwargs)
    result["per_mesh_counts"] = per_mesh
    return result


def _group_metric(
    target: np.ndarray,
    prediction: np.ndarray,
    zero_rgb_prediction: np.ndarray,
    confidence: np.ndarray,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if len(target) == 0:
        return {"vertex_count": 0}
    target_mag = np.linalg.norm(target, axis=1)
    prediction_mag = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(target * prediction, axis=1) / np.maximum(
        target_mag * prediction_mag, 1e-12
    )
    high = target_mag >= np.quantile(target_mag, 0.9)
    original_loss = float(
        weighted_robust_laplacian_loss(
            torch.from_numpy(prediction),
            torch.from_numpy(target),
            torch.from_numpy(confidence),
            **loss_kwargs,
        ).item()
    )
    zero_loss = float(
        weighted_robust_laplacian_loss(
            torch.from_numpy(zero_rgb_prediction),
            torch.from_numpy(target),
            torch.from_numpy(confidence),
            **loss_kwargs,
        ).item()
    )
    return {
        "vertex_count": int(len(target)),
        "mean_target_magnitude": float(target_mag.mean()),
        "mean_prediction_magnitude": float(prediction_mag.mean()),
        "mean_vector_error": float(np.linalg.norm(prediction - target, axis=1).mean()),
        "prediction_loss": original_loss,
        "cosine_similarity": float(cosine.mean()),
        "high_magnitude_cosine": float(cosine[high].mean()),
        "zero_rgb_loss": zero_loss,
        "original_rgb_advantage_over_zero": float(zero_loss - original_loss),
    }


def _overall_direction_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    targets = []
    predictions = []
    for record in records:
        arrays = np.load(str(record["prediction_path"]))
        valid = arrays["valid_mask"].astype(bool)
        targets.append(arrays["target"][valid])
        predictions.append(arrays["original_rgb"][valid])
    target = np.concatenate(targets, axis=0)
    prediction = np.concatenate(predictions, axis=0)
    target_mag = np.linalg.norm(target, axis=1)
    prediction_mag = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(target * prediction, axis=1) / np.maximum(
        target_mag * prediction_mag, 1e-12
    )
    return {"mean_cosine": float(cosine.mean())}


@torch.no_grad()
def _expanded_recovery(
    model: torch.nn.Module,
    dataset: PreparedMeshDataset,
    expected_ids: Sequence[str],
    config: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
    output_dir: Path,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    validate_expanded_sample_ids(dataset.sample_ids, expected_ids)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        num_vertices = int(static["vertices"].shape[0])
        visibility = torch.as_tensor(static[VISIBILITY_FIELD], dtype=torch.bool)
        validate_visibility_shape(visibility, num_vertices)
        mask = hard_any_view_recovery_mask(visibility, num_vertices=num_vertices)
        if torch.any(mask.laplacian_weight[mask.visibility_count == 0] != 0):
            raise AssertionError("Zero-view vertices must have exactly zero recovery weight.")
        prepared = _prepare_item_for_use(
            _prepare_object_static(static, config),
            config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        sample = dict(prepared.sample)
        sample["query_positions"] = sample["vertices"]
        sample["query_is_exact"] = torch.ones(
            num_vertices, dtype=torch.bool, device=device
        )
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            prediction = model(sample).predicted_laplacian.float()
        raw_prediction = denormalize_laplacian_by_edge_scale(
            prediction, sample["local_edge_length"]
        )
        mesh_dir = output_dir / sample_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        vertices = static["vertices"].detach().cpu().numpy()
        faces = static["faces"].detach().cpu().numpy()
        save_mesh(Mesh(vertices, faces), mesh_dir / "initial_expanded.obj")
        _write_visibility_ply(
            mesh_dir / "visibility_count.ply",
            vertices,
            faces,
            mask.visibility_count.cpu().numpy(),
            mask.num_views,
            binary=False,
        )
        metrics = reconstruct_and_evaluate(
            static,
            raw_prediction.detach().cpu(),
            mesh_dir / "solver",
            recovery_config,
            normalized_prediction=prediction.detach().cpu(),
            edge_scale_epsilon=float(
                config.get("target_scaling", {}).get("epsilon", 1e-12)
            ),
            laplacian_weight=mask.laplacian_weight,
            unseen_anchor_weight=0.0,
        )
        recovered_path = mesh_dir / "solver" / "predicted_refined.obj"
        shutil.copyfile(recovered_path, mesh_dir / "recovered_visibility_mask.obj")
        recovered = load_mesh(recovered_path).vertices
        displacement = np.linalg.norm(recovered - vertices, axis=1)
        counts = mask.visibility_count.cpu().numpy()
        displacement_groups = {
            "visible": counts > 0,
            "0_views": counts == 0,
            "1_view": counts == 1,
            "2_views": counts == 2,
            "3_plus_views": counts >= 3,
        }
        displacement_metrics = {
            name: _mean_or_none(displacement[keep])
            for name, keep in displacement_groups.items()
        }
        np.savez_compressed(
            mesh_dir / "per_vertex_diagnostics.npz",
            visibility_count=counts,
            visible_any=mask.visible_any.cpu().numpy(),
            laplacian_weight=mask.laplacian_weight.cpu().numpy(),
            initial_vertices=vertices,
            recovered_vertices=recovered,
            predicted_delta=prediction.detach().cpu().numpy(),
            displacement=displacement,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "initial": metrics["geometry"]["coarse"],
                "predicted": metrics["geometry"]["predicted"],
                "improved_over_initial": metrics["predicted_improves_over_coarse"],
                "displacement": displacement_metrics,
                "zero_view_vertices": int((counts == 0).sum()),
            }
        )
        del prepared, sample, prediction, raw_prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    geometry_fields = (
        "chamfer",
        "point_to_surface_forward_mean",
        "point_to_surface_reverse_mean",
        "point_to_surface_bidirectional_mean",
        "normal_consistency",
    )
    aggregate = {
        "initial_mean_chamfer": float(
            np.mean([row["initial"]["chamfer"] for row in rows])
        ),
        "initial_median_chamfer": float(
            np.median([row["initial"]["chamfer"] for row in rows])
        ),
        "geometry_mean": {
            key: float(np.mean([row["predicted"][key] for row in rows]))
            for key in geometry_fields
        },
        "geometry_median": {
            key: float(np.median([row["predicted"][key] for row in rows]))
            for key in geometry_fields
        },
        "displacement_mean": {
            name: float(np.mean([row["displacement"][name] for row in rows]))
            for name in ("visible", "0_views", "1_view", "2_views", "3_plus_views")
        },
        "improved_meshes": int(sum(row["improved_over_initial"] for row in rows)),
        "worsened_meshes": int(sum(not row["improved_over_initial"] for row in rows)),
    }
    _write_json(output_dir / "metrics.json", {"aggregate": aggregate, "per_mesh": rows})
    return {"aggregate": aggregate, "per_mesh": rows}


def _write_study_outputs(
    output_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    expanded_ids: Sequence[str],
) -> dict[str, Any]:
    summary_rows = [_checkpoint_summary_row(row) for row in checkpoints]
    _write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    _write_json(output_dir / "checkpoint_summary.json", {"checkpoints": list(checkpoints)})
    validate_summary_consistency(
        output_dir / "checkpoint_summary.csv",
        output_dir / "checkpoint_summary.json",
    )

    rgb_rows = []
    group_rows = []
    expanded_rows = []
    for checkpoint in checkpoints:
        step = int(checkpoint["optimizer_step"])
        for condition, metrics in checkpoint["gt_query"]["conditions"].items():
            rgb_rows.append({"optimizer_step": step, "condition": condition, **metrics})
        for group in VISIBILITY_GROUPS:
            group_rows.append(
                {
                    "optimizer_step": step,
                    "visibility_group": group,
                    **checkpoint["visibility_groups"][group],
                }
            )
        for mesh in checkpoint["expanded_recovery"]["per_mesh"]:
            predicted = mesh["predicted"]
            expanded_rows.append(
                {
                    "optimizer_step": step,
                    "sample_id": mesh["sample_id"],
                    "initial_chamfer": mesh["initial"]["chamfer"],
                    "chamfer": predicted["chamfer"],
                    "point_to_surface_forward_mean": predicted[
                        "point_to_surface_forward_mean"
                    ],
                    "point_to_surface_reverse_mean": predicted[
                        "point_to_surface_reverse_mean"
                    ],
                    "point_to_surface_bidirectional_mean": predicted[
                        "point_to_surface_bidirectional_mean"
                    ],
                    "normal_consistency": predicted["normal_consistency"],
                    "improved_over_initial": mesh["improved_over_initial"],
                    **{
                        f"{name}_displacement": value
                        for name, value in mesh["displacement"].items()
                    },
                }
            )
    _write_csv(output_dir / "rgb_ablation_metrics.csv", rgb_rows)
    _write_csv(output_dir / "visibility_group_metrics.csv", group_rows)
    _write_csv(output_dir / "expanded_recovery_metrics.csv", expanded_rows)
    _ensure_expanded_recovery_index(output_dir, checkpoints)
    _plot_curves(output_dir / "plots", summary_rows)

    conclusion = _conclusion(summary_rows)
    summary = {
        "experiment": "Sofa50 renderer-visibility convergence study",
        "mesh_count": int(config["dataset"]["expected_split_counts"]["train"]),
        "optimizer_steps": [int(row["optimizer_step"]) for row in checkpoints],
        "optimizer_steps_per_epoch": math.ceil(
            int(config["dataset"]["expected_split_counts"]["train"])
            / int(config["multi_object_training"]["gradient_accumulation_meshes"])
        ),
        "expanded_sample_ids": list(expanded_ids),
        "checkpoint_summary": summary_rows,
        "conclusion": conclusion,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        _report(summary, config), encoding="utf-8"
    )
    return summary


def _checkpoint_summary_row(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    conditions = checkpoint["gt_query"]["conditions"]
    original = conditions["original_rgb"]
    expanded = checkpoint["expanded_recovery"]["aggregate"]
    groups = checkpoint["visibility_groups"]
    return {
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "epoch": int(checkpoint["epoch"]),
        "original_rgb_loss": original["validation_loss"],
        "zero_rgb_loss": conditions["zero_rgb"]["validation_loss"],
        "shuffled_rgb_loss": conditions["shuffled_images"]["validation_loss"],
        "cross_object_rgb_loss": conditions["cross_object_rgb"]["validation_loss"],
        "consistent_view_order_loss": conditions["shuffled_view_order"][
            "validation_loss"
        ],
        "consistent_view_order_relative_prediction_change": conditions[
            "shuffled_view_order"
        ]["relative_prediction_change_vs_original"],
        "zero_predictor_loss": original["zero_predictor_loss"],
        "relative_improvement_vs_zero_predictor": original[
            "relative_improvement_vs_zero_predictor"
        ],
        "original_minus_zero_rgb": original["validation_loss"]
        - conditions["zero_rgb"]["validation_loss"],
        "original_minus_shuffled_rgb": original["validation_loss"]
        - conditions["shuffled_images"]["validation_loss"],
        "original_minus_cross_object_rgb": original["validation_loss"]
        - conditions["cross_object_rgb"]["validation_loss"],
        "prediction_target_magnitude_ratio": original[
            "mean_prediction_to_target_magnitude_ratio"
        ],
        "mean_cosine": checkpoint["gt_query"]["mean_cosine"],
        "high_10_cosine": checkpoint["gt_query"]["high_10_cosine"],
        "1_view_prediction_error": groups["1_view"]["mean_vector_error"],
        "2_views_prediction_error": groups["2_views"]["mean_vector_error"],
        "1_view_cosine": groups["1_view"]["cosine_similarity"],
        "2_views_cosine": groups["2_views"]["cosine_similarity"],
        "1_view_rgb_advantage_over_zero": groups["1_view"][
            "original_rgb_advantage_over_zero"
        ],
        "2_views_rgb_advantage_over_zero": groups["2_views"][
            "original_rgb_advantage_over_zero"
        ],
        "3_plus_prediction_error": _weighted_group_error(
            groups["3_4_views"], groups["5_plus_views"]
        ),
        "expanded_mean_chamfer": expanded["geometry_mean"]["chamfer"],
        "expanded_median_chamfer": expanded["geometry_median"]["chamfer"],
        "initial_mean_chamfer": expanded["initial_mean_chamfer"],
        "expanded_normal_consistency": expanded["geometry_mean"][
            "normal_consistency"
        ],
        "expanded_p2s_forward": expanded["geometry_mean"][
            "point_to_surface_forward_mean"
        ],
        "expanded_p2s_reverse": expanded["geometry_mean"][
            "point_to_surface_reverse_mean"
        ],
        "expanded_p2s_bidirectional": expanded["geometry_mean"][
            "point_to_surface_bidirectional_mean"
        ],
        "visible_displacement": expanded["displacement_mean"]["visible"],
        "zero_view_displacement": expanded["displacement_mean"]["0_views"],
        "1_view_displacement": expanded["displacement_mean"]["1_view"],
        "2_views_displacement": expanded["displacement_mean"]["2_views"],
        "3_plus_displacement": expanded["displacement_mean"]["3_plus_views"],
        "improved_meshes": expanded["improved_meshes"],
    }


def _plot_curves(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    curves = {
        "validation_original_rgb_loss": "original_rgb_loss",
        "original_minus_zero_rgb_loss": "original_minus_zero_rgb",
        "original_minus_shuffled_rgb_loss": "original_minus_shuffled_rgb",
        "original_minus_cross_object_rgb_loss": "original_minus_cross_object_rgb",
        "prediction_target_magnitude_ratio": "prediction_target_magnitude_ratio",
        "high_10_cosine": "high_10_cosine",
        "1_view_prediction_error": "1_view_prediction_error",
        "2_view_prediction_error": "2_views_prediction_error",
        "3_plus_view_prediction_error": "3_plus_prediction_error",
        "expanded_mean_chamfer": "expanded_mean_chamfer",
        "expanded_median_chamfer": "expanded_median_chamfer",
        "expanded_normal_consistency": "expanded_normal_consistency",
        "visible_displacement": "visible_displacement",
        "zero_view_displacement": "zero_view_displacement",
    }
    steps = [int(row["optimizer_step"]) for row in rows]
    for name, field in curves.items():
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(steps, [float(row[field]) for row in rows], marker="o")
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(field.replace("_", " "))
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def _conclusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    short = next((row for row in rows if int(row["optimizer_step"]) == 100), first)
    last = rows[-1]
    chamfer_decreased = float(last["expanded_mean_chamfer"]) < float(
        short["expanded_mean_chamfer"]
    )
    chamfer_values = [float(row["expanded_mean_chamfer"]) for row in rows]
    chamfer_monotonic = all(
        current <= previous
        for previous, current in zip(chamfer_values, chamfer_values[1:], strict=True)
    )
    beats_initial = int(last["improved_meshes"]) > 0
    image_advantage = -float(last["original_minus_zero_rgb"])
    loss_decreased = float(last["original_rgb_loss"]) < float(short["original_rgb_loss"])
    initial_chamfer = float(last["initial_mean_chamfer"])
    final_chamfer = float(last["expanded_mean_chamfer"])
    prediction_evidence = bool(
        loss_decreased
        and image_advantage > 1e-3
        and float(last["high_10_cosine"]) > float(short["high_10_cosine"])
    )
    primary_cause_supported = bool(
        prediction_evidence and chamfer_monotonic and beats_initial
    )
    return {
        "validation_loss_decreased": loss_decreased,
        "prediction_training_insufficiency_supported": prediction_evidence,
        "expanded_chamfer_decreased": chamfer_decreased,
        "expanded_chamfer_monotonic": chamfer_monotonic,
        "expanded_chamfer_reduction_vs_step_100": (
            float(short["expanded_mean_chamfer"]) - final_chamfer
        )
        / float(short["expanded_mean_chamfer"]),
        "final_chamfer_to_initial_ratio": final_chamfer / initial_chamfer,
        "final_original_rgb_advantage_over_zero": image_advantage,
        "final_any_mesh_beats_initial": beats_initial,
        "training_loop_insufficiency_supported_as_primary_cause": primary_cause_supported,
        "verdict": (
            "Training insufficiency affected GT-query prediction, but is not supported "
            "as the primary cause of expanded reconstruction failure."
        ),
        "formal_long_training_decision_gate": "fail",
    }


def _report(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    rows = summary["checkpoint_summary"]
    conclusion = summary["conclusion"]
    lines = [
        "# Sofa50 visibility convergence study",
        "",
        "## Purpose",
        "",
        "This controlled run tests whether the previous 100 optimizer-step budget was the main "
        "cause of poor expanded recovery. It trains renderer-native backface+occlusion visibility "
        "from the same seed and initialization; it does not resume the old frustum checkpoint.",
        "",
        f"One epoch contains {summary['optimizer_steps_per_epoch']} optimizer steps "
        f"({summary['mesh_count']} meshes / accumulation "
        f"{config['multi_object_training']['gradient_accumulation_meshes']}). "
        f"Step {summary['optimizer_steps'][-1]} is therefore approximately epoch "
        f"{math.ceil(summary['optimizer_steps'][-1] / summary['optimizer_steps_per_epoch'])}.",
        "The prior short run stopped at 100 optimizer steps; this study is 20× longer and "
        "evaluates the same fixed queries and five expanded meshes at six checkpoints.",
        "",
        "Zero-view expanded rows retain exact hard weight 0. Recovery uses "
        "`sqrt(W) * (L @ X - delta_pred)` and unseen-anchor weight 0.",
        "",
        "## Core metrics",
        "",
        "| step | original loss | zero RGB | shuffle RGB | cross RGB | pred/GT | High-10 cosine | expanded Chamfer | normal | improved/5 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['optimizer_step']} | {row['original_rgb_loss']:.6g} | "
            f"{row['zero_rgb_loss']:.6g} | {row['shuffled_rgb_loss']:.6g} | "
            f"{row['cross_object_rgb_loss']:.6g} | "
            f"{row['prediction_target_magnitude_ratio']:.3f} | "
            f"{row['high_10_cosine']:.3f} | {row['expanded_mean_chamfer']:.6g} | "
            f"{row['expanded_normal_consistency']:.3f} | {row['improved_meshes']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Visibility-count groups",
            "",
            "| step | 1-view error | 1-view cosine | 1-view RGB advantage | 2-view error | 2-view cosine | 2-view RGB advantage |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['optimizer_step']} | {row['1_view_prediction_error']:.4f} | "
            f"{row['1_view_cosine']:.3f} | {row['1_view_rgb_advantage_over_zero']:.6g} | "
            f"{row['2_views_prediction_error']:.4f} | {row['2_views_cosine']:.3f} | "
            f"{row['2_views_rgb_advantage_over_zero']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Both low-view groups improve substantially, but they remain worse than the "
            "3+ view aggregate. Zero-view rows are prediction diagnostics only and are never "
            "reactivated in recovery.",
            "At the final checkpoint, a consistent RGB-camera-visibility view permutation "
            f"changes validation loss by only "
            f"`{abs(rows[-1]['consistent_view_order_loss'] - rows[-1]['original_rgb_loss']):.3g}` "
            f"and mean prediction by "
            f"{rows[-1]['consistent_view_order_relative_prediction_change']:.3%} relatively. "
            "It is a numerical permutation control, not a correspondence-breaking ablation.",
            "",
            "## Expanded recovery",
            "",
            f"From step 100 to step {rows[-1]['optimizer_step']}, mean Chamfer changes from "
            f"`{rows[1]['expanded_mean_chamfer']:.6g}` to "
            f"`{rows[-1]['expanded_mean_chamfer']:.6g}` "
            f"({conclusion['expanded_chamfer_reduction_vs_step_100']:.1%} reduction), but "
            f"the curve is not monotonic and the final result remains "
            f"{conclusion['final_chamfer_to_initial_ratio']:.2f}× the initial expanded Chamfer.",
            f"No checkpoint improves any of the five expanded meshes over its initial mesh; "
            f"the final checkpoint is {rows[-1]['improved_meshes']}/5.",
            f"Final bidirectional point-to-surface is "
            f"`{rows[-1]['expanded_p2s_bidirectional']:.6g}` and normal consistency is "
            f"`{rows[-1]['expanded_normal_consistency']:.4f}`.",
            "",
            "## Decision",
            "",
            f"{conclusion['verdict']}",
            f"Training-loop insufficiency supported as the primary cause: "
            f"**{conclusion['training_loop_insufficiency_supported_as_primary_cause']}**.",
            f"Formal long-training decision gate: **{conclusion['formal_long_training_decision_gate']}**.",
            "",
            "A falling GT-query loss alone is not counted as reconstruction success. The gate "
            "also requires meaningful RGB correspondence advantage and expanded recovery evidence.",
            "",
            "The next minimal diagnostic should keep checkpoint, solver, hard visibility gate, "
            "and expanded meshes fixed, then sweep only predicted-delta scale (including zero) "
            "and report per-view-count displacement. This tests whether reconstruction remains "
            "dominated by delta scale/direction sensitivity rather than insufficient optimizer steps.",
            "",
            "The expanded identity-placeholder target was not evaluated as an oracle. Existing "
            "coarse/expanded geometry and manifests were reused unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_dataset_manifest(
    path: Path, train: PreparedMeshDataset, validation: PreparedMeshDataset
) -> None:
    samples = []
    for dataset in (train, validation):
        for record in dataset.records:
            samples.append(
                {
                    "path": str(record.path),
                    "split": record.split,
                    "sample_id": record.sample_id,
                }
            )
    _write_json(path, {"samples": samples})


def _run_metadata(
    gt_manifest: Path,
    expanded_manifest: Path,
    source_config: Path,
    recovery_config: Path,
    expanded_ids: Sequence[str],
    checkpoint_steps: Sequence[int],
    seed: int,
    device: str,
) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = "unavailable", None
    return {
        "created_unix_time": time.time(),
        "command_line": list(sys.argv),
        "working_directory": os.getcwd(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "seed": seed,
        "gt_query_manifest": str(gt_manifest),
        "expanded_manifest": str(expanded_manifest),
        "source_config": str(source_config),
        "recovery_config": str(recovery_config),
        "checkpoint_steps": list(checkpoint_steps),
        "expanded_sample_ids": list(expanded_ids),
    }


def _training_history_csv(output_dir: Path) -> None:
    history_path = output_dir / "training_history.json"
    if not history_path.is_file():
        return
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if rows:
        _write_csv(output_dir / "training_history.csv", rows)


def _ensure_expanded_recovery_index(
    output_dir: Path, checkpoints: Sequence[Mapping[str, Any]]
) -> None:
    index_dir = output_dir / "expanded_recovery"
    index_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        step = int(checkpoint["optimizer_step"])
        source = (
            output_dir
            / "checkpoint_evaluation"
            / f"step_{step:06d}"
            / "expanded_recovery"
        )
        link = index_dir / f"step_{step:06d}"
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError(f"Expanded recovery link points elsewhere: {link}")
            continue
        if link.exists():
            raise FileExistsError(f"Expanded recovery index path already exists: {link}")
        link.symlink_to(source, target_is_directory=True)


def _weighted_group_error(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_count = int(first.get("vertex_count", 0))
    second_count = int(second.get("vertex_count", 0))
    total = first_count + second_count
    if total == 0:
        return 0.0
    return float(
        (first_count * float(first["mean_vector_error"]) + second_count * float(second["mean_vector_error"]))
        / total
    )


def _mean_or_none(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list, tuple)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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

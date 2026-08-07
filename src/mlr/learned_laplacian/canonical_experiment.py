from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Camera, Mesh
from mlr.io import load_mesh

from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings, _loss_kwargs
from .evaluation import reconstruct_and_evaluate
from .image_ablation import _condition_sample
from .losses import (
    confidence_calibration_metrics,
    laplacian_prediction_metrics,
    weighted_robust_laplacian_loss,
)
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .trainer import load_checkpoint
from .visualization import render_mesh_comparison_grid


GT_CONDITIONS = ("original_rgb", "zero_rgb", "shuffled_images", "cross_object_rgb")
EXPANDED_VARIANTS = ("main_confidence", "hard_visibility_only", "zero_rgb")


def run_canonical_experiment_evaluation(
    run_dir: str | Path,
    gt_manifest: str | Path,
    expanded_manifest: str | Path,
    *,
    checkpoint_epochs: Sequence[int] = (100, 250, 500, 1000, 1500, 2000),
    device: str = "cuda",
) -> dict[str, Any]:
    """Create the requested checkpoint, RGB, confidence, and expanded tables."""

    run = Path(run_dir).resolve()
    gt_manifest = Path(gt_manifest).resolve()
    expanded_manifest = Path(expanded_manifest).resolve()
    config = _read_json(run / "config.json")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    query_config = copy.deepcopy(config)
    query_config.setdefault("query_training", {})["enabled"] = False
    query_config["query_training"]["zero_initial_laplacian"] = True
    checkpoint_rows, ablation_rows, confidence_rows = _evaluate_gt_checkpoints(
        run,
        gt_manifest,
        query_config,
        checkpoint_epochs,
        resolved,
    )
    per_mesh_rows, expanded_rows, visual_failures = _evaluate_expanded(
        run, expanded_manifest, query_config, resolved
    )
    _write_csv(run / "checkpoint_metrics.csv", checkpoint_rows)
    _write_csv(run / "image_ablation.csv", ablation_rows)
    _write_csv(run / "confidence_calibration.csv", confidence_rows)
    _write_csv(run / "per_mesh_metrics.csv", per_mesh_rows)
    _write_csv(run / "expanded_validation.csv", expanded_rows)
    summary = {
        "method": "absolute_gt_h2_normalized_laplacian_with_confidence",
        "gt_manifest": str(gt_manifest),
        "expanded_manifest": str(expanded_manifest),
        "checkpoint_epochs": [int(value) for value in checkpoint_epochs],
        "checkpoint_metrics": checkpoint_rows,
        "image_ablation": ablation_rows,
        "confidence_calibration": confidence_rows,
        "expanded_validation": expanded_rows,
        "per_mesh_metrics": per_mesh_rows,
        "visualization_failures": visual_failures,
        "oracle_confidence": {
            "available": False,
            "reason": (
                "Real expanded meshes have no valid vertexwise GT differential "
                "correspondence. Building one would require the prohibited GT-to-expanded "
                "Laplacian transfer; the placeholder target is deliberately ignored."
            ),
        },
        "legacy_baselines": _legacy_baseline_context(run.parent),
    }
    _write_json(run / "evaluation_summary.json", summary)
    return summary


def _evaluate_gt_checkpoints(
    run: Path,
    manifest: Path,
    config: Mapping[str, Any],
    checkpoint_epochs: Sequence[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    loss_kwargs = _loss_kwargs(config)
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    checkpoint_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    for requested_epoch in checkpoint_epochs:
        checkpoint = run / f"checkpoint_epoch_{requested_epoch:06d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing requested checkpoint: {checkpoint}")
        model = _build_model(config, None, False).to(device)
        payload = load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        condition_arrays: dict[str, list[np.ndarray]] = {
            name: [] for name in GT_CONDITIONS
        }
        targets: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        h_values: list[np.ndarray] = []
        confidence_values: list[np.ndarray] = []
        view_counts: list[np.ndarray] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            prepared = _load_device_item(dataset, index, config, device)
            donor = _load_device_item(dataset, (index + 1) % len(dataset), config, device)
            base = _exact_query_sample(prepared.sample, device)
            permutation = torch.randperm(
                int(base["images"].shape[0]),
                generator=torch.Generator().manual_seed(7 + index * 104729),
            ).to(device)
            original_prediction: np.ndarray | None = None
            original_confidence: np.ndarray | None = None
            for condition in GT_CONDITIONS:
                conditioned = _condition_sample(
                    base, donor.sample["images"], permutation, condition
                )
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    output = model(conditioned)
                prediction = output.delta_hat_prediction.float()
                condition_arrays[condition].append(
                    prediction.detach().cpu().numpy()
                )
                if condition == "original_rgb":
                    if output.confidence_prediction is None:
                        raise RuntimeError("Canonical checkpoint has no confidence head.")
                    original_prediction = prediction.detach().cpu().numpy()
                    original_confidence = (
                        output.confidence_prediction.float().detach().cpu().numpy()
                    )
                    confidence_values.append(original_confidence)
            targets.append(prepared.training_target.detach().cpu().numpy())
            valid_masks.append(
                prepared.sample["valid_scale_mask"].detach().cpu().numpy()
            )
            h_values.append(
                prepared.sample["local_edge_length"].detach().cpu().numpy()
            )
            visibility_value = prepared.sample.get("visibility")
            if visibility_value is None:
                view_counts.append(
                    np.full(len(valid_masks[-1]), int(prepared.sample["num_views"]))
                )
            else:
                view_counts.append(
                    visibility_value.detach().cpu().sum(dim=0).numpy()
                )
            if requested_epoch == checkpoint_epochs[-1]:
                assert original_prediction is not None
                assert original_confidence is not None
                target_array = prepared.training_target.detach().cpu().numpy()
                h_array = prepared.sample["local_edge_length"].detach().cpu().numpy()
                diagnostic_path = (
                    run
                    / "per_vertex_diagnostics"
                    / f"{static['sample_id']}_gt_query.npz"
                )
                diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    diagnostic_path,
                    delta_hat_prediction=original_prediction,
                    delta_hat_gt=target_array,
                    confidence_prediction=original_confidence,
                    h_gt=h_array,
                    delta_pred_raw=original_prediction
                    * (h_array**2 + epsilon)[:, None],
                    delta_gt_raw=target_array * (h_array**2 + epsilon)[:, None],
                    valid_scale_mask=valid_masks[-1],
                )
                vertices = static["vertices"].detach().cpu().numpy()
                faces = static["faces"].detach().cpu().numpy()
                _write_heatmap_ply(
                    run
                    / "gt_normalized_laplacian_magnitude_heatmaps"
                    / f"{static['sample_id']}.ply",
                    vertices,
                    faces,
                    np.linalg.norm(target_array, axis=1),
                )
                _write_heatmap_ply(
                    run
                    / "predicted_normalized_laplacian_magnitude_heatmaps"
                    / f"{static['sample_id']}_gt_query.ply",
                    vertices,
                    faces,
                    np.linalg.norm(original_prediction, axis=1),
                )
                _write_heatmap_ply(
                    run / "confidence_heatmaps" / f"{static['sample_id']}_gt_query.ply",
                    vertices,
                    faces,
                    original_confidence,
                )
        valid_target = _concat_valid(targets, valid_masks)
        valid_h = _concat_valid(h_values, valid_masks)
        target_magnitude = np.linalg.norm(valid_target, axis=1)
        magnitude_order = np.argsort(target_magnitude)
        top_10_count = max(1, int(round(0.10 * len(valid_target))))
        top_1_count = max(1, int(round(0.01 * len(valid_target))))
        top_10_indices = magnitude_order[-top_10_count:]
        top_1_indices = magnitude_order[-top_1_count:]
        smooth_indices = magnitude_order[:-top_10_count]
        condition_metrics: dict[str, Mapping[str, float]] = {}
        for condition in GT_CONDITIONS:
            prediction = _concat_valid(condition_arrays[condition], valid_masks)
            metrics = laplacian_prediction_metrics(
                torch.from_numpy(prediction), torch.from_numpy(valid_target)
            )
            robust_loss = weighted_robust_laplacian_loss(
                torch.from_numpy(prediction),
                torch.from_numpy(valid_target),
                torch.ones(len(prediction)),
                **loss_kwargs,
            )
            condition_metrics[condition] = {
                **metrics,
                "normalized_laplacian_loss": float(robust_loss.item()),
            }
            normalized_endpoint = np.linalg.norm(prediction - valid_target, axis=1)
            raw_endpoint = np.linalg.norm(
                (prediction - valid_target) * (valid_h**2 + epsilon)[:, None],
                axis=1,
            )
            ablation_rows.append(
                {
                    "epoch": int(payload.get("epoch", requested_epoch)),
                    "condition": condition,
                    "normalized_laplacian_mse": metrics["mse"],
                    "normalized_laplacian_loss": float(robust_loss.item()),
                    "global_cosine": metrics["global_cosine"],
                    "mean_per_vertex_cosine": metrics["mean_per_vertex_cosine"],
                    "high_10_percent_cosine": metrics["top_10_percent_cosine"],
                    "high_1_percent_cosine": metrics["top_1_percent_cosine"],
                    "normalized_laplacian_vector_endpoint_error": metrics[
                        "vector_endpoint_error"
                    ],
                    "high_10_percent_normalized_laplacian_error": float(
                        normalized_endpoint[top_10_indices].mean()
                    ),
                    "high_1_percent_normalized_laplacian_error": float(
                        normalized_endpoint[top_1_indices].mean()
                    ),
                    "smooth_bottom_90_percent_normalized_laplacian_error": (
                        float(normalized_endpoint[smooth_indices].mean())
                        if len(smooth_indices)
                        else None
                    ),
                    "raw_laplacian_vector_endpoint_error": float(
                        raw_endpoint.mean()
                    ),
                    "high_10_percent_raw_laplacian_error": float(
                        raw_endpoint[top_10_indices].mean()
                    ),
                    "high_1_percent_raw_laplacian_error": float(
                        raw_endpoint[top_1_indices].mean()
                    ),
                    "prediction_to_gt_norm_ratio": metrics[
                        "prediction_to_target_norm_ratio"
                    ],
                }
            )
        correct = condition_metrics["original_rgb"]
        zero = condition_metrics["zero_rgb"]
        checkpoint_rows.append(
            {
                "epoch": int(payload.get("epoch", requested_epoch)),
                "train_loss": payload.get("train_loss"),
                "validation_loss": payload.get("validation_loss"),
                "validation_normalized_laplacian_mse": correct["mse"],
                "validation_global_cosine": correct["global_cosine"],
                "validation_mean_per_vertex_cosine": correct[
                    "mean_per_vertex_cosine"
                ],
                "validation_high_10_percent_cosine": correct[
                    "top_10_percent_cosine"
                ],
                "validation_high_1_percent_cosine": correct[
                    "top_1_percent_cosine"
                ],
                "validation_normalized_laplacian_vector_endpoint_error": correct[
                    "vector_endpoint_error"
                ],
                "validation_high_10_percent_normalized_laplacian_error": correct[
                    "top_10_percent_vector_endpoint_error"
                ],
                "validation_high_1_percent_normalized_laplacian_error": correct[
                    "top_1_percent_vector_endpoint_error"
                ],
                "validation_smooth_bottom_90_percent_normalized_laplacian_error": (
                    ablation_rows[-len(GT_CONDITIONS)][
                        "smooth_bottom_90_percent_normalized_laplacian_error"
                    ]
                ),
                "validation_raw_laplacian_vector_endpoint_error": ablation_rows[
                    -len(GT_CONDITIONS)
                ]["raw_laplacian_vector_endpoint_error"],
                "validation_high_10_percent_raw_laplacian_error": ablation_rows[
                    -len(GT_CONDITIONS)
                ]["high_10_percent_raw_laplacian_error"],
                "validation_high_1_percent_raw_laplacian_error": ablation_rows[
                    -len(GT_CONDITIONS)
                ]["high_1_percent_raw_laplacian_error"],
                "prediction_to_gt_norm_ratio": correct[
                    "prediction_to_target_norm_ratio"
                ],
                "correct_minus_zero_rgb_mse_gap": zero["mse"] - correct["mse"],
                "correct_minus_zero_rgb_loss_gap": zero[
                    "normalized_laplacian_loss"
                ]
                - correct["normalized_laplacian_loss"],
                "correct_minus_zero_rgb_cosine_gap": correct["global_cosine"]
                - zero["global_cosine"],
            }
        )
        valid_views = _concat_valid(view_counts, valid_masks)
        valid_prediction = _concat_valid(
            condition_arrays["original_rgb"], valid_masks
        )
        endpoint_error = np.linalg.norm(valid_prediction - valid_target, axis=1)
        for name, mask in (
            ("one_view", valid_views == 1),
            ("two_view", valid_views == 2),
            ("three_plus_views", valid_views >= 3),
            ("invisible", valid_views == 0),
        ):
            checkpoint_rows[-1][f"{name}_normalized_laplacian_error"] = (
                float(endpoint_error[mask].mean()) if np.any(mask) else None
            )
        if requested_epoch == checkpoint_epochs[-1]:
            confidence_rows.extend(
                _confidence_table(
                    confidence_values,
                    condition_arrays["original_rgb"],
                    targets,
                    h_values,
                    valid_masks,
                    epsilon,
                )
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return checkpoint_rows, ablation_rows, confidence_rows


def _confidence_table(
    confidence_values: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    h_values: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    epsilon: float,
) -> list[dict[str, Any]]:
    confidence = _concat_valid(confidence_values, valid_masks)
    prediction = _concat_valid(predictions, valid_masks)
    target = _concat_valid(targets, valid_masks)
    h = _concat_valid(h_values, valid_masks)
    calibration = confidence_calibration_metrics(
        torch.from_numpy(confidence),
        torch.from_numpy(prediction),
        torch.from_numpy(target),
        quantile_bins=5,
    )
    order = np.argsort(confidence)
    rows = []
    normalized_error = np.linalg.norm(prediction - target, axis=1)
    raw_error = np.linalg.norm(
        (prediction - target) * (h**2 + epsilon)[:, None], axis=1
    )
    for bin_index, indices in enumerate(np.array_split(order, 5), start=1):
        rows.append(
            {
                "confidence_bin": bin_index,
                "mean_confidence": float(confidence[indices].mean()),
                "normalized_laplacian_error": float(normalized_error[indices].mean()),
                "raw_laplacian_error": float(raw_error[indices].mean()),
                "vertex_count": int(len(indices)),
                "global_confidence_negative_error_correlation": calibration[
                    "correlation_with_negative_error"
                ],
            }
        )
    return rows


def _evaluate_expanded(
    run: Path,
    manifest: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    model = _build_model(config, None, False).to(device)
    load_checkpoint(run / "checkpoint_best.pt", model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    reconstruction = dict(config.get("recovery", {}))
    unseen_anchor_weight = float(reconstruction.get("unseen_anchor_weight", 0.0))
    reconstruction.update(
        {
            "dense_vertex_limit": 5000,
            "chamfer_samples": 3000,
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    per_mesh: list[dict[str, Any]] = []
    visual_failures: list[dict[str, str]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        prepared = _load_device_item(dataset, index, config, device)
        base = _exact_query_sample(prepared.sample, device)
        initial_vertices = static["vertices"].detach().cpu().numpy()
        faces = static["faces"].detach().cpu().numpy()
        visibility = static["visibility_backface_and_occlusion"]
        variants: dict[str, Mesh] = {
            "initial": Mesh(initial_vertices, faces).ensure_normals()
        }
        for variant in EXPANDED_VARIANTS:
            sample = dict(base)
            if variant == "zero_rgb":
                sample["images"] = torch.zeros_like(sample["images"])
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(sample)
            if output.confidence_prediction is None:
                raise RuntimeError("Canonical checkpoint has no confidence head.")
            confidence = (
                None
                if variant == "hard_visibility_only"
                else output.confidence_prediction.float().detach().cpu()
            )
            recovery_inputs = canonical_current_graph_recovery_inputs(
                static["vertices"],
                static["faces"],
                output.delta_hat_prediction.float().detach().cpu(),
                visibility,
                confidence,
                epsilon=epsilon,
            )
            sample_id = str(static["sample_id"])
            variant_dir = run / "recovered_meshes" / sample_id / variant
            metrics = reconstruct_and_evaluate(
                static,
                recovery_inputs.delta_pred_raw,
                variant_dir,
                reconstruction,
                normalized_prediction=recovery_inputs.delta_hat_prediction,
                edge_scale_epsilon=epsilon,
                laplacian_weight=recovery_inputs.weight,
                unseen_anchor_weight=unseen_anchor_weight,
                evaluate_laplacian_prediction=False,
                evaluate_initial_geometry=True,
                solver_confidence=np.ones(len(initial_vertices), dtype=np.float64),
            )
            recovered = load_mesh(variant_dir / "predicted_refined.obj")
            variants[variant] = recovered
            displacement = np.linalg.norm(recovered.vertices - initial_vertices, axis=1)
            visible = recovery_inputs.visible.cpu().numpy().astype(bool)
            count = torch.as_tensor(visibility, dtype=torch.bool).sum(dim=0).numpy()
            topology = _topology_change(initial_vertices, recovered.vertices, faces)
            geometry = metrics["geometry"]["predicted"]
            initial_geometry = metrics["geometry"]["coarse"]
            row = {
                "sample_id": sample_id,
                "variant": variant,
                "initial_chamfer": initial_geometry.get("chamfer"),
                "refined_chamfer": geometry.get("chamfer"),
                "initial_point_to_surface": initial_geometry.get(
                    "point_to_surface_bidirectional_mean"
                ),
                "refined_point_to_surface": geometry.get(
                    "point_to_surface_bidirectional_mean"
                ),
                "initial_normal_consistency": initial_geometry.get(
                    "normal_consistency"
                ),
                "refined_normal_consistency": geometry.get("normal_consistency"),
                "mean_vertex_displacement": float(displacement.mean()),
                "max_vertex_displacement": float(displacement.max()),
                "visible_mean_displacement": _masked_mean(displacement, visible),
                "invisible_mean_displacement": _masked_mean(displacement, ~visible),
                "low_view_1_2_mean_displacement": _masked_mean(
                    displacement, (count >= 1) & (count <= 2)
                ),
                "introduced_flips": topology["introduced_flips"],
                "new_degeneracies": topology["new_degeneracies"],
                "better_than_initial": bool(
                    geometry.get("chamfer", math.inf)
                    < initial_geometry.get("chamfer", -math.inf)
                ),
                "mean_confidence": float(
                    recovery_inputs.confidence_prediction.mean().item()
                ),
                "visible_vertices": int(visible.sum()),
                "invisible_vertices": int((~visible).sum()),
            }
            per_mesh.append(row)
            np.savez_compressed(
                variant_dir / "per_vertex_diagnostics.npz",
                delta_hat_prediction=recovery_inputs.delta_hat_prediction.numpy(),
                delta_pred_raw=recovery_inputs.delta_pred_raw.numpy(),
                h_current=recovery_inputs.h_current.numpy(),
                delta_current_raw=recovery_inputs.delta_current_raw.numpy(),
                confidence_prediction=recovery_inputs.confidence_prediction.numpy(),
                visible=recovery_inputs.visible.numpy(),
                weight=recovery_inputs.weight.numpy(),
                displacement=displacement,
                visibility_count=count,
            )
            _write_heatmap_ply(
                run / "confidence_heatmaps" / f"{sample_id}_{variant}.ply",
                initial_vertices,
                faces,
                recovery_inputs.confidence_prediction.numpy(),
            )
            magnitude = np.linalg.norm(
                recovery_inputs.delta_hat_prediction.numpy(), axis=1
            )
            _write_heatmap_ply(
                run
                / "predicted_normalized_laplacian_magnitude_heatmaps"
                / f"{sample_id}_{variant}.ply",
                initial_vertices,
                faces,
                magnitude,
            )
        try:
            camera = _first_camera(static, image_size=256)
            render_mesh_comparison_grid(
                [(name, mesh) for name, mesh in variants.items()],
                camera,
                run / "fixed-camera_visualizations" / f"{static['sample_id']}.png",
                image_size=256,
                columns=2,
            )
        except Exception as error:
            visual_failures.append(
                {
                    "sample_id": str(static["sample_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    aggregate = []
    for variant in EXPANDED_VARIANTS:
        selected = [row for row in per_mesh if row["variant"] == variant]
        aggregate.append(
            {
                "variant": variant,
                "initial_chamfer": _mean(selected, "initial_chamfer"),
                "refined_chamfer": _mean(selected, "refined_chamfer"),
                "refined_point_to_surface": _mean(
                    selected, "refined_point_to_surface"
                ),
                "refined_normal_consistency": _mean(
                    selected, "refined_normal_consistency"
                ),
                "introduced_flips": int(sum(row["introduced_flips"] for row in selected)),
                "new_degeneracies": int(
                    sum(row["new_degeneracies"] for row in selected)
                ),
                "better_than_initial": int(
                    sum(row["better_than_initial"] for row in selected)
                ),
                "mesh_count": len(selected),
            }
        )
    aggregate.extend(
        [
            {
                "variant": "direct_displacement_baseline",
                "note": "legacy controlled same-topology diagnostic; no valid real-expanded target",
            },
            {
                "variant": "normalized_laplacian_residual_baseline",
                "note": "legacy controlled same-topology diagnostic; no valid real-expanded target",
            },
        ]
    )
    return per_mesh, aggregate, visual_failures


def _load_device_item(
    dataset: PreparedMeshDataset,
    index: int,
    config: Mapping[str, Any],
    device: torch.device,
):
    return _prepare_item_for_use(
        _prepare_object_static(dataset.load_static(index), config),
        config,
        device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )


def _exact_query_sample(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(sample)
    result["query_positions"] = result["vertices"]
    result["query_is_exact"] = torch.ones(
        result["vertices"].shape[0], dtype=torch.bool, device=device
    )
    return result


def _concat_valid(values: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value)[np.asarray(mask).astype(bool)] for value, mask in zip(values, masks)],
        axis=0,
    )


def _topology_change(
    initial: np.ndarray, recovered: np.ndarray, faces: np.ndarray
) -> dict[str, int]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        recovered[faces[:, 1]] - recovered[faces[:, 0]],
        recovered[faces[:, 2]] - recovered[faces[:, 0]],
    )
    before_degenerate = np.linalg.norm(before, axis=1) <= 1e-14
    after_degenerate = np.linalg.norm(after, axis=1) <= 1e-14
    return {
        "introduced_flips": int(np.sum(np.einsum("ij,ij->i", before, after) < 0)),
        "new_degeneracies": int(np.sum(after_degenerate & ~before_degenerate)),
    }


def _first_camera(sample: Mapping[str, Any], image_size: int) -> Camera:
    intrinsics = np.asarray(sample["intrinsics"][0])
    source_size = int(sample.get("prepared_image_size", sample.get("image_width", 960)))
    scaled = intrinsics.copy() * (float(image_size) / float(source_size))
    scaled[2, 2] = 1.0
    extrinsics = np.asarray(sample["extrinsics"][0])
    return Camera(
        intrinsics=scaled,
        rotation=extrinsics[:3, :3],
        translation=extrinsics[:3, 3],
        image_size=(image_size, image_size),
        name="fixed_view_0",
    )


def _write_heatmap_ply(
    path: Path, vertices: np.ndarray, faces: np.ndarray, values: np.ndarray
) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = np.quantile(values, [0.01, 0.99])
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    colors = np.stack(
        (
            np.round(255 * normalized),
            np.round(255 * (1.0 - np.abs(2.0 * normalized - 1.0))),
            np.round(255 * (1.0 - normalized)),
        ),
        axis=1,
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors):
            stream.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n"
            )
        for face in faces:
            stream.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _legacy_baseline_context(root: Path) -> dict[str, Any]:
    path = root / "sofa50_residual_target_comparison" / "summary.json"
    if not path.is_file():
        return {"available": False}
    payload = _read_json(path)
    return {
        "available": True,
        "path": str(path),
        "comparable_to_real_expanded_evaluation": False,
        "reason": (
            "These baselines use controlled perturbed/control expanded pairs and 500 "
            "optimizer steps on one training mesh. They are retained for reproducibility "
            "but cannot fill a 40-mesh GT-query real-expanded comparison without forbidden "
            "expanded target fabrication."
        ),
        "aggregates": payload.get("aggregates"),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if len(selected) else None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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

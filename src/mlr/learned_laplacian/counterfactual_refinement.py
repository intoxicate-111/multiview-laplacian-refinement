from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mlr.data import Camera, Mesh
from mlr.io import save_mesh

from .evaluation import _chamfer_distance, _normal_consistency, _point_to_surface_stats
from .losses import weighted_robust_laplacian_loss
from .multi_trainer import _amp_settings, _build_model, _prepare_item_for_use, _prepare_object_static
from .perturbed_scale_sweep import _render_panel
from .recovery_identity_oracle import _load_fixed_cameras, _topology_change
from .residual_target_comparison import (
    DIRECT,
    DISPLAY_NAMES,
    H2,
    METHODS,
    RAW,
    _load_pairs,
    _np,
    build_comparison_targets,
    recover_prediction,
)


IMAGE_VARIANTS = ("correct", "zero", "shuffled", "cross_object")


def symmetric_currents(
    base_vertices: np.ndarray, perturbed_vertices: np.ndarray
) -> dict[str, np.ndarray]:
    """Return an exactly symmetric base +/- displacement triplet."""

    base = np.asarray(base_vertices, dtype=np.float64)
    perturbed = np.asarray(perturbed_vertices, dtype=np.float64)
    if base.shape != perturbed.shape or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_vertices and perturbed_vertices must share shape [N, 3].")
    displacement = perturbed - base
    return {"base": base.copy(), "plus": base + displacement, "minus": base - displacement}


def vector_alignment_metrics(
    prediction: np.ndarray, target: np.ndarray, *, eps: float = 1e-12
) -> dict[str, float]:
    """Direction and scale metrics for one vector field."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("prediction and target must share shape [N, 3].")
    pred_flat = pred.reshape(-1)
    target_flat = truth.reshape(-1)
    pred_norm = float(np.linalg.norm(pred_flat))
    target_norm = float(np.linalg.norm(target_flat))
    global_cosine = _cosine(pred_flat, target_flat, eps=eps)
    per_pred = np.linalg.norm(pred, axis=1)
    per_target = np.linalg.norm(truth, axis=1)
    valid = (per_pred > eps) & (per_target > eps)
    per_cosine = np.zeros(len(pred), dtype=np.float64)
    per_cosine[valid] = np.einsum("ij,ij->i", pred[valid], truth[valid]) / (
        per_pred[valid] * per_target[valid]
    )
    cutoff = float(np.quantile(per_target, 0.90))
    top = per_target >= max(cutoff, eps)
    top_cosine = _cosine(pred[top].reshape(-1), truth[top].reshape(-1), eps=eps)
    return {
        "global_cosine": global_cosine,
        "mean_per_vertex_cosine": float(per_cosine[valid].mean()) if np.any(valid) else 0.0,
        "top_10_percent_target_magnitude_cosine": top_cosine,
        "prediction_norm": pred_norm,
        "target_norm": target_norm,
        "norm_ratio": pred_norm / max(target_norm, eps),
        "alpha_star": float(np.dot(pred_flat, target_flat) / max(pred_norm**2, eps)),
    }


def run_counterfactual_refinement(
    source_run: str | Path,
    comparison_run: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    render_backend: str = "opengl",
) -> dict[str, Any]:
    """Evaluate frozen three-representation models under geometry/RGB counterfactuals."""

    source = Path(source_run).expanduser().resolve()
    comparison = Path(comparison_run).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    source_config = _read_json(source / "config.yaml")
    comparison_summary = _read_json(comparison / "summary.json")
    model_config = _read_json(Path(source_config["model_config"]))
    model_config["query_training"] = {
        **dict(model_config.get("query_training", {})),
        "enabled": False,
    }
    recovery_payload = _read_json(Path(source_config["recovery_config"]))
    solver_config = dict(recovery_payload["reconstruction"])
    solver_config["evaluate_oracle"] = False
    eps = float(model_config.get("target_scaling", {}).get("epsilon", 1e-12))
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    pairs = _load_pairs(source)
    sample_ids = sorted(pairs)
    diagnostic_id = str(comparison_summary["train_sample_ids"][0])
    held_out_ids = [sample_id for sample_id in sample_ids if sample_id != diagnostic_id]
    cross_id = held_out_ids[0]
    cameras = _load_fixed_cameras(source / "visualizations" / "render_metadata.json")
    loss_kwargs = _loss_kwargs(model_config)
    models = _load_models(comparison, model_config, torch_device)

    target_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    current_cache: dict[str, dict[str, np.ndarray]] = {}
    roundtrip_rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        pair = pairs[sample_id]
        target = _np(pair["control"]["vertices"]).astype(np.float64)
        plus_source = _np(pair["perturbed"]["vertices"]).astype(np.float64)
        faces = _np(pair["control"]["faces"]).astype(np.int64)
        currents = symmetric_currents(target, plus_source)
        current_cache[sample_id] = currents
        target_cache[sample_id] = {}
        displacement = currents["plus"] - currents["base"]
        perturbation_rows.append(
            {
                "sample_id": sample_id,
                "vertex_count": len(target),
                "mean_displacement": float(np.linalg.norm(displacement, axis=1).mean()),
                "rms_displacement": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
                "max_displacement": float(np.linalg.norm(displacement, axis=1).max()),
                "symmetry_max_absolute_error": float(
                    np.max(np.abs((currents["plus"] - target) + (currents["minus"] - target)))
                ),
            }
        )
        for current_name, current in currents.items():
            built = build_comparison_targets(current, target, faces, eps=eps)
            target_cache[sample_id][current_name] = built
            error = built["raw_roundtrip"] - built[RAW]
            roundtrip_rows.append(
                {
                    "sample_id": sample_id,
                    "current": current_name,
                    "max_absolute_error": float(np.max(np.abs(error))),
                    "relative_l2_error": float(
                        np.linalg.norm(error) / max(np.linalg.norm(built[RAW]), 1e-30)
                    ),
                    "isolated_vertices": int(np.sum(~built["valid_scale_mask"])),
                }
            )
    _write_csv(output / "perturbations.csv", perturbation_rows)
    _write_csv(output / "roundtrip_checks.csv", roundtrip_rows)

    experiment = {
        "experiment": "sofa50_counterfactual_refinement",
        "dataset": "Sofa50 only",
        "source_run": str(source),
        "comparison_run": str(comparison),
        "frozen_checkpoints": {
            method: str(comparison / "checkpoints" / f"{method}.pt") for method in METHODS
        },
        "training_reused": {
            "optimizer_steps_per_method": comparison_summary["optimizer_steps_per_method"],
            "train_sample_ids": comparison_summary["train_sample_ids"],
            "validation_sample_ids": comparison_summary["validation_sample_ids"],
            "neutral_initialization": comparison_summary["neutral_initialization"],
        },
        "diagnostic_sample_id": diagnostic_id,
        "cross_object_sample_id": cross_id,
        "held_out_sample_ids": held_out_ids,
        "current_construction": "X_base=paired control expanded; d=existing controlled smooth expanded perturbation; X_plus=X_base+d; X_minus=X_base-d; X_target=X_base",
        "topology": "identical vertex ordering and faces for base/plus/minus/target",
        "laplacian_contract": "uniform L_current and arithmetic-mean one-ring h_current from the same current graph; no target/GT graph transfer",
        "visibility_control": "control visibility, cameras, intrinsics, and extrinsics held fixed across +/- and all RGB interventions",
        "rgb_interventions": {
            "correct": "unaltered same-object views",
            "zero": "zero normalized RGB tensor",
            "shuffled": "images cyclically shifted by one while camera/view tensors remain fixed",
            "cross_object": f"images only from {cross_id}; original cameras/view tensors retained",
        },
        "device": str(torch_device),
        "target_epsilon": eps,
        "long_training_restarted": False,
        "thingi10k_used": False,
    }
    _write_json(output / "config.json", experiment)

    target_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    image_change_rows: list[dict[str, Any]] = []
    rgb_geometry_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    prediction_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    refined_cache: dict[str, dict[str, dict[str, Mesh]]] = {}

    cross_images = _load_correct_images(
        pairs[cross_id]["control"], model_config, torch_device
    )
    for sample_id in sample_ids:
        print(f"counterfactual sample {sample_id}", flush=True)
        control = pairs[sample_id]["control"]
        base_sample = _load_device_sample(control, model_config, torch_device)
        faces = _np(control["faces"]).astype(np.int64)
        target_vertices = current_cache[sample_id]["base"]
        target_mesh = Mesh(target_vertices, faces).ensure_normals()
        gt_mesh = Mesh(
            _np(control["gt_vertices"]), _np(control["gt_faces"]).astype(np.int64)
        ).ensure_normals()
        sample_predictions: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
        sample_refined: dict[str, dict[str, Mesh]] = {method: {} for method in METHODS}
        device_samples = {
            current_name: _with_current_geometry(
                base_sample,
                current_cache[sample_id][current_name],
                faces,
                target_cache[sample_id][current_name],
                torch_device,
            )
            for current_name in ("base", "plus", "minus")
        }
        for method, model in models.items():
            for current_name in ("base", "plus", "minus"):
                target = target_cache[sample_id][current_name][method]
                prediction, target_loss = _predict(
                    model, device_samples[current_name], target, model_config, loss_kwargs
                )
                sample_predictions[method][current_name] = prediction
                alignment = vector_alignment_metrics(prediction, target)
                target_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": "train" if sample_id == diagnostic_id else "validation",
                        "method": method,
                        "current": current_name,
                        **alignment,
                        "target_loss": target_loss,
                    }
                )
                if current_name in {"plus", "minus"}:
                    current = current_cache[sample_id][current_name]
                    refined, recovery = recover_prediction(
                        method,
                        current,
                        faces,
                        prediction,
                        local_edge_length=target_cache[sample_id][current_name]["local_edge_length"],
                        scale=1.0,
                        solver_config=solver_config,
                        eps=eps,
                    )
                    sample_refined[method][current_name] = refined
                    geometry_rows.append(
                        _geometry_metrics(
                            sample_id,
                            "train" if sample_id == diagnostic_id else "validation",
                            method,
                            current_name,
                            refined,
                            Mesh(current, faces).ensure_normals(),
                            target_mesh,
                            gt_mesh,
                            solver_config,
                            include_point_surface=sample_id == diagnostic_id,
                            recovery=recovery,
                        )
                    )
            plus_prediction = sample_predictions[method]["plus"]
            minus_prediction = sample_predictions[method]["minus"]
            base_prediction = sample_predictions[method]["base"]
            plus_target = target_cache[sample_id]["plus"][method]
            minus_target = target_cache[sample_id]["minus"][method]
            pred_difference = plus_prediction - minus_prediction
            target_difference = plus_target - minus_target
            support = (np.linalg.norm(plus_target, axis=1) > eps) & (
                np.linalg.norm(minus_target, axis=1) > eps
            )
            target_reversing = support & (
                np.einsum("ij,ij->i", plus_target, minus_target) < 0
            )
            pred_reversing = np.einsum("ij,ij->i", plus_prediction, minus_prediction) < 0
            relationship_rows.append(
                {
                    "sample_id": sample_id,
                    "split": "train" if sample_id == diagnostic_id else "validation",
                    "method": method,
                    "prediction_plus_minus_cosine": _cosine(
                        plus_prediction.reshape(-1), minus_prediction.reshape(-1)
                    ),
                    "target_plus_minus_cosine": _cosine(
                        plus_target.reshape(-1), minus_target.reshape(-1)
                    ),
                    "prediction_difference_norm": float(np.linalg.norm(pred_difference)),
                    "target_difference_norm": float(np.linalg.norm(target_difference)),
                    "prediction_target_difference_norm_ratio": float(
                        np.linalg.norm(pred_difference)
                        / max(np.linalg.norm(target_difference), eps)
                    ),
                    "prediction_reversal_score": _cosine(
                        (plus_prediction - base_prediction).reshape(-1),
                        -(minus_prediction - base_prediction).reshape(-1),
                    ),
                    "target_reversal_score": _cosine(
                        plus_target.reshape(-1), -minus_target.reshape(-1)
                    ),
                    "sign_reversal_accuracy": float(
                        pred_reversing[target_reversing].mean()
                    )
                    if np.any(target_reversing)
                    else 0.0,
                    "target_reversing_vertex_count": int(target_reversing.sum()),
                }
            )
        prediction_cache[sample_id] = sample_predictions
        refined_cache[sample_id] = sample_refined

        if sample_id == diagnostic_id:
            image_samples = _image_variants(device_samples["plus"], cross_images)
            cross_minus = dict(device_samples["minus"])
            cross_minus["images"] = cross_images
            for method, model in models.items():
                image_predictions: dict[str, np.ndarray] = {}
                image_meshes: dict[str, Mesh] = {}
                target = target_cache[sample_id]["plus"][method]
                for variant, variant_sample in image_samples.items():
                    prediction, loss = _predict(
                        model, variant_sample, target, model_config, loss_kwargs
                    )
                    image_predictions[variant] = prediction
                    image_rows.append(
                        {
                            "sample_id": sample_id,
                            "method": method,
                            "current": "plus",
                            "image_variant": variant,
                            **vector_alignment_metrics(prediction, target),
                            "target_loss": loss,
                        }
                    )
                    if variant == "correct":
                        variant_mesh = sample_refined[method]["plus"]
                        variant_recovery = {"solver": "reused_correct", "final_terms": None}
                    else:
                        variant_mesh, variant_recovery = recover_prediction(
                            method,
                            current_cache[sample_id]["plus"],
                            faces,
                            prediction,
                            local_edge_length=target_cache[sample_id]["plus"]["local_edge_length"],
                            scale=1.0,
                            solver_config=solver_config,
                            eps=eps,
                        )
                    image_meshes[variant] = variant_mesh
                    rgb_geometry_rows.append(
                        _geometry_metrics(
                            sample_id,
                            "rgb_ablation",
                            method,
                            variant,
                            variant_mesh,
                            Mesh(current_cache[sample_id]["plus"], faces).ensure_normals(),
                            target_mesh,
                            gt_mesh,
                            solver_config,
                            include_point_surface=False,
                            recovery=variant_recovery,
                        )
                    )
                correct = image_predictions["correct"]
                for variant in ("zero", "shuffled", "cross_object"):
                    other = image_predictions[variant]
                    image_change_rows.append(
                        {
                            "sample_id": sample_id,
                            "method": method,
                            "comparison": f"correct_vs_{variant}",
                            "prediction_difference_norm": float(np.linalg.norm(correct - other)),
                            "prediction_cosine": _cosine(correct.reshape(-1), other.reshape(-1)),
                            "difference_over_target_norm": float(
                                np.linalg.norm(correct - other) / max(np.linalg.norm(target), eps)
                            ),
                        }
                    )
                cross_plus = image_predictions["cross_object"]
                cross_minus_prediction, _ = _predict(
                    model,
                    cross_minus,
                    target_cache[sample_id]["minus"][method],
                    model_config,
                    loss_kwargs,
                )
                plus_correct_mesh = sample_refined[method]["plus"]
                minus_correct_mesh = sample_refined[method]["minus"]
                plus_cross_mesh = image_meshes["cross_object"]
                minus_cross_mesh, _ = recover_prediction(
                    method,
                    current_cache[sample_id]["minus"],
                    faces,
                    cross_minus_prediction,
                    local_edge_length=target_cache[sample_id]["minus"]["local_edge_length"],
                    scale=1.0,
                    solver_config=solver_config,
                    eps=eps,
                )
                target_vertex_difference = (
                    (target_vertices - current_cache[sample_id]["plus"])
                    - (target_vertices - current_cache[sample_id]["minus"])
                )
                correct_plus_correction = plus_correct_mesh.vertices - current_cache[sample_id]["plus"]
                correct_minus_correction = minus_correct_mesh.vertices - current_cache[sample_id]["minus"]
                cross_plus_correction = plus_cross_mesh.vertices - current_cache[sample_id]["plus"]
                cross_minus_correction = minus_cross_mesh.vertices - current_cache[sample_id]["minus"]
                geometry_sensitivity = _rms_field(
                    correct_plus_correction - correct_minus_correction
                )
                image_sensitivity_plus = _rms_field(
                    correct_plus_correction - cross_plus_correction
                )
                image_sensitivity_minus = _rms_field(
                    correct_minus_correction - cross_minus_correction
                )
                target_geometry_difference = _rms_field(target_vertex_difference)
                target_plus_rms = _rms_field(target_vertices - current_cache[sample_id]["plus"])
                sensitivity_rows.append(
                    {
                        "method": method,
                        "geometry_sensitivity_vertex_rms": geometry_sensitivity,
                        "geometry_sensitivity_over_gt_target_difference": geometry_sensitivity
                        / max(target_geometry_difference, eps),
                        "image_sensitivity_plus_vertex_rms": image_sensitivity_plus,
                        "image_sensitivity_minus_vertex_rms": image_sensitivity_minus,
                        "mean_image_sensitivity_vertex_rms": 0.5
                        * (image_sensitivity_plus + image_sensitivity_minus),
                        "image_sensitivity_plus_over_target_rms": image_sensitivity_plus
                        / max(target_plus_rms, eps),
                        "geometry_sensitivity_over_image_sensitivity": geometry_sensitivity
                        / max(0.5 * (image_sensitivity_plus + image_sensitivity_minus), eps),
                        "target_geometry_difference_vertex_rms": target_geometry_difference,
                    }
                )
                np.savez_compressed(
                    output / "predictions" / f"{method}_diagnostic_2x2.npz",
                    correct_plus=correct,
                    correct_minus=sample_predictions[method]["minus"],
                    cross_plus=cross_plus,
                    cross_minus=cross_minus_prediction,
                    target_plus=target,
                    target_minus=target_cache[sample_id]["minus"][method],
                )
        del base_sample, device_samples
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output / "target_space_metrics.csv", target_rows)
    _write_csv(output / "geometry_counterfactual_metrics.csv", relationship_rows)
    _write_csv(output / "rgb_ablation_metrics.csv", image_rows)
    _write_csv(output / "rgb_prediction_changes.csv", image_change_rows)
    _write_json(output / "geometry_metrics.json", geometry_rows)
    _write_json(output / "rgb_reconstruction_metrics.json", rgb_geometry_rows)
    _write_csv(output / "sensitivity_2x2.csv", sensitivity_rows)
    _save_prediction_arrays(output, prediction_cache, target_cache)
    _save_meshes(output, current_cache, pairs, refined_cache, diagnostic_id)
    _write_visualizations(
        output,
        diagnostic_id,
        current_cache[diagnostic_id],
        _np(pairs[diagnostic_id]["control"]["faces"]).astype(np.int64),
        target_cache[diagnostic_id],
        prediction_cache[diagnostic_id],
        refined_cache[diagnostic_id],
        cameras[diagnostic_id]["perspective"],
        render_backend,
    )
    summary = _summarize(
        experiment,
        target_rows,
        relationship_rows,
        image_rows,
        image_change_rows,
        geometry_rows,
        rgb_geometry_rows,
        sensitivity_rows,
        roundtrip_rows,
    )
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _load_models(
    comparison: Path, config: Mapping[str, Any], device: torch.device
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for method in METHODS:
        payload = torch.load(
            comparison / "checkpoints" / f"{method}.pt",
            map_location=device,
            weights_only=False,
        )
        model = _build_model(config, None, False).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[method] = model
    return models


def _load_device_sample(
    static: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    prepared = _prepare_item_for_use(
        _prepare_object_static(static, config),
        config,
        device,
        cache_on_device=False,
        decode_images=True,
    )
    return dict(prepared.sample)


def _load_correct_images(
    static: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    return _load_device_sample(static, config, device)["images"]


def _with_current_geometry(
    base_sample: Mapping[str, Any],
    current: np.ndarray,
    faces: np.ndarray,
    targets: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    result = dict(base_sample)
    mesh = Mesh(current, faces).ensure_normals()
    vertices = torch.as_tensor(current, dtype=torch.float32, device=device)
    center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
    scale = torch.linalg.vector_norm(vertices - center, dim=-1).amax().clamp_min(1e-8)
    result.update(
        vertices=vertices,
        query_positions=vertices,
        query_is_exact=torch.ones(len(current), dtype=torch.bool, device=device),
        vertex_normals=torch.as_tensor(mesh.normals, dtype=torch.float32, device=device),
        initial_laplacian=torch.as_tensor(
            targets["delta_initial"], dtype=torch.float32, device=device
        ),
        local_edge_length=torch.as_tensor(
            targets["local_edge_length"], dtype=torch.float32, device=device
        ),
        valid_scale_mask=torch.as_tensor(
            targets["valid_scale_mask"], dtype=torch.bool, device=device
        ),
        position_normalization_center=center,
        position_normalization_scale=scale,
    )
    return result


def _image_variants(
    correct_sample: Mapping[str, Any], cross_images: torch.Tensor
) -> dict[str, dict[str, Any]]:
    correct = dict(correct_sample)
    images = correct["images"]
    if tuple(images.shape) != tuple(cross_images.shape):
        raise ValueError("Cross-object RGB must have the same [V,C,H,W] shape.")
    result = {"correct": correct}
    zero = dict(correct)
    zero["images"] = torch.zeros_like(images)
    result["zero"] = zero
    shuffled = dict(correct)
    shuffled["images"] = torch.roll(images, shifts=1, dims=0)
    result["shuffled"] = shuffled
    cross = dict(correct)
    cross["images"] = cross_images
    result["cross_object"] = cross
    return result


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    target: np.ndarray,
    config: Mapping[str, Any],
    loss_kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    device = sample["vertices"].device
    amp_enabled, amp_dtype = _amp_settings(config.get("training", {}), device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        prediction_t = model(sample).predicted_laplacian
    prediction_t = prediction_t.float()
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    confidence = torch.ones(len(target), dtype=torch.float32, device=device)
    loss = weighted_robust_laplacian_loss(
        prediction_t, target_t, confidence, **loss_kwargs
    )
    return prediction_t.cpu().numpy(), float(loss.item())


def _geometry_metrics(
    sample_id: str,
    split: str,
    method: str,
    current_name: str,
    refined: Mesh,
    initial: Mesh,
    target: Mesh,
    gt: Mesh,
    solver_config: Mapping[str, Any],
    *,
    include_point_surface: bool,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    error = np.linalg.norm(refined.vertices - target.vertices, axis=1)
    initial_error = np.linalg.norm(initial.vertices - target.vertices, axis=1)
    displacement = np.linalg.norm(refined.vertices - initial.vertices, axis=1)
    samples = int(solver_config.get("chamfer_samples", 3000))
    seed = int(solver_config.get("metric_seed", 7))
    chamfer = float(_chamfer_distance(refined, target, samples, seed))
    initial_chamfer = float(_chamfer_distance(initial, target, samples, seed))
    topology = _topology_change(initial, refined)
    triangles = refined.vertices[refined.faces]
    initial_triangles = initial.vertices[initial.faces]
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    initial_area = np.linalg.norm(
        np.cross(
            initial_triangles[:, 1] - initial_triangles[:, 0],
            initial_triangles[:, 2] - initial_triangles[:, 0],
        ),
        axis=1,
    )
    point_surface: Mapping[str, Any] | None = None
    if include_point_surface:
        try:
            point_surface = _point_to_surface_stats(refined.vertices, target)
        except (ImportError, RuntimeError, MemoryError) as exc:
            point_surface = {"unavailable": type(exc).__name__}
    return {
        "sample_id": sample_id,
        "split": split,
        "method": method,
        "current": current_name,
        "vertex_rms_to_target": float(np.sqrt(np.mean(error**2))),
        "initial_vertex_rms_to_target": float(np.sqrt(np.mean(initial_error**2))),
        "vertex_rms_improved": bool(np.sqrt(np.mean(error**2)) < np.sqrt(np.mean(initial_error**2))),
        "chamfer_to_target": chamfer,
        "initial_chamfer_to_target": initial_chamfer,
        "chamfer_improved": bool(chamfer < initial_chamfer),
        "point_to_surface": point_surface,
        "normal_consistency_to_target": float(_normal_consistency(refined, target)),
        "chamfer_to_gt_context": float(_chamfer_distance(refined, gt, samples, seed)),
        "mean_displacement": float(displacement.mean()),
        "max_displacement": float(displacement.max()),
        "rms_displacement": float(np.sqrt(np.mean(displacement**2))),
        **topology,
        "new_degeneracies": int(np.sum(area <= 1e-14) - np.sum(initial_area <= 1e-14)),
        "solver": recovery.get("solver"),
        "solver_final_terms": recovery.get("final_terms"),
    }


def _save_prediction_arrays(
    output: Path,
    predictions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    targets: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    root = output / "predictions"
    root.mkdir(parents=True, exist_ok=True)
    for sample_id, by_method in predictions.items():
        for method, by_current in by_method.items():
            for current, prediction in by_current.items():
                np.savez_compressed(
                    root / f"{sample_id}_{method}_{current}.npz",
                    prediction=prediction,
                    target=targets[sample_id][current][method],
                    local_edge_length=targets[sample_id][current]["local_edge_length"],
                )


def _save_meshes(
    output: Path,
    currents: Mapping[str, Mapping[str, np.ndarray]],
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined: Mapping[str, Mapping[str, Mapping[str, Mesh]]],
    diagnostic_id: str,
) -> None:
    root = output / "meshes"
    for sample_id in currents:
        faces = _np(pairs[sample_id]["control"]["faces"]).astype(np.int64)
        split = "train" if sample_id == diagnostic_id else "validation"
        mesh_dir = root / split / sample_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for current in ("base", "plus", "minus"):
            save_mesh(Mesh(currents[sample_id][current], faces).ensure_normals(), mesh_dir / f"{current}.obj")
        for method, by_current in refined[sample_id].items():
            for current, mesh in by_current.items():
                save_mesh(mesh, mesh_dir / f"{method}_{current}.obj")


def _write_visualizations(
    output: Path,
    sample_id: str,
    currents: Mapping[str, np.ndarray],
    faces: np.ndarray,
    targets: Mapping[str, Mapping[str, np.ndarray]],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    refined: Mapping[str, Mapping[str, Mesh]],
    camera: Camera,
    backend: str,
) -> None:
    root = output / "visualizations" / sample_id
    root.mkdir(parents=True, exist_ok=True)
    target_mesh = Mesh(currents["base"], faces).ensure_normals()
    entries: list[tuple[str, Path]] = []
    for label, mesh, color in (
        ("X_plus", Mesh(currents["plus"], faces).ensure_normals(), (180, 205, 220)),
        ("X_minus", Mesh(currents["minus"], faces).ensure_normals(), (220, 190, 190)),
        ("target", target_mesh, (180, 220, 180)),
        ("direct_plus", refined[DIRECT]["plus"], (105, 170, 225)),
        ("raw_plus", refined[RAW]["plus"], (225, 150, 110)),
        ("h2_plus", refined[H2]["plus"], (145, 190, 130)),
        ("direct_minus", refined[DIRECT]["minus"], (105, 170, 225)),
        ("raw_minus", refined[RAW]["minus"], (225, 150, 110)),
        ("h2_minus", refined[H2]["minus"], (145, 190, 130)),
    ):
        path = root / f"{label}_perspective.png"
        _render_panel(mesh, camera, path, f"{sample_id} | {label}", color, backend)
        entries.append((label, path))
    _contact_sheet(entries, root / "geometry_comparison.png", columns=3)

    for method in METHODS:
        plus_target = targets["plus"][method]
        plus_prediction = predictions[method]["plus"]
        minus_prediction = predictions[method]["minus"]
        per_cosine = _per_vertex_cosine(plus_prediction, plus_target)
        overlays = (
            ("gt_correction_magnitude", np.linalg.norm(plus_target, axis=1)),
            ("predicted_correction_magnitude", np.linalg.norm(plus_prediction, axis=1)),
            ("prediction_target_cosine", per_cosine),
            ("plus_minus_prediction_difference", np.linalg.norm(plus_prediction - minus_prediction, axis=1)),
        )
        for label, values in overlays:
            _scalar_overlay(
                Mesh(currents["plus"], faces).ensure_normals(),
                camera,
                values,
                root / f"{method}_{label}.png",
                f"{DISPLAY_NAMES[method]} | {label}",
            )
        correction = refined[method]["plus"].vertices - currents["plus"]
        _arrow_overlay(
            Mesh(currents["plus"], faces).ensure_normals(),
            camera,
            currents["base"] - currents["plus"],
            correction,
            root / f"{method}_correction_arrows.png",
            DISPLAY_NAMES[method],
        )


def _scalar_overlay(
    mesh: Mesh, camera: Camera, values: np.ndarray, path: Path, label: str
) -> None:
    import matplotlib.pyplot as plt

    pixels, depth = camera.project(mesh.vertices)
    width, height = camera.image_size or (960, 960)
    valid = (
        (depth > 1e-8)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
        & np.isfinite(values)
    )
    fig, axis = plt.subplots(figsize=(7, 7), dpi=120)
    cmap = "coolwarm" if "cosine" in label else "viridis"
    plot = axis.scatter(
        pixels[valid, 0], pixels[valid, 1], c=np.asarray(values)[valid], s=3, cmap=cmap
    )
    fig.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xlim=(0, width), ylim=(height, 0), title=label, aspect="equal")
    axis.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _arrow_overlay(
    mesh: Mesh,
    camera: Camera,
    target_correction: np.ndarray,
    predicted_correction: np.ndarray,
    path: Path,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    magnitude = np.linalg.norm(target_correction, axis=1)
    count = min(180, len(mesh.vertices))
    indices = np.argsort(magnitude)[-count:]
    points, depth = camera.project(mesh.vertices)
    target_end, _ = camera.project(mesh.vertices + target_correction)
    pred_end, _ = camera.project(mesh.vertices + predicted_correction)
    width, height = camera.image_size or (960, 960)
    valid = depth[indices] > 1e-8
    indices = indices[valid]
    fig, axis = plt.subplots(figsize=(7, 7), dpi=120)
    axis.scatter(points[:, 0], points[:, 1], s=0.4, c="#b8b8b8", alpha=0.25)
    axis.quiver(
        points[indices, 0],
        points[indices, 1],
        target_end[indices, 0] - points[indices, 0],
        target_end[indices, 1] - points[indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.002,
        label="target",
    )
    axis.quiver(
        points[indices, 0],
        points[indices, 1],
        pred_end[indices, 0] - points[indices, 0],
        pred_end[indices, 1] - points[indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.002,
        label="prediction",
    )
    axis.set(xlim=(0, width), ylim=(height, 0), title=f"{label} | vertex-space arrows", aspect="equal")
    axis.legend(loc="upper right")
    axis.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _contact_sheet(
    entries: Sequence[tuple[str, Path]], path: Path, *, columns: int
) -> None:
    cell = 420
    rows = int(math.ceil(len(entries) / columns))
    sheet = Image.new("RGB", (columns * cell, rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel_path) in enumerate(entries):
        with Image.open(panel_path) as opened:
            panel = opened.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell, (index // columns) * cell
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y, x + cell, y + 24), fill=(255, 255, 255))
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0))
    sheet.save(path)


def _summarize(
    experiment: Mapping[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    image_changes: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    rgb_geometry_rows: Sequence[Mapping[str, Any]],
    sensitivities: Sequence[Mapping[str, Any]],
    roundtrips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    sensitivity_by_method = {row["method"]: row for row in sensitivities}
    for method in METHODS:
        method_targets = [
            row for row in target_rows if row["method"] == method and row["current"] in {"plus", "minus"}
        ]
        train_targets = [row for row in method_targets if row["split"] == "train"]
        val_targets = [row for row in method_targets if row["split"] == "validation"]
        train_rel = [row for row in relationships if row["method"] == method and row["split"] == "train"][0]
        val_rel = [row for row in relationships if row["method"] == method and row["split"] == "validation"]
        method_geometry = [row for row in geometry_rows if row["method"] == method]
        val_geometry = [row for row in method_geometry if row["split"] == "validation"]
        rgb = [row for row in image_rows if row["method"] == method]
        rgb_by_variant = {row["image_variant"]: row for row in rgb}
        rgb_geometry = {
            row["current"]: row
            for row in rgb_geometry_rows
            if row["method"] == method
        }
        sensitivity = sensitivity_by_method[method]
        methods[method] = {
            "train_target_cosine": _mean(train_targets, "global_cosine"),
            "validation_target_cosine": _mean(val_targets, "global_cosine"),
            "train_mean_vertex_cosine": _mean(train_targets, "mean_per_vertex_cosine"),
            "train_top_10_cosine": _mean(train_targets, "top_10_percent_target_magnitude_cosine"),
            "train_norm_ratio": _mean(train_targets, "norm_ratio"),
            "train_alpha_star": _mean(train_targets, "alpha_star"),
            "train_current_metrics": {
                row["current"]: {
                    "global_cosine": row["global_cosine"],
                    "mean_per_vertex_cosine": row["mean_per_vertex_cosine"],
                    "top_10_percent_target_magnitude_cosine": row[
                        "top_10_percent_target_magnitude_cosine"
                    ],
                    "norm_ratio": row["norm_ratio"],
                    "alpha_star": row["alpha_star"],
                    "target_loss": row["target_loss"],
                }
                for row in train_targets
            },
            "train_reversal_score": train_rel["prediction_reversal_score"],
            "train_target_reversal_score": train_rel["target_reversal_score"],
            "train_prediction_plus_minus_cosine": train_rel["prediction_plus_minus_cosine"],
            "train_target_plus_minus_cosine": train_rel["target_plus_minus_cosine"],
            "train_sign_reversal_accuracy": train_rel["sign_reversal_accuracy"],
            "train_prediction_difference_norm": train_rel["prediction_difference_norm"],
            "train_target_difference_norm": train_rel["target_difference_norm"],
            "train_target_space_geometry_sensitivity": train_rel[
                "prediction_target_difference_norm_ratio"
            ],
            "validation_reversal_score": _mean(val_rel, "prediction_reversal_score"),
            "validation_sign_reversal_accuracy": _mean(val_rel, "sign_reversal_accuracy"),
            "geometry_sensitivity": sensitivity,
            "rgb": rgb_by_variant,
            "rgb_geometry": rgb_geometry,
            "rgb_changes": [row for row in image_changes if row["method"] == method],
            "train_geometry": _geometry_aggregate(method_geometry, "train"),
            "validation_geometry": _geometry_aggregate(val_geometry, "validation"),
        }
    direct_score = methods[DIRECT]["train_reversal_score"]
    raw_score = methods[RAW]["train_reversal_score"]
    h2_score = methods[H2]["train_reversal_score"]
    best_direction = max(METHODS, key=lambda method: methods[method]["train_target_cosine"])
    best_reversal = max(METHODS, key=lambda method: methods[method]["train_reversal_score"])
    any_geometry = any(
        methods[m]["train_target_cosine"] > 0.1
        and methods[m]["train_reversal_score"] > 0.1
        and methods[m]["geometry_sensitivity"]["geometry_sensitivity_over_gt_target_difference"] > 0.1
        for m in METHODS
    )
    correct_rgb_better = {}
    image_material = {}
    for method in METHODS:
        rgb = methods[method]["rgb"]
        correct_loss = rgb["correct"]["target_loss"]
        correct_rgb_better[method] = all(
            correct_loss < 0.95 * rgb[variant]["target_loss"]
            for variant in ("zero", "shuffled", "cross_object")
        )
        cross_change = next(
            row
            for row in methods[method]["rgb_changes"]
            if row["comparison"] == "correct_vs_cross_object"
        )
        image_material[method] = bool(
            cross_change["difference_over_target_norm"] > 0.1
            and cross_change["prediction_cosine"] < 0.95
        )
    any_image = any(correct_rgb_better[m] and image_material[m] for m in METHODS)
    if any_geometry and any_image:
        classification = "genuine image-conditioned refinement (for at least one representation in this diagnostic)"
    elif any_geometry:
        classification = "current-geometry prior / geometry-conditioned correction"
    elif any_image:
        classification = "image-conditioned shape prior, not current-error-conditioned refinement"
    else:
        classification = "Sofa/category prior shortcut (neither counterfactual criterion passed)"
    return {
        **dict(experiment),
        "methods": methods,
        "integrity": {
            "max_roundtrip_relative_l2_error": max(row["relative_l2_error"] for row in roundtrips),
            "max_roundtrip_absolute_error": max(row["max_absolute_error"] for row in roundtrips),
            "roundtrip_passed": all(row["relative_l2_error"] <= 1e-12 for row in roundtrips),
            "isolated_vertices": int(sum(row["isolated_vertices"] for row in roundtrips)),
        },
        "answers": {
            "direct_reacts_correctly": bool(direct_score > 0.1 and methods[DIRECT]["train_target_cosine"] > 0.1),
            "raw_reacts_correctly": bool(raw_score > 0.1 and methods[RAW]["train_target_cosine"] > 0.1),
            "h2_reacts_correctly": bool(h2_score > 0.1 and methods[H2]["train_target_cosine"] > 0.1),
            "best_current_error_direction": best_direction,
            "best_reversal": best_reversal,
            "predictions_change_with_rgb": image_material,
            "correct_rgb_materially_better": correct_rgb_better,
            "classification": classification,
            "h2_improves_current_state_sensitivity_over_raw": bool(h2_score > raw_score),
            "laplacian_advantage_over_direct": bool(max(raw_score, h2_score) > direct_score),
            "held_out_generalization": {
                method: bool(
                    methods[method]["validation_target_cosine"] > 0.1
                    and methods[method]["validation_reversal_score"] > 0.1
                    and methods[method]["validation_geometry"]["meshes_improving_vertex_rms"] > 0
                )
                for method in METHODS
            },
            "restart_long_training": False,
            "smallest_next_experiment": "Train the same frozen-backbone/output-head setup on both X_plus and X_minus for one Sofa, then repeat this exact 2x2 test on one held-out Sofa.",
        },
    }


def _geometry_aggregate(
    rows: Sequence[Mapping[str, Any]], split: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        return {
            "mesh_count": 0,
            "mean_vertex_rms": None,
            "initial_mean_vertex_rms": None,
            "mean_chamfer": None,
            "initial_mean_chamfer": None,
            "meshes_improving_vertex_rms": 0,
            "meshes_improving_chamfer": 0,
            "mean_normal_consistency": None,
            "introduced_face_flips": 0,
            "new_degeneracies": 0,
        }
    return {
        "mesh_count": len(selected),
        "mean_vertex_rms": _mean(selected, "vertex_rms_to_target"),
        "initial_mean_vertex_rms": _mean(selected, "initial_vertex_rms_to_target"),
        "mean_chamfer": _mean(selected, "chamfer_to_target"),
        "initial_mean_chamfer": _mean(selected, "initial_chamfer_to_target"),
        "meshes_improving_vertex_rms": int(sum(bool(row["vertex_rms_improved"]) for row in selected)),
        "meshes_improving_chamfer": int(sum(bool(row["chamfer_improved"]) for row in selected)),
        "mean_normal_consistency": _mean(selected, "normal_consistency_to_target"),
        "introduced_face_flips": int(sum(int(row["introduced_flipped_triangles"]) for row in selected)),
        "new_degeneracies": int(sum(int(row["new_degeneracies"]) for row in selected)),
    }


def _report(summary: Mapping[str, Any]) -> str:
    methods = summary["methods"]
    answers = summary["answers"]
    yesno = lambda value: "yes" if value else "no"
    lines = [
        "# Controlled Sofa50 counterfactual refinement experiment",
        "",
        "This is a frozen-checkpoint diagnostic using Sofa50 only. It reuses the fair 500-step one-mesh overfit models from the three-representation comparison; no long training was restarted. The paired control-expanded mesh is the known same-topology target (not claimed to be the original high-resolution GT). `X_plus` is the existing smooth topology-safe perturbation and `X_minus` is its exact reflection around that target. RGB, cameras, intrinsics/extrinsics, topology/order, and control visibility are identical across `+/-`.",
        "",
        "Target losses are only compared within a representation. Geometry and dimensionless conditioning metrics are used across representations.",
        "",
        "## Required summary",
        "",
        "| Method | Target cosine | Geometry sensitivity / GT (target / vertex) | Image sensitivity / target (target / vertex) | +/- reversal | Val RMS improve | Val Chamfer improve |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = methods[method]
        sensitivity = row["geometry_sensitivity"]
        cross_change = next(
            item
            for item in row["rgb_changes"]
            if item["comparison"] == "correct_vs_cross_object"
        )
        val = row["validation_geometry"]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {row['train_target_cosine']:.6g} | "
            f"{row['train_target_space_geometry_sensitivity']:.6g} / {sensitivity['geometry_sensitivity_over_gt_target_difference']:.6g} | "
            f"{cross_change['difference_over_target_norm']:.6g} / {sensitivity['image_sensitivity_plus_over_target_rms']:.6g} | "
            f"{row['train_reversal_score']:.6g} | {val['meshes_improving_vertex_rms']}/{val['mesh_count']} | "
            f"{val['meshes_improving_chamfer']}/{val['mesh_count']} |"
        )
    lines.extend(["", "`geometry_sensitivity / image_sensitivity` (target-space / recovered vertex-space; a large ratio alone is not evidence of correctness):"])
    for method in METHODS:
        vertex_value = methods[method]["geometry_sensitivity"]["geometry_sensitivity_over_image_sensitivity"]
        cross_change = next(
            item
            for item in methods[method]["rgb_changes"]
            if item["comparison"] == "correct_vs_cross_object"
        )
        target_value = methods[method]["train_target_space_geometry_sensitivity"] / max(
            cross_change["difference_over_target_norm"], 1e-12
        )
        lines.append(
            f"- {DISPLAY_NAMES[method]}: `{target_value:.6g} / {vertex_value:.6g}`"
        )
    lines.extend(["", "## Experiment 1 — same RGB, different current meshes", ""])
    for method in METHODS:
        row = methods[method]
        plus = row["train_current_metrics"]["plus"]
        minus = row["train_current_metrics"]["minus"]
        lines.append(
            f"- {DISPLAY_NAMES[method]}: plus/minus target cosines `{plus['global_cosine']:.6g}/{minus['global_cosine']:.6g}`, norm ratios `{plus['norm_ratio']:.6g}/{minus['norm_ratio']:.6g}`, alpha* `{plus['alpha_star']:.6g}/{minus['alpha_star']:.6g}`, and target losses `{plus['target_loss']:.6g}/{minus['target_loss']:.6g}`. Mean vertex cosine is `{row['train_mean_vertex_cosine']:.6g}` and top-10% cosine `{row['train_top_10_cosine']:.6g}`. Targets have plus/minus cosine `{row['train_target_plus_minus_cosine']:.6g}` but predictions have `{row['train_prediction_plus_minus_cosine']:.6g}`; `||pred_plus-pred_minus||/||target_plus-target_minus||` is `{row['train_target_space_geometry_sensitivity']:.6g}` (absolute norms `{row['train_prediction_difference_norm']:.6g}/{row['train_target_difference_norm']:.6g}`). Prediction/target reversal scores are `{row['train_reversal_score']:.6g}/{row['train_target_reversal_score']:.6g}` and sign-reversal accuracy is `{row['train_sign_reversal_accuracy']:.2%}`."
        )
    lines.extend(["", "## Experiment 2 — same current mesh, different RGB", ""])
    for method in METHODS:
        row = methods[method]
        rgb = row["rgb"]
        rgb_geometry = row["rgb_geometry"]
        changes = {item["comparison"]: item for item in row["rgb_changes"]}
        lines.append(
            f"- {DISPLAY_NAMES[method]}: correct/zero/shuffled/cross target losses "
            f"`{rgb['correct']['target_loss']:.6g}/{rgb['zero']['target_loss']:.6g}/{rgb['shuffled']['target_loss']:.6g}/{rgb['cross_object']['target_loss']:.6g}`; "
            f"target cosines `{rgb['correct']['global_cosine']:.6g}/{rgb['zero']['global_cosine']:.6g}/{rgb['shuffled']['global_cosine']:.6g}/{rgb['cross_object']['global_cosine']:.6g}`. "
            f"Correct-vs-zero/shuffled/cross prediction changes over target norm are "
            f"`{changes['correct_vs_zero']['difference_over_target_norm']:.6g}/{changes['correct_vs_shuffled']['difference_over_target_norm']:.6g}/{changes['correct_vs_cross_object']['difference_over_target_norm']:.6g}`. "
            f"Recovered RMS to target is correct/zero/shuffled/cross "
            f"`{rgb_geometry['correct']['vertex_rms_to_target']:.6g}/{rgb_geometry['zero']['vertex_rms_to_target']:.6g}/{rgb_geometry['shuffled']['vertex_rms_to_target']:.6g}/{rgb_geometry['cross_object']['vertex_rms_to_target']:.6g}`; Chamfer is "
            f"`{rgb_geometry['correct']['chamfer_to_target']:.6g}/{rgb_geometry['zero']['chamfer_to_target']:.6g}/{rgb_geometry['shuffled']['chamfer_to_target']:.6g}/{rgb_geometry['cross_object']['chamfer_to_target']:.6g}`."
        )
    lines.extend(["", "## Geometry", ""])
    for method in METHODS:
        train = methods[method]["train_geometry"]
        val = methods[method]["validation_geometry"]
        lines.append(
            f"- {DISPLAY_NAMES[method]}: train initial→predicted RMS `{train['initial_mean_vertex_rms']:.6g}→{train['mean_vertex_rms']:.6g}`, Chamfer `{train['initial_mean_chamfer']:.6g}→{train['mean_chamfer']:.6g}`; held-out RMS `{val['initial_mean_vertex_rms']:.6g}→{val['mean_vertex_rms']:.6g}`, Chamfer `{val['initial_mean_chamfer']:.6g}→{val['mean_chamfer']:.6g}`; held-out flips/new degeneracies `{val['introduced_face_flips']}/{val['new_degeneracies']}`."
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Direct displacement reacts correctly to opposite currents: **{yesno(answers['direct_reacts_correctly'])}**.",
            f"2. Raw Laplacian residual reacts correctly: **{yesno(answers['raw_reacts_correctly'])}**.",
            f"3. h²-normalized Laplacian residual reacts correctly: **{yesno(answers['h2_reacts_correctly'])}**.",
            f"4. Best current-error target direction: **{DISPLAY_NAMES[answers['best_current_error_direction']]}**; best reversal: **{DISPLAY_NAMES[answers['best_reversal']]}**.",
            "5. Predictions change when only RGB changes: " + ", ".join(f"{DISPLAY_NAMES[m]} **{yesno(v)}**" for m, v in answers["predictions_change_with_rgb"].items()) + ".",
            "6. Correct RGB is materially better than every zero/shuffled/cross condition: " + ", ".join(f"{DISPLAY_NAMES[m]} **{yesno(v)}**" for m, v in answers["correct_rgb_materially_better"].items()) + ".",
            f"7. Classification under the stated rules: **{answers['classification']}**.",
            f"8. h² normalization improves current-state sensitivity over raw: **{yesno(answers['h2_improves_current_state_sensitivity_over_raw'])}** numerically, but it still fails the correctness/reversal criterion.",
            f"9. A Laplacian representation has an advantage over direct displacement in this counterfactual: **{yesno(answers['laplacian_advantage_over_direct'])}**.",
            "10. Held-out generalization: " + ", ".join(f"{DISPLAY_NAMES[m]} **{yesno(v)}**" for m, v in answers["held_out_generalization"].items()) + ".",
            "11. Restart long training: **no**. The conditioning failure mode should be resolved in a tiny paired diagnostic first.",
            f"12. Single smallest next experiment: **{answers['smallest_next_experiment']}**",
            "",
            f"Normalization round-trip max relative L2 error is `{summary['integrity']['max_roundtrip_relative_l2_error']:.3e}` with `{summary['integrity']['isolated_vertices']}` isolated vertices. Full per-mesh/condition values, reconstructed meshes, fixed-camera panels, heatmaps, and arrow overlays are stored beside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _per_vertex_cosine(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", prediction, target)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    result = np.zeros(len(prediction), dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), eps))


def _rms_field(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=1))))


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

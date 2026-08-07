from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mlr.data import Mesh
from mlr.io import save_mesh

from .dataset import load_prepared_sample
from .evaluation import _chamfer_distance, _normal_consistency, _reconstruct
from .graph_layers import faces_to_edge_index
from .losses import weighted_robust_laplacian_loss
from .multi_trainer import _amp_settings, _build_model, _prepare_item_for_use, _prepare_object_static
from .perturbed_scale_sweep import _render_panel
from .recovery_identity_oracle import (
    _load_fixed_cameras,
    _refinement_config,
    _topology_change,
)
from .recovery_targets import compose_residual_laplacian_target, initial_uniform_laplacian
from .target_scaling import (
    denormalize_laplacian_by_edge_scale,
    incident_edge_length_and_valid_mask,
    normalize_laplacian_by_edge_scale,
)
from .trainer import _resolve_device, _seed_everything, load_checkpoint


DIRECT = "direct_vertex_residual"
RAW = "raw_laplacian_residual"
H2 = "h2_normalized_laplacian_residual"
METHODS = (DIRECT, RAW, H2)
DISPLAY_NAMES = {
    DIRECT: "Direct vertex residual",
    RAW: "Raw Laplacian residual",
    H2: "h2-normalized Laplacian residual",
}


def build_comparison_targets(
    current_vertices: np.ndarray,
    target_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Build all three targets from one current same-topology graph."""

    current = np.asarray(current_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if current.shape != target.shape or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("current_vertices and target_vertices must share shape [N, 3].")
    edge_index = faces_to_edge_index(torch.as_tensor(triangles), len(current))
    h, valid = incident_edge_length_and_valid_mask(
        torch.as_tensor(current, dtype=torch.float64), edge_index, eps=eps
    )
    if not bool(valid.all()):
        raise ValueError("The expanded comparison graph contains isolated vertices.")
    delta_initial = initial_uniform_laplacian(current, triangles)
    delta_target = initial_uniform_laplacian(target, triangles)
    raw_residual = delta_target - delta_initial
    normalized = normalize_laplacian_by_edge_scale(
        torch.as_tensor(raw_residual), h, eps=eps, valid_scale_mask=valid
    )
    raw_roundtrip = denormalize_laplacian_by_edge_scale(normalized, h, eps=eps)
    return {
        DIRECT: target - current,
        RAW: raw_residual,
        H2: normalized.numpy(),
        "delta_initial": delta_initial,
        "delta_target": delta_target,
        "local_edge_length": h.numpy(),
        "valid_scale_mask": valid.numpy(),
        "raw_roundtrip": raw_roundtrip.numpy(),
    }


def recover_prediction(
    method: str,
    current_vertices: np.ndarray,
    faces: np.ndarray,
    prediction: np.ndarray,
    *,
    local_edge_length: np.ndarray,
    scale: float,
    solver_config: Mapping[str, Any],
    eps: float = 1e-12,
) -> tuple[Mesh, dict[str, Any]]:
    """Recover geometry without mixing residual representations."""

    current = np.asarray(current_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if predicted.shape != current.shape:
        raise ValueError("prediction must have shape [N, 3].")
    initial = Mesh(current.copy(), triangles.copy()).ensure_normals()
    if method == DIRECT:
        refined = initial.with_vertices(current + float(scale) * predicted).ensure_normals()
        return refined, {"solver": "none", "final_terms": None}
    if method not in {RAW, H2}:
        raise ValueError(f"Unsupported method: {method!r}.")
    raw_residual = predicted
    if method == H2:
        raw_residual = denormalize_laplacian_by_edge_scale(
            torch.as_tensor(predicted),
            torch.as_tensor(local_edge_length),
            eps=eps,
        ).numpy()
    delta_initial = initial_uniform_laplacian(current, triangles)
    delta_refined = compose_residual_laplacian_target(
        delta_initial, raw_residual, float(scale)
    )
    refinement, solver_name = _reconstruct(
        initial,
        delta_refined,
        np.ones(len(current), dtype=np.float64),
        _refinement_config(solver_config),
        int(solver_config.get("dense_vertex_limit", 5000)),
        laplacian_weight=np.ones(len(current), dtype=np.float64),
    )
    return refinement.mesh.ensure_normals(), {
        "solver": solver_name,
        "final_terms": dict(refinement.history[-1]),
    }


def run_residual_target_comparison(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 100,
    device: str = "cuda",
    render_backend: str = "opengl",
) -> dict[str, Any]:
    if steps < 1 or steps > 500:
        raise ValueError("steps must be in [1, 500] for this short diagnostic.")
    source = Path(source_run).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = _read_json(source / "config.yaml")
    model_config = _read_json(Path(source_config["model_config"]))
    # These are exact current expanded vertices, not the old GT-query
    # augmentation samples.  Disabling augmentation preserves the checkpoint
    # architecture while making the query contract explicit.
    model_config["query_training"] = {
        **dict(model_config.get("query_training", {})),
        "enabled": False,
    }
    recovery_payload = _read_json(Path(source_config["recovery_config"]))
    solver_config = dict(recovery_payload["reconstruction"])
    solver_config["evaluate_oracle"] = False
    eps = float(model_config.get("target_scaling", {}).get("epsilon", 1e-12))
    pairs = _load_pairs(source)
    sample_ids = sorted(pairs)
    train_ids = sample_ids[:1]
    validation_ids = sample_ids[1:]
    cameras = _load_fixed_cameras(source / "visualizations" / "render_metadata.json")
    torch_device = _resolve_device(device)
    seed = int(model_config.get("seed", 7))
    training_config = dict(model_config.get("training", {}))
    loss_kwargs = {
        "loss_type": str(training_config.get("loss", "huber")),
        "huber_delta": float(training_config.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training_config.get("charbonnier_epsilon", 1e-3)),
        "target_magnitude_weight_lambda": float(
            training_config.get("target_magnitude_weight_lambda", 0.0)
        ),
    }
    experiment = {
        "experiment": "sofa50_residual_target_comparison",
        "source_run": str(source),
        "source_checkpoint": source_config["checkpoint"],
        "model_config": source_config["model_config"],
        "recovery_config": source_config["recovery_config"],
        "dataset": "Sofa50 only",
        "supervision": "perturbed-expanded to paired control-expanded (same topology; control is not GT)",
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "optimizer_steps_per_method": int(steps),
        "optimizer": "Adam",
        "learning_rate": float(training_config.get("learning_rate", 1e-3)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "gradient_clip_norm": float(training_config.get("gradient_clip_norm", 0.0)),
        "loss": loss_kwargs,
        "neutral_initialization": "same pretrained checkpoint, then identical zero-initialized final 3-vector output layer",
        "query_vertices": "exact current perturbed-expanded vertices; no query augmentation",
        "target_epsilon": eps,
        "target_loss_comparability": "Losses are comparable within a representation, not numerically across differently scaled target spaces.",
        "recovery": solver_config,
        "correction_visibility_gate": "none for all three formulations",
        "device": str(torch_device),
        "long_training_blocked": True,
    }
    _write_json(output / "config.json", experiment)

    targets: dict[str, dict[str, np.ndarray]] = {}
    roundtrip_rows = []
    for sample_id in sample_ids:
        pair = pairs[sample_id]
        _assert_same_topology(pair["perturbed"], pair["control"], sample_id)
        current = _np(pair["perturbed"]["vertices"])
        control = _np(pair["control"]["vertices"])
        faces = _np(pair["perturbed"]["faces"]).astype(np.int64)
        built = build_comparison_targets(current, control, faces, eps=eps)
        targets[sample_id] = built
        raw = built[RAW]
        error = built["raw_roundtrip"] - raw
        prepared_h = _np(pair["perturbed"]["local_edge_length"])
        roundtrip_rows.append(
            {
                "sample_id": sample_id,
                "max_absolute_error": float(np.max(np.abs(error))),
                "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(raw), 1e-30)),
                "max_h_difference_from_prepared_current_graph": float(
                    np.max(np.abs(built["local_edge_length"] - prepared_h))
                ),
                "isolated_vertices": int(np.sum(~built["valid_scale_mask"])),
            }
        )
    _write_csv(output / "roundtrip_checks.csv", roundtrip_rows)

    train_sample = pairs[train_ids[0]]["perturbed"]
    prepared_train = _prepare_item_for_use(
        _prepare_object_static(train_sample, model_config),
        model_config,
        torch_device,
        cache_on_device=False,
        decode_images=True,
    )
    initial_state = _neutral_initial_state(
        model_config, Path(source_config["checkpoint"]), torch_device
    )
    method_states: dict[str, dict[str, torch.Tensor]] = {}
    training_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for method in METHODS:
        model, history, best_loss, runtime, peak = _train_one_method(
            method,
            prepared_train.sample,
            torch.as_tensor(targets[train_ids[0]][method], device=torch_device),
            model_config,
            initial_state,
            steps,
            torch_device,
            loss_kwargs,
        )
        method_states[method] = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        for row in history:
            training_rows.append({"method": method, **row})
        runtime_rows.append(
            {
                "method": method,
                "best_training_target_loss": best_loss,
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": peak,
            }
        )
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "optimizer_steps": steps,
                "model_state_dict": method_states[method],
                "experiment_config": experiment,
            },
            checkpoint_dir / f"{method}.pt",
        )
        del model
    _write_csv(output / "training_history.csv", training_rows)
    _write_csv(output / "runtime.csv", runtime_rows)

    models = {}
    for method in METHODS:
        model = _build_model(model_config, None, False).to(torch_device)
        model.load_state_dict(method_states[method])
        model.eval()
        models[method] = model

    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    refined_meshes: dict[str, dict[str, Mesh]] = {sample_id: {} for sample_id in sample_ids}
    for sample_id in sample_ids:
        split = "train" if sample_id in train_ids else "validation"
        pair = pairs[sample_id]
        if sample_id == train_ids[0]:
            device_sample = prepared_train.sample
        else:
            prepared = _prepare_item_for_use(
                _prepare_object_static(pair["perturbed"], model_config),
                model_config,
                torch_device,
                cache_on_device=False,
                decode_images=True,
            )
            device_sample = prepared.sample
        current = _np(pair["perturbed"]["vertices"])
        control = _np(pair["control"]["vertices"])
        faces = _np(pair["perturbed"]["faces"]).astype(np.int64)
        gt = Mesh(
            _np(pair["perturbed"]["gt_vertices"]),
            _np(pair["perturbed"]["gt_faces"]).astype(np.int64),
        ).ensure_normals()
        initial = Mesh(current, faces).ensure_normals()
        paired = Mesh(control, faces).ensure_normals()
        initial_row = _geometry_row(
            sample_id, split, "initial", initial, initial, paired, gt, solver_config
        )
        metric_rows.append(initial_row)
        for method, model in models.items():
            prediction, target_loss = _predict_and_loss(
                model,
                device_sample,
                torch.as_tensor(targets[sample_id][method], device=torch_device),
                training_config,
                loss_kwargs,
                torch_device,
            )
            loss_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "method": method,
                    "target_loss": target_loss,
                }
            )
            refined, recovery = recover_prediction(
                method,
                current,
                faces,
                prediction,
                local_edge_length=targets[sample_id]["local_edge_length"],
                scale=1.0,
                solver_config=solver_config,
                eps=eps,
            )
            refined_meshes[sample_id][method] = refined
            mesh_dir = output / "meshes" / split / sample_id
            mesh_dir.mkdir(parents=True, exist_ok=True)
            save_mesh(refined, mesh_dir / f"{method}.obj")
            np.savez_compressed(
                mesh_dir / f"{method}_prediction.npz",
                prediction=prediction,
                target=targets[sample_id][method],
                local_edge_length=targets[sample_id]["local_edge_length"],
            )
            row = _geometry_row(
                sample_id, split, method, refined, initial, paired, gt, solver_config
            )
            row["target_loss"] = target_loss
            row["solver"] = recovery["solver"]
            row["solver_final_terms"] = recovery["final_terms"]
            metric_rows.append(row)
        if sample_id != train_ids[0]:
            del prepared
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    _write_csv(output / "target_losses.csv", loss_rows)
    _write_json(output / "per_mesh_metrics.json", metric_rows)

    zero_scale_rows = []
    train_id = train_ids[0]
    current = _np(pairs[train_id]["perturbed"]["vertices"])
    faces = _np(pairs[train_id]["perturbed"]["faces"]).astype(np.int64)
    for method in METHODS:
        zero_mesh, recovery = recover_prediction(
            method,
            current,
            faces,
            targets[train_id][method],
            local_edge_length=targets[train_id]["local_edge_length"],
            scale=0.0,
            solver_config=solver_config,
            eps=eps,
        )
        displacement = np.linalg.norm(zero_mesh.vertices - current, axis=1)
        zero_scale_rows.append(
            {
                "method": method,
                "max_displacement": float(displacement.max()),
                "rms_displacement": float(np.sqrt(np.mean(displacement**2))),
                "solver": recovery["solver"],
            }
        )
    _write_csv(output / "scale_zero_checks.csv", zero_scale_rows)

    visual_records = _render_comparisons(
        output,
        pairs,
        refined_meshes,
        cameras,
        train_id,
        render_backend,
    )
    _write_json(output / "visualizations" / "render_metadata.json", visual_records)

    summary = _summarize(
        experiment,
        roundtrip_rows,
        zero_scale_rows,
        loss_rows,
        metric_rows,
        runtime_rows,
    )
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _neutral_initial_state(
    config: Mapping[str, Any], checkpoint_path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    final = model.predictor.output_mlp[-1]
    if not isinstance(final, torch.nn.Linear) or final.out_features != 3:
        raise TypeError("Expected a final three-channel linear prediction layer.")
    torch.nn.init.zeros_(final.weight)
    torch.nn.init.zeros_(final.bias)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_one_method(
    method: str,
    sample: Mapping[str, Any],
    target: torch.Tensor,
    config: Mapping[str, Any],
    initial_state: Mapping[str, torch.Tensor],
    steps: int,
    device: torch.device,
    loss_kwargs: Mapping[str, Any],
) -> tuple[torch.nn.Module, list[dict[str, float]], float, float, float | None]:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False).to(device)
    model.load_state_dict(initial_state)
    training = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    amp_enabled, amp_dtype = _amp_settings(training, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    confidence = torch.ones(target.shape[0], dtype=torch.float32, device=device)
    target = target.float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    model.train()
    with torch.no_grad():
        initial_prediction = model(sample).predicted_laplacian.float()
        initial_loss = weighted_robust_laplacian_loss(
            initial_prediction, target, confidence, **loss_kwargs
        )
    history.append({"step": 0.0, "target_loss": float(initial_loss.item())})
    best_loss = float(initial_loss.item())
    best_state = copy.deepcopy(model.state_dict())
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction = model(sample).predicted_laplacian
        loss = weighted_robust_laplacian_loss(
            prediction.float(), target, confidence, **loss_kwargs
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {method} loss at step {step}.")
        value = float(loss.detach().item())
        # This loss belongs to the pre-update state. Save that exact state so
        # a small target cannot select an Adam-overshot post-update checkpoint.
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step == steps or step % 10 == 0:
            history.append({"step": float(step), "target_loss": value})
            print(f"{method}: step={step:04d}/{steps} target_loss={value:.8g}", flush=True)
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            final_prediction = model(sample).predicted_laplacian
        final_loss = weighted_robust_laplacian_loss(
            final_prediction.float(), target, confidence, **loss_kwargs
        )
    final_value = float(final_loss.item())
    history.append({"step": float(steps) + 0.5, "target_loss": final_value})
    if final_value < best_loss:
        best_loss = final_value
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    runtime = float(time.perf_counter() - start)
    peak = None
    if device.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return model, history, best_loss, runtime, peak


@torch.no_grad()
def _predict_and_loss(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    target: torch.Tensor,
    training: Mapping[str, Any],
    loss_kwargs: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    amp_enabled, amp_dtype = _amp_settings(training, device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        prediction = model(sample).predicted_laplacian
    prediction = prediction.float()
    confidence = torch.ones(target.shape[0], dtype=torch.float32, device=device)
    loss = weighted_robust_laplacian_loss(
        prediction, target.float(), confidence, **loss_kwargs
    )
    return prediction.cpu().numpy(), float(loss.item())


def _geometry_row(
    sample_id: str,
    split: str,
    method: str,
    mesh: Mesh,
    initial: Mesh,
    paired: Mesh,
    gt: Mesh,
    solver_config: Mapping[str, Any],
) -> dict[str, Any]:
    displacement = np.linalg.norm(mesh.vertices - initial.vertices, axis=1)
    paired_error = np.linalg.norm(mesh.vertices - paired.vertices, axis=1)
    topology = _topology_change(initial, mesh)
    triangles = mesh.vertices[mesh.faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    initial_triangles = initial.vertices[initial.faces]
    initial_double_area = np.linalg.norm(
        np.cross(
            initial_triangles[:, 1] - initial_triangles[:, 0],
            initial_triangles[:, 2] - initial_triangles[:, 0],
        ),
        axis=1,
    )
    samples = int(solver_config.get("chamfer_samples", 3000))
    seed = int(solver_config.get("metric_seed", 7))
    paired_chamfer = _chamfer_distance(mesh, paired, samples, seed)
    return {
        "sample_id": sample_id,
        "split": split,
        "method": method,
        "vertex_rms_to_paired_target": float(np.sqrt(np.mean(paired_error**2))),
        "vertex_mae_to_paired_target": float(np.mean(paired_error)),
        "chamfer_to_paired_target": float(paired_chamfer),
        "chamfer_to_gt_context": float(_chamfer_distance(mesh, gt, samples, seed)),
        "normal_consistency_to_paired_target": float(_normal_consistency(mesh, paired)),
        "mean_displacement_from_initial": float(displacement.mean()),
        "max_displacement_from_initial": float(displacement.max()),
        "rms_displacement_from_initial": float(np.sqrt(np.mean(displacement**2))),
        **topology,
        "degenerate_triangles": int(np.sum(double_area <= 1e-14)),
        "initial_degenerate_triangles": int(np.sum(initial_double_area <= 1e-14)),
    }


def _summarize(
    experiment: Mapping[str, Any],
    roundtrips: Sequence[Mapping[str, Any]],
    zero_checks: Sequence[Mapping[str, Any]],
    losses: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    runtimes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, Any]] = {}
    initial_by_split = {
        split: [row for row in metrics if row["method"] == "initial" and row["split"] == split]
        for split in ("train", "validation")
    }
    for method in METHODS:
        aggregates[method] = {}
        for split in ("train", "validation"):
            selected = [row for row in metrics if row["method"] == method and row["split"] == split]
            selected_losses = [row["target_loss"] for row in losses if row["method"] == method and row["split"] == split]
            initial = initial_by_split[split]
            aggregates[method][split] = {
                "mesh_count": len(selected),
                "mean_target_loss": float(np.mean(selected_losses)),
                "mean_vertex_rms_to_paired_target": float(np.mean([row["vertex_rms_to_paired_target"] for row in selected])),
                "mean_chamfer_to_paired_target": float(np.mean([row["chamfer_to_paired_target"] for row in selected])),
                "mean_chamfer_to_gt_context": float(np.mean([row["chamfer_to_gt_context"] for row in selected])),
                "mean_normal_consistency_to_paired_target": float(np.mean([row["normal_consistency_to_paired_target"] for row in selected])),
                "mean_displacement_from_initial": float(np.mean([row["mean_displacement_from_initial"] for row in selected])),
                "maximum_displacement_from_initial": float(np.max([row["max_displacement_from_initial"] for row in selected])),
                "introduced_flipped_triangles": int(sum(row["introduced_flipped_triangles"] for row in selected)),
                "newly_degenerate_triangles": int(sum(row["newly_degenerate_triangles"] for row in selected)),
                "meshes_improving_vertex_rms": int(sum(row["vertex_rms_to_paired_target"] < base["vertex_rms_to_paired_target"] for row, base in zip(selected, initial))),
                "meshes_improving_chamfer": int(sum(row["chamfer_to_paired_target"] < base["chamfer_to_paired_target"] for row, base in zip(selected, initial))),
                "initial_mean_vertex_rms": float(np.mean([row["vertex_rms_to_paired_target"] for row in initial])),
                "initial_mean_chamfer": float(np.mean([row["chamfer_to_paired_target"] for row in initial])),
            }
    train_rank = sorted(METHODS, key=lambda method: aggregates[method]["train"]["mean_vertex_rms_to_paired_target"])
    val_rank = sorted(METHODS, key=lambda method: aggregates[method]["validation"]["mean_vertex_rms_to_paired_target"])
    return {
        **dict(experiment),
        "roundtrip": {
            "max_absolute_error": max(row["max_absolute_error"] for row in roundtrips),
            "max_relative_l2_error": max(row["relative_l2_error"] for row in roundtrips),
            "max_h_difference_from_prepared_current_graph": max(row["max_h_difference_from_prepared_current_graph"] for row in roundtrips),
            "passed": all(row["relative_l2_error"] <= 1e-12 for row in roundtrips),
        },
        "scale_zero": {
            "max_displacement": max(row["max_displacement"] for row in zero_checks),
            "passed": all(row["max_displacement"] <= 1e-12 for row in zero_checks),
        },
        "aggregates": aggregates,
        "train_vertex_rms_ranking": train_rank,
        "validation_vertex_rms_ranking": val_rank,
        "runtime": list(runtimes),
    }


def _report(summary: Mapping[str, Any]) -> str:
    aggregates = summary["aggregates"]
    lines = [
        "# Sofa50 residual-target comparison",
        "",
        "This is a short controlled one-mesh overfit diagnostic. The supervision is perturbed-expanded → paired control-expanded with identical ordering/connectivity; **control expanded is not claimed as GT**. The other four Sofa50 pairs are held out. No Thingi10K data and no long training were used.",
        "",
        f"Each method used {summary['optimizer_steps_per_method']} Adam steps, the same pretrained multi-view/query/graph backbone, the same neutral zeroed output head, RGB inputs, cameras, loss settings, and recovery settings where applicable.",
        "",
        "## Summary",
        "",
        "Values are `train overfit / held-out validation mean`; validation flips are totals over four meshes. Target losses have different physical scales and must not be compared directly across representations.",
        "",
        "| Method                           | Target loss | Vertex RMS | Chamfer | Flips | Improves initial? |",
        "| -------------------------------- | ----------: | ---------: | ------: | ----: | ----------------- |",
    ]
    for method in METHODS:
        train = aggregates[method]["train"]
        val = aggregates[method]["validation"]
        improves = (
            f"train {'yes' if train['meshes_improving_vertex_rms'] == 1 and train['meshes_improving_chamfer'] == 1 else 'no'}; "
            f"val {min(val['meshes_improving_vertex_rms'], val['meshes_improving_chamfer'])}/4"
        )
        lines.append(
            f"| {DISPLAY_NAMES[method]:32s} | {train['mean_target_loss']:.6g} / {val['mean_target_loss']:.6g} | "
            f"{train['mean_vertex_rms_to_paired_target']:.6g} / {val['mean_vertex_rms_to_paired_target']:.6g} | "
            f"{train['mean_chamfer_to_paired_target']:.6g} / {val['mean_chamfer_to_paired_target']:.6g} | "
            f"{train['introduced_flipped_triangles']} / {val['introduced_flipped_triangles']} | {improves} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Raw → normalized → raw: max relative L2 error `{summary['roundtrip']['max_relative_l2_error']:.3e}`, max absolute error `{summary['roundtrip']['max_absolute_error']:.3e}`; {'passed' if summary['roundtrip']['passed'] else 'FAILED'}.",
            f"- `h` was recomputed from each current perturbed-expanded graph. Max difference from its prepared current-graph value: `{summary['roundtrip']['max_h_difference_from_prepared_current_graph']:.3e}`.",
            f"- `scale=0` recovery: maximum displacement `{summary['scale_zero']['max_displacement']:.3e}`; {'passed' if summary['scale_zero']['passed'] else 'FAILED'}.",
            "",
            "## Additional geometry",
            "",
        ]
    )
    for method in METHODS:
        train = aggregates[method]["train"]
        val = aggregates[method]["validation"]
        lines.append(
            f"- {DISPLAY_NAMES[method]}: train/validation normal consistency "
            f"`{train['mean_normal_consistency_to_paired_target']:.6f} / {val['mean_normal_consistency_to_paired_target']:.6f}`; "
            f"mean displacement `{train['mean_displacement_from_initial']:.6g} / {val['mean_displacement_from_initial']:.6g}`; "
            f"maximum displacement `{train['maximum_displacement_from_initial']:.6g} / {val['maximum_displacement_from_initial']:.6g}`; "
            f"new degeneracies `{train['newly_degenerate_triangles']} / {val['newly_degenerate_triangles']}`; "
            f"GT-context Chamfer `{train['mean_chamfer_to_gt_context']:.6g} / {val['mean_chamfer_to_gt_context']:.6g}`."
        )
    direct = aggregates[DIRECT]
    raw = aggregates[RAW]
    h2 = aggregates[H2]
    train_best = DISPLAY_NAMES[summary["train_vertex_rms_ranking"][0]]
    val_best = DISPLAY_NAMES[summary["validation_vertex_rms_ranking"][0]]
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"1. Direct correction learning: **{'yes' if direct['train']['meshes_improving_vertex_rms'] else 'no'}** on the overfit mesh.",
            f"2. Raw Laplacian correction learning: **{'yes' if raw['train']['meshes_improving_vertex_rms'] else 'no'}** on the overfit mesh.",
            f"3. h² normalization improves raw Laplacian geometry: **{'yes' if h2['train']['mean_vertex_rms_to_paired_target'] < raw['train']['mean_vertex_rms_to_paired_target'] else 'no'}** on train and **{'yes' if h2['validation']['mean_vertex_rms_to_paired_target'] < raw['validation']['mean_vertex_rms_to_paired_target'] else 'no'}** on held-out mean.",
            f"4. Best geometry by vertex RMS: **{train_best}** on train; **{val_best}** on held-out mean.",
            f"5. Measurable Laplacian benefit over direct displacement: **{'yes' if min(raw['train']['mean_vertex_rms_to_paired_target'], h2['train']['mean_vertex_rms_to_paired_target']) < direct['train']['mean_vertex_rms_to_paired_target'] else 'no'}** on the overfit mesh; inspect held-out values before treating this as reliable.",
            f"6. Measurable h² benefit over raw: **{'yes' if h2['train']['mean_vertex_rms_to_paired_target'] < raw['train']['mean_vertex_rms_to_paired_target'] else 'no'}** on train by vertex RMS.",
            f"7. Recovery-stage spike/flip evidence: direct introduced `{direct['train']['introduced_flipped_triangles']}` train flips, raw `{raw['train']['introduced_flipped_triangles']}`, h² `{h2['train']['introduced_flipped_triangles']}`. A Laplacian-only excess supports recovery amplification; similar counts do not.",
            f"8. Simplest formulation that worked on the overfit mesh: **{train_best}**. **None demonstrated held-out reliability**; if later runs are statistically tied, prefer direct displacement because it removes the solver.",
            "",
            "Visual comparisons use identical fixed cameras under `visualizations/`. Per-mesh metrics, target losses, predictions, recovered meshes, checkpoints, and histories are stored alongside this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_comparisons(
    output: Path,
    pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined: Mapping[str, Mapping[str, Mesh]],
    cameras: Mapping[str, Mapping[str, Any]],
    train_id: str,
    backend: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    colors = {
        "initial": (180, 205, 220),
        DIRECT: (105, 170, 225),
        RAW: (225, 150, 110),
        H2: (145, 190, 130),
        "paired_control": (180, 220, 180),
        "gt_context": (230, 220, 180),
    }
    for sample_id, pair in pairs.items():
        faces = _np(pair["perturbed"]["faces"]).astype(np.int64)
        meshes = {
            "initial": Mesh(_np(pair["perturbed"]["vertices"]), faces).ensure_normals(),
            **refined[sample_id],
            "paired_control": Mesh(_np(pair["control"]["vertices"]), faces).ensure_normals(),
            "gt_context": Mesh(
                _np(pair["perturbed"]["gt_vertices"]),
                _np(pair["perturbed"]["gt_faces"]).astype(np.int64),
            ).ensure_normals(),
        }
        views = tuple(cameras[sample_id]) if sample_id == train_id else ("perspective",)
        for view in views:
            entries = []
            for name in ("initial", DIRECT, RAW, H2, "paired_control", "gt_context"):
                panel = output / "visualizations" / sample_id / f"{name}_{view}.png"
                _render_panel(
                    meshes[name], cameras[sample_id][view], panel,
                    f"{sample_id} | {DISPLAY_NAMES.get(name, name)} | {view}",
                    colors[name], backend,
                )
                entries.append((name, panel))
                records.append({"sample_id": sample_id, "view": view, "method": name, "path": str(panel)})
            _contact_sheet(entries, output / "visualizations" / sample_id / f"comparison_{view}.png")
    return records


def _contact_sheet(entries: Sequence[tuple[str, Path]], path: Path) -> None:
    cell = 480
    sheet = Image.new("RGB", (3 * cell, 2 * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel_path) in enumerate(entries):
        with Image.open(panel_path) as opened:
            panel = opened.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        x, y = (index % 3) * cell, (index // 3) * cell
        sheet.paste(panel, (x, y))
        draw.rectangle((x, y, x + cell, y + 24), fill=(255, 255, 255))
        draw.text((x + 5, y + 5), DISPLAY_NAMES.get(label, label), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _load_pairs(source: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    root = source / "manifests"
    for variant in ("control", "perturbed"):
        for path in sorted((root / f"prepared_{variant}").glob("*.pt")):
            result.setdefault(path.stem, {})[variant] = load_prepared_sample(
                path, materialize_images=False, dataset_root=root
            )
    if len(result) != 5 or any(set(pair) != {"control", "perturbed"} for pair in result.values()):
        raise ValueError("Expected exactly five Sofa50 control/perturbed expanded pairs.")
    return result


def _assert_same_topology(
    perturbed: Mapping[str, Any], control: Mapping[str, Any], sample_id: str
) -> None:
    if _np(perturbed["vertices"]).shape != _np(control["vertices"]).shape:
        raise ValueError(f"Vertex order/count mismatch for {sample_id}.")
    if not np.array_equal(_np(perturbed["faces"]), _np(control["faces"])):
        raise ValueError(f"Face topology/order mismatch for {sample_id}.")


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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

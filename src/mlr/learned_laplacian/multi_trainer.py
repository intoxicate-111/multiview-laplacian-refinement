from __future__ import annotations

import copy
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dataset import move_sample_to_device, validate_sample
from .graph_layers import faces_to_edge_index
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .model import LearnedLaplacianModel
from .target_scaling import (
    EDGE_SCALE_DEFINITION,
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    TARGET_MODES,
    denormalize_laplacian_by_edge_scale,
    normalize_laplacian_by_edge_scale,
)
from .trainer import _resolve_device, _seed_everything


@dataclass
class MultiObjectTrainingResult:
    model: LearnedLaplacianModel
    history: list[dict[str, float | int | None]]
    best_epoch: int
    best_selection_loss: float
    final_train_loss: float
    final_validation_loss: float | None
    per_object_metrics: dict[str, dict[str, dict[str, Any]]]
    optimizer_steps: int
    device: str
    runtime_seconds: float
    peak_gpu_memory_mb: float | None
    target_mode: str


@dataclass(frozen=True)
class _PreparedObject:
    sample: dict[str, Any]
    training_target: torch.Tensor
    clipped_target_vertices: int


def train_multi_object(
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    device_override: str | None = None,
    input_mode_override: str | None = None,
    zero_images: bool = False,
    progress: bool = True,
) -> MultiObjectTrainingResult:
    """Train one shared model over ragged mesh samples, one mesh forward at a time."""

    if len(train_samples) < 1:
        raise ValueError("train_samples must contain at least one mesh.")
    validation_samples = validation_samples or ()
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    device = _resolve_device(device_override or str(config.get("device", "cpu")))
    target_mode, epsilon = _target_settings(config)
    model = _build_model(config, input_mode_override, zero_images).to(device)
    training = config.get("training", {})
    multi = config.get("multi_object_training", {})
    epochs = int(multi.get("epochs", 1))
    accumulation_meshes = int(multi.get("gradient_accumulation_meshes", 1))
    validation_every = int(multi.get("validation_every_epochs", 1))
    checkpoint_every = int(multi.get("checkpoint_every_epochs", 0))
    if epochs < 1 or accumulation_meshes < 1 or validation_every < 1:
        raise ValueError(
            "epochs, gradient_accumulation_meshes, and validation_every_epochs must be positive."
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    loss_kwargs = _loss_kwargs(training)
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    history: list[dict[str, float | int | None]] = []
    best_epoch = 0
    best_selection_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    optimizer_steps = 0
    shuffle = bool(multi.get("shuffle", True))

    for epoch in range(1, epochs + 1):
        order = list(range(len(train_samples)))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)
        model.train()
        mesh_losses: list[float] = []
        for group_start in range(0, len(order), accumulation_meshes):
            group = order[group_start : group_start + accumulation_meshes]
            optimizer.zero_grad(set_to_none=True)
            for sample_index in group:
                prepared = _prepare_object(train_samples[sample_index], config, device)
                prediction = model(prepared.sample).predicted_laplacian
                loss = weighted_robust_laplacian_loss(
                    prediction,
                    prepared.training_target,
                    prepared.sample["target_confidence"],
                    **loss_kwargs,
                )
                if not torch.isfinite(loss):
                    sample_id = prepared.sample["sample_id"]
                    raise FloatingPointError(
                        f"Training produced a non-finite loss for {sample_id!r} at epoch {epoch}."
                    )
                (loss / len(group)).backward()
                mesh_losses.append(float(loss.detach().item()))
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer_steps += 1

        train_loss = float(np.mean(mesh_losses))
        should_validate = bool(validation_samples) and (
            epoch == 1 or epoch == epochs or epoch % validation_every == 0
        )
        validation_loss = None
        if should_validate:
            validation_loss, _ = _evaluate_dataset(
                model, validation_samples, config, device, loss_kwargs
            )
        selection_loss = validation_loss if validation_samples else train_loss
        if selection_loss is not None and selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if output_path is not None:
                _save_multi_checkpoint(
                    output_path / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    validation_loss,
                    config,
                    len(train_samples),
                    len(validation_samples),
                )
        record: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "optimizer_steps": optimizer_steps,
        }
        history.append(record)
        if progress:
            val_text = "n/a" if validation_loss is None else f"{validation_loss:.8f}"
            print(
                f"epoch={epoch:04d} train={train_loss:.8f} validation={val_text} "
                f"best={best_selection_loss:.8f}",
                flush=True,
            )
        if output_path is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            _save_multi_checkpoint(
                output_path / f"checkpoint_epoch_{epoch:06d}.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                validation_loss,
                config,
                len(train_samples),
                len(validation_samples),
            )

    model.load_state_dict(best_state)
    model.eval()
    predictions_path = None if output_path is None else output_path / "predictions"
    final_train_loss, train_metrics = _evaluate_dataset(
        model,
        train_samples,
        config,
        device,
        loss_kwargs,
        None if predictions_path is None else predictions_path / "train",
    )
    final_validation_loss = None
    validation_metrics: dict[str, dict[str, Any]] = {}
    if validation_samples:
        final_validation_loss, validation_metrics = _evaluate_dataset(
            model,
            validation_samples,
            config,
            device,
            loss_kwargs,
            None if predictions_path is None else predictions_path / "validation",
        )
    runtime_seconds = float(time.perf_counter() - start_time)
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    per_object_metrics = {"train": train_metrics, "validation": validation_metrics}
    if output_path is not None:
        (output_path / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        summary = {
            "best_epoch": best_epoch,
            "best_selection_loss": best_selection_loss,
            "final_train_loss": final_train_loss,
            "final_validation_loss": final_validation_loss,
            "optimizer_steps": optimizer_steps,
            "train_meshes": len(train_samples),
            "validation_meshes": len(validation_samples),
            "target_mode": target_mode,
            "target_scaling_epsilon": epsilon,
            "device": str(device),
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "per_object_metrics": per_object_metrics,
        }
        (output_path / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return MultiObjectTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_selection_loss=best_selection_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        per_object_metrics=per_object_metrics,
        optimizer_steps=optimizer_steps,
        device=str(device),
        runtime_seconds=runtime_seconds,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        target_mode=target_mode,
    )


def _build_model(
    config: Mapping[str, Any], input_mode_override: str | None, zero_images: bool
) -> LearnedLaplacianModel:
    image_config = config.get("image_encoder", {})
    model_config = config.get("model", {})
    return LearnedLaplacianModel(
        image_feature_dim=int(image_config.get("feature_dim", 32)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_graph_layers=int(model_config.get("num_graph_layers", 3)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_mode=input_mode_override or str(config.get("input_mode", "coarse_plus_multiview")),
        zero_images=zero_images,
    )


def _target_settings(config: Mapping[str, Any]) -> tuple[str, float]:
    target_mode = str(config.get("target_mode", RAW_LAPLACIAN))
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(TARGET_MODES)}.")
    scaling = config.get("target_scaling", {})
    method = str(scaling.get("method", EDGE_SCALE_DEFINITION))
    if method != EDGE_SCALE_DEFINITION:
        raise ValueError(f"Unsupported target scaling method: {method}.")
    epsilon = float(scaling.get("epsilon", 1e-12))
    if epsilon <= 0:
        raise ValueError("target_scaling.epsilon must be positive.")
    return target_mode, epsilon


def _prepare_object(
    sample: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> _PreparedObject:
    device_sample = move_sample_to_device(validate_sample(sample), device)
    edge_index = faces_to_edge_index(
        device_sample["faces"], device_sample["vertices"].shape[0]
    )
    degree = device_sample["vertices"].new_zeros((device_sample["vertices"].shape[0], 1))
    if edge_index.numel() > 0:
        degree.index_add_(
            0,
            edge_index[1],
            torch.ones((edge_index.shape[1], 1), dtype=degree.dtype, device=device),
        )
    device_sample["edge_index"] = edge_index
    device_sample["vertex_degree"] = degree
    target_mode, epsilon = _target_settings(config)
    target = device_sample["raw_laplacian_target"]
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        target = normalize_laplacian_by_edge_scale(
            target,
            device_sample["local_edge_length"],
            eps=epsilon,
            valid_scale_mask=device_sample["valid_scale_mask"],
        )
    clip_max_norm = config.get("target_scaling", {}).get("clip_max_norm")
    clipped_count = 0
    if clip_max_norm is not None:
        clip_max_norm = float(clip_max_norm)
        if clip_max_norm <= 0:
            raise ValueError("target_scaling.clip_max_norm must be positive when enabled.")
        magnitudes = torch.linalg.vector_norm(target, dim=-1)
        clipped = magnitudes > clip_max_norm
        clipped_count = int(clipped.sum().item())
        factors = (clip_max_norm / magnitudes.clamp_min(1e-12)).clamp_max(1.0)
        target = target * factors.unsqueeze(-1)
    return _PreparedObject(device_sample, target, clipped_count)


def _loss_kwargs(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_type": str(training.get("loss", "huber")),
        "huber_delta": float(training.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training.get("charbonnier_epsilon", 1e-3)),
    }


@torch.no_grad()
def _evaluate_dataset(
    model: LearnedLaplacianModel,
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
    loss_kwargs: Mapping[str, Any],
    prediction_dir: Path | None = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    model.eval()
    target_mode, _ = _target_settings(config)
    losses = []
    metrics: dict[str, dict[str, Any]] = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        prepared = _prepare_object(sample, config, device)
        prediction = model(prepared.sample).predicted_laplacian
        loss = weighted_robust_laplacian_loss(
            prediction,
            prepared.training_target,
            prepared.sample["target_confidence"],
            **loss_kwargs,
        )
        loss_value = float(loss.item())
        losses.append(loss_value)
        valid_mask = prepared.sample["valid_scale_mask"]
        target_metrics = laplacian_prediction_metrics(
            prediction, prepared.training_target, valid_mask=valid_mask
        )
        if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
            raw_prediction = denormalize_laplacian_by_edge_scale(
                prediction, prepared.sample["local_edge_length"]
            )
        else:
            raw_prediction = prediction
        raw_metrics = laplacian_prediction_metrics(
            raw_prediction,
            prepared.sample["raw_laplacian_target"],
            valid_mask=valid_mask,
        )
        sample_id = str(prepared.sample["sample_id"])
        if sample_id in metrics:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in one dataset split.")
        metrics[sample_id] = {
            "loss": loss_value,
            "vertex_count": int(prepared.sample["vertices"].shape[0]),
            "face_count": int(prepared.sample["faces"].shape[0]),
            "view_count": int(prepared.sample["images"].shape[0]),
            "clipped_target_vertices": prepared.clipped_target_vertices,
            "target_space": target_metrics,
            "recovered_raw_space": raw_metrics,
        }
        if prediction_dir is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
            np.save(
                prediction_dir / f"{safe_id}_target_space_delta.npy",
                prediction.detach().cpu().numpy(),
            )
            np.save(
                prediction_dir / f"{safe_id}_raw_delta.npy",
                raw_prediction.detach().cpu().numpy(),
            )
    if not losses:
        raise ValueError("Cannot evaluate an empty dataset split.")
    return float(np.mean(losses)), metrics


def _save_multi_checkpoint(
    path: Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float | None,
    config: Mapping[str, Any],
    train_meshes: int,
    validation_meshes: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_config": model.architecture_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment_config": dict(config),
            "train_meshes": train_meshes,
            "validation_meshes": validation_meshes,
        },
        path,
    )

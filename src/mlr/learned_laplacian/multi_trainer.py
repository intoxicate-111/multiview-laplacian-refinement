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
    static_preparation_seconds: float = 0.0
    device_cache_seconds: float = 0.0
    mean_epoch_train_seconds: float = 0.0
    mean_validation_seconds: float = 0.0
    initial_loading_seconds: float = 0.0
    initial_learning_rate: float = 0.0
    final_learning_rate: float = 0.0
    lr_scheduler_type: str = "none"
    lr_reduction_count: int = 0


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
    initial_loading_seconds: float = 0.0,
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
    cache_on_device = bool(multi.get("cache_prepared_samples_on_device", False))
    profile_training = bool(multi.get("profile_training", False))
    if epochs < 1 or accumulation_meshes < 1 or validation_every < 1:
        raise ValueError(
            "epochs, gradient_accumulation_meshes, and validation_every_epochs must be positive."
        )
    initial_learning_rate = float(training.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = _build_lr_scheduler(optimizer, training)
    lr_scheduler_type = "none" if scheduler is None else "reduce_on_plateau"
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    loss_kwargs = _loss_kwargs(training)
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    static_start = time.perf_counter()
    prepared_train = tuple(
        _prepare_object_static(_static_sample_at(train_samples, index), config)
        for index in range(len(train_samples))
    )
    prepared_validation = tuple(
        _prepare_object_static(_static_sample_at(validation_samples, index), config)
        for index in range(len(validation_samples))
    )
    static_preparation_seconds = float(time.perf_counter() - static_start)
    device_cache_seconds = 0.0
    if cache_on_device and any(
        item.sample.get("prepared_storage_format") == "lazy_image_paths_v1"
        for item in (*prepared_train, *prepared_validation)
    ):
        raise ValueError(
            "lazy_image_paths_v1 requires cache_prepared_samples_on_device=false"
        )
    if cache_on_device:
        cache_start = time.perf_counter()
        try:
            prepared_train = tuple(
                _move_prepared_object_to_device(item, device) for item in prepared_train
            )
            prepared_validation = tuple(
                _move_prepared_object_to_device(item, device) for item in prepared_validation
            )
            _synchronize_device(device)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "CUDA ran out of memory while caching prepared samples. Set "
                "multi_object_training.cache_prepared_samples_on_device to false "
                "to keep the static cache on CPU."
            ) from error
        device_cache_seconds = float(time.perf_counter() - cache_start)
    if progress and profile_training:
        print(f"static preparation: {static_preparation_seconds:.2f}s", flush=True)
        print(f"device cache: {device_cache_seconds:.2f}s", flush=True)
        print(f"prepared train meshes: {len(prepared_train)}", flush=True)
        print(f"prepared validation meshes: {len(prepared_validation)}", flush=True)
        print(f"device: {device}", flush=True)

    history: list[dict[str, float | int | None]] = []
    best_epoch = 0
    best_selection_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    optimizer_steps = 0
    shuffle = bool(multi.get("shuffle", True))
    epoch_train_seconds: list[float] = []
    epoch_validation_seconds: list[float] = []
    lr_reduction_count = 0

    for epoch in range(1, epochs + 1):
        _synchronize_device(device)
        train_start = time.perf_counter()
        order = list(range(len(prepared_train)))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)
        model.train()
        mesh_loss_tensors: list[torch.Tensor] = []
        for group_start in range(0, len(order), accumulation_meshes):
            group = order[group_start : group_start + accumulation_meshes]
            optimizer.zero_grad(set_to_none=True)
            for sample_index in group:
                prepared = _prepare_item_for_use(
                    prepared_train[sample_index], config, device, cache_on_device
                )
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
                mesh_loss_tensors.append(loss.detach())
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer_steps += 1

        train_loss = float(torch.stack(mesh_loss_tensors).mean().item())
        _synchronize_device(device)
        train_seconds = float(time.perf_counter() - train_start)
        epoch_train_seconds.append(train_seconds)
        should_validate = bool(prepared_validation) and (
            epoch == 1 or epoch == epochs or epoch % validation_every == 0
        )
        validation_loss = None
        validation_seconds = 0.0
        if should_validate:
            _synchronize_device(device)
            validation_start = time.perf_counter()
            validation_loss, _ = _evaluate_dataset(
                model,
                prepared_validation,
                config,
                device,
                loss_kwargs,
                cache_on_device=cache_on_device,
            )
            _synchronize_device(device)
            validation_seconds = float(time.perf_counter() - validation_start)
            epoch_validation_seconds.append(validation_seconds)
        selection_loss = validation_loss if prepared_validation else train_loss
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
                    len(prepared_train),
                    len(prepared_validation),
                )
        lr_before_scheduler = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None and validation_loss is not None:
            scheduler.step(validation_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_was_reduced = current_lr < lr_before_scheduler
        if lr_was_reduced:
            lr_reduction_count += 1
        record: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "learning_rate": current_lr,
        }
        history.append(record)
        if progress:
            val_text = "n/a" if validation_loss is None else f"{validation_loss:.8f}"
            print(
                f"epoch={epoch:04d} train={train_loss:.8f} validation={val_text} "
                f"best={best_selection_loss:.8f} lr={current_lr:.8e}",
                flush=True,
            )
            if lr_was_reduced:
                print(
                    "learning rate reduced: "
                    f"{lr_before_scheduler:.8e} -> {current_lr:.8e}",
                    flush=True,
                )
            if profile_training:
                print(
                    f"timing train={train_seconds:.2f}s validation={validation_seconds:.2f}s",
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
                len(prepared_train),
                len(prepared_validation),
            )

    model.load_state_dict(best_state)
    model.eval()
    predictions_path = None if output_path is None else output_path / "predictions"
    final_train_loss, train_metrics = _evaluate_dataset(
        model,
        prepared_train,
        config,
        device,
        loss_kwargs,
        None if predictions_path is None else predictions_path / "train",
        cache_on_device=cache_on_device,
    )
    final_validation_loss = None
    validation_metrics: dict[str, dict[str, Any]] = {}
    if prepared_validation:
        final_validation_loss, validation_metrics = _evaluate_dataset(
            model,
            prepared_validation,
            config,
            device,
            loss_kwargs,
            None if predictions_path is None else predictions_path / "validation",
            cache_on_device=cache_on_device,
        )
    _synchronize_device(device)
    runtime_seconds = float(initial_loading_seconds + time.perf_counter() - start_time)
    mean_epoch_train_seconds = float(np.mean(epoch_train_seconds))
    mean_validation_seconds = (
        float(np.mean(epoch_validation_seconds)) if epoch_validation_seconds else 0.0
    )
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    if progress and profile_training:
        peak_text = "n/a" if peak_gpu_memory_mb is None else f"{peak_gpu_memory_mb:.2f} MB"
        print(f"mean train epoch time: {mean_epoch_train_seconds:.2f}s", flush=True)
        print(f"mean validation time: {mean_validation_seconds:.2f}s", flush=True)
        print(f"peak GPU memory: {peak_text}", flush=True)
        print(f"total runtime: {runtime_seconds:.2f}s", flush=True)
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
            "train_meshes": len(prepared_train),
            "validation_meshes": len(prepared_validation),
            "target_mode": target_mode,
            "target_scaling_epsilon": epsilon,
            "device": str(device),
            "runtime_seconds": runtime_seconds,
            "initial_loading_seconds": float(initial_loading_seconds),
            "static_preparation_seconds": static_preparation_seconds,
            "device_cache_seconds": device_cache_seconds,
            "mean_epoch_train_seconds": mean_epoch_train_seconds,
            "mean_validation_seconds": mean_validation_seconds,
            "initial_learning_rate": initial_learning_rate,
            "final_learning_rate": final_learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_reduction_count": lr_reduction_count,
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
        static_preparation_seconds=static_preparation_seconds,
        device_cache_seconds=device_cache_seconds,
        mean_epoch_train_seconds=mean_epoch_train_seconds,
        mean_validation_seconds=mean_validation_seconds,
        initial_loading_seconds=float(initial_loading_seconds),
        initial_learning_rate=initial_learning_rate,
        final_learning_rate=final_learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        lr_reduction_count=lr_reduction_count,
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


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_config = training_config.get("lr_scheduler", {})
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("training.lr_scheduler must be an object.")
    scheduler_type = str(scheduler_config.get("type", "none"))
    if scheduler_type == "none":
        return None
    if scheduler_type != "reduce_on_plateau":
        raise ValueError(f"Unsupported lr_scheduler type: {scheduler_type!r}.")

    factor = float(scheduler_config.get("factor", 0.5))
    patience = int(scheduler_config.get("patience_validations", 10))
    threshold = float(scheduler_config.get("threshold", 1e-4))
    threshold_mode = str(scheduler_config.get("threshold_mode", "abs"))
    cooldown = int(scheduler_config.get("cooldown_validations", 0))
    min_lr = float(scheduler_config.get("min_lr", 1e-6))
    if not 0.0 < factor < 1.0:
        raise ValueError("lr_scheduler.factor must be between 0 and 1.")
    if patience < 0:
        raise ValueError("lr_scheduler.patience_validations must be non-negative.")
    if threshold < 0:
        raise ValueError("lr_scheduler.threshold must be non-negative.")
    if threshold_mode not in {"rel", "abs"}:
        raise ValueError("lr_scheduler.threshold_mode must be 'rel' or 'abs'.")
    if cooldown < 0:
        raise ValueError("lr_scheduler.cooldown_validations must be non-negative.")
    if min_lr < 0:
        raise ValueError("lr_scheduler.min_lr must be non-negative.")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
    )


def _prepare_object_static(
    sample: Mapping[str, Any], config: Mapping[str, Any]
) -> _PreparedObject:
    static_sample = (
        dict(sample) if sample.get("_static_prepared") is True else validate_sample(sample)
    )
    target_mode, epsilon = _target_settings(config)
    target = static_sample["raw_laplacian_target"]
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        prepared_epsilon = float(
            static_sample.get("metadata", {}).get("edge_scale_epsilon", 1e-12)
        )
        if prepared_epsilon == epsilon:
            target = static_sample["normalized_laplacian_target"]
        else:
            target = normalize_laplacian_by_edge_scale(
                target,
                static_sample["local_edge_length"],
                eps=epsilon,
                valid_scale_mask=static_sample["valid_scale_mask"],
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
    if static_sample.get("prepared_storage_format") == "lazy_image_paths_v1":
        static_sample.pop("images", None)
    return _PreparedObject(static_sample, target, clipped_count)


def _static_sample_at(samples: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    load_static = getattr(samples, "load_static", None)
    return load_static(index) if callable(load_static) else samples[index]


def _move_prepared_object_to_device(
    prepared: _PreparedObject, device: torch.device
) -> _PreparedObject:
    moved_sample = move_sample_to_device(prepared.sample, device)
    moved_target = prepared.training_target.to(device)
    for name in ("raw_laplacian_target", "normalized_laplacian_target"):
        if prepared.training_target is prepared.sample.get(name):
            moved_target = moved_sample[name]
            break
    return _PreparedObject(
        sample=moved_sample,
        training_target=moved_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
    )


def _prepared_for_use(
    prepared: _PreparedObject, device: torch.device, cache_on_device: bool
) -> _PreparedObject:
    if cache_on_device:
        return prepared
    return _move_prepared_object_to_device(prepared, device)


def _prepare_item_for_use(
    item: Mapping[str, Any] | _PreparedObject,
    config: Mapping[str, Any],
    device: torch.device,
    cache_on_device: bool,
) -> _PreparedObject:
    prepared = item if isinstance(item, _PreparedObject) else _prepare_object_static(item, config)
    if "images" not in prepared.sample and prepared.sample.get("image_paths"):
        from .sample_io import load_and_resize_images

        dataset_root = Path(str(prepared.sample["_dataset_root"]))
        image_paths = [
            Path(value) if Path(value).is_absolute() else dataset_root / value
            for value in prepared.sample["image_paths"]
        ]
        images, _ = load_and_resize_images(
            image_paths, int(prepared.sample["prepared_image_size"])
        )
        materialized_sample = dict(prepared.sample)
        materialized_sample["images"] = images
        prepared = _PreparedObject(
            materialized_sample,
            prepared.training_target,
            prepared.clipped_target_vertices,
        )
    return _prepared_for_use(prepared, device, cache_on_device)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_kwargs(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_type": str(training.get("loss", "huber")),
        "huber_delta": float(training.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training.get("charbonnier_epsilon", 1e-3)),
    }


@torch.no_grad()
def _evaluate_dataset(
    model: LearnedLaplacianModel,
    samples: Sequence[Mapping[str, Any] | _PreparedObject],
    config: Mapping[str, Any],
    device: torch.device,
    loss_kwargs: Mapping[str, Any],
    prediction_dir: Path | None = None,
    cache_on_device: bool = True,
) -> tuple[float, dict[str, dict[str, Any]]]:
    model.eval()
    target_mode, _ = _target_settings(config)
    losses = []
    metrics: dict[str, dict[str, Any]] = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for item in samples:
        prepared = _prepare_item_for_use(item, config, device, cache_on_device)
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

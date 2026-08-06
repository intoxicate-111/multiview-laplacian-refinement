from __future__ import annotations

import copy
import json
import re
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from .dataset import validate_sample
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .model import LearnedLaplacianModel
from .query_training import (
    QUERY_FOURIER_GEOMETRY_MODE,
    QueryAugmentationSettings,
    apply_query_augmentation,
    query_augmentation_settings,
    validate_gt_query_contract,
)
from .sample_io import load_and_resize_images
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
    completed_epochs: int = 0
    stopped_early: bool = False
    stop_reason: str = "max_epochs"
    amp_enabled: bool = False
    peak_cpu_memory_mb: float = 0.0
    mean_epoch_data_loading_seconds: float = 0.0
    mean_epoch_gpu_transfer_seconds: float = 0.0
    mean_epoch_forward_backward_seconds: float = 0.0
    mean_optimizer_step_seconds: float = 0.0


@dataclass(frozen=True)
class _PreparedObject:
    sample: dict[str, Any]
    training_target: torch.Tensor
    clipped_target_vertices: int


@dataclass(frozen=True)
class _DataLoaderSettings:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int


@dataclass(frozen=True)
class _EarlyStoppingSettings:
    enabled: bool
    patience_validations: int
    min_delta: float


class _MaterializedPreparedDataset(Dataset[dict[str, Any]]):
    """Materialize only the image tensor requested by a DataLoader worker."""

    def __init__(self, items: Sequence[_PreparedObject]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prepared = _materialize_prepared_images(self.items[index], dtype=torch.uint8)
        return {
            "sample": prepared.sample,
            "training_target": prepared.training_target,
            "clipped_target_vertices": prepared.clipped_target_vertices,
        }


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
    query_settings = query_augmentation_settings(config)
    if query_settings.enabled and str(
        config.get("model", {}).get("geometry_mode", "legacy")
    ) != QUERY_FOURIER_GEOMETRY_MODE:
        raise ValueError(
            "query_training.enabled=true requires model.geometry_mode='query_fourier'."
        )
    model = _build_model(config, input_mode_override, zero_images).to(device)
    training = config.get("training", {})
    multi = config.get("multi_object_training", {})
    epochs = int(multi.get("epochs", 1))
    accumulation_meshes = int(multi.get("gradient_accumulation_meshes", 1))
    validation_every = int(multi.get("validation_every_epochs", 1))
    checkpoint_every = int(multi.get("checkpoint_every_epochs", 0))
    max_optimizer_steps_value = multi.get("max_optimizer_steps")
    max_optimizer_steps = (
        None if max_optimizer_steps_value is None else int(max_optimizer_steps_value)
    )
    cache_on_device = bool(multi.get("cache_prepared_samples_on_device", False))
    profile_training = bool(multi.get("profile_training", False))
    early_stopping = _early_stopping_settings(multi)
    loader_settings = _data_loader_settings(config)
    if early_stopping.enabled and not validation_samples:
        raise ValueError("Early stopping requires at least one validation sample.")
    if epochs < 1 or accumulation_meshes < 1 or validation_every < 1:
        raise ValueError(
            "epochs, gradient_accumulation_meshes, and validation_every_epochs must be positive."
        )
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be positive when provided.")
    initial_learning_rate = float(training.get("learning_rate", 1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = _build_lr_scheduler(optimizer, training)
    lr_scheduler_type = "none" if scheduler is None else "reduce_on_plateau"
    amp_enabled, amp_dtype = _amp_settings(training, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
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
                _move_prepared_object_to_device(item, device, config=config)
                for item in prepared_train
            )
            prepared_validation = tuple(
                _move_prepared_object_to_device(item, device, config=config)
                for item in prepared_validation
            )
            _synchronize_device(device)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "CUDA ran out of memory while caching prepared samples. Set "
                "multi_object_training.cache_prepared_samples_on_device to false "
                "to keep the static cache on CPU."
            ) from error
        device_cache_seconds = float(time.perf_counter() - cache_start)
    train_generator = torch.Generator()
    train_loader = None
    validation_loader = None
    if not cache_on_device:
        train_loader = _build_prepared_loader(
            prepared_train,
            loader_settings,
            shuffle=bool(multi.get("shuffle", True)),
            generator=train_generator,
        )
        if prepared_validation:
            validation_loader = _build_prepared_loader(
                prepared_validation,
                loader_settings,
                shuffle=False,
            )
    if progress and profile_training:
        print(f"static preparation: {static_preparation_seconds:.2f}s", flush=True)
        print(f"device cache: {device_cache_seconds:.2f}s", flush=True)
        print(f"prepared train meshes: {len(prepared_train)}", flush=True)
        print(f"prepared validation meshes: {len(prepared_validation)}", flush=True)
        print(f"device: {device}", flush=True)
        print(
            "data loader: "
            f"workers={loader_settings.num_workers} "
            f"pin_memory={loader_settings.pin_memory} "
            f"persistent_workers={loader_settings.persistent_workers} "
            f"prefetch_factor={loader_settings.prefetch_factor}",
            flush=True,
        )
        print(f"AMP: {amp_enabled} ({amp_dtype})", flush=True)

    history: list[dict[str, float | int | None]] = []
    best_epoch = 0
    best_selection_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    optimizer_steps = 0
    shuffle = bool(multi.get("shuffle", True))
    epoch_train_seconds: list[float] = []
    epoch_validation_seconds: list[float] = []
    epoch_data_loading_seconds: list[float] = []
    epoch_gpu_transfer_seconds: list[float] = []
    epoch_forward_backward_seconds: list[float] = []
    lr_reduction_count = 0
    early_stopping_best = float("inf")
    early_stopping_bad_validations = 0
    stopped_early = False
    stop_reason = "max_epochs"

    for epoch in range(1, epochs + 1):
        _synchronize_device(device)
        train_start = time.perf_counter()
        train_generator.manual_seed(seed + epoch)
        if train_loader is None:
            order = torch.randperm(len(prepared_train), generator=train_generator).tolist()
            if not shuffle:
                order = list(range(len(prepared_train)))
            epoch_items: Iterable[Any] = (prepared_train[index] for index in order)
        else:
            epoch_items = train_loader
        item_iterator = iter(epoch_items)
        model.train()
        mesh_loss_tensors: list[torch.Tensor] = []
        exact_query_loss_tensors: list[torch.Tensor] = []
        perturbed_query_loss_tensors: list[torch.Tensor] = []
        data_loading_seconds = 0.0
        gpu_transfer_seconds = 0.0
        forward_backward_seconds = 0.0
        steps_at_epoch_start = optimizer_steps
        transfer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        reached_max_steps = False
        while True:
            if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                reached_max_steps = True
                break
            group: list[_PreparedObject] = []
            for _ in range(accumulation_meshes):
                loading_start = time.perf_counter()
                try:
                    item = next(item_iterator)
                except StopIteration:
                    data_loading_seconds += time.perf_counter() - loading_start
                    break
                data_loading_seconds += time.perf_counter() - loading_start
                group.append(_prepared_from_loader_item(item))
            if not group:
                break
            optimizer.zero_grad(set_to_none=True)
            for cpu_prepared in group:
                transfer_start = time.perf_counter()
                transfer_event = _start_cuda_timing(device)
                prepared = _prepare_item_for_use(
                    cpu_prepared,
                    config,
                    device,
                    cache_on_device,
                    non_blocking=loader_settings.pin_memory,
                )
                prepared = _with_query_augmentation(
                    prepared,
                    query_settings,
                    base_seed=seed,
                    epoch=epoch,
                    enabled=query_settings.enabled,
                )
                transfer_events.append(_finish_cuda_timing(device, transfer_event))
                if device.type != "cuda":
                    gpu_transfer_seconds += time.perf_counter() - transfer_start
                forward_start = time.perf_counter()
                forward_event = _start_cuda_timing(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    prediction = model(prepared.sample).predicted_laplacian
                prediction_fp32 = prediction.float()
                loss = weighted_robust_laplacian_loss(
                    prediction_fp32,
                    prepared.training_target.float(),
                    prepared.sample["target_confidence"].float(),
                    **loss_kwargs,
                )
                if not torch.isfinite(loss):
                    sample_id = prepared.sample["sample_id"]
                    raise FloatingPointError(
                        f"Training produced a non-finite loss for {sample_id!r} at epoch {epoch}."
                    )
                scaler.scale(loss / len(group)).backward()
                forward_events.append(_finish_cuda_timing(device, forward_event))
                if device.type != "cuda":
                    forward_backward_seconds += time.perf_counter() - forward_start
                mesh_loss_tensors.append(loss.detach())
                with torch.no_grad():
                    exact_loss, perturbed_loss = _query_subset_losses(
                        prediction_fp32,
                        prepared.training_target.float(),
                        prepared.sample["target_confidence"].float(),
                        prepared.sample.get("query_is_exact"),
                        loss_kwargs,
                    )
                if exact_loss is not None:
                    exact_query_loss_tensors.append(exact_loss.detach())
                if perturbed_loss is not None:
                    perturbed_query_loss_tensors.append(perturbed_loss.detach())
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1

        train_loss = float(torch.stack(mesh_loss_tensors).mean().item())
        train_exact_query_loss = _mean_optional_tensors(exact_query_loss_tensors)
        train_perturbed_query_loss = _mean_optional_tensors(perturbed_query_loss_tensors)
        _synchronize_device(device)
        if device.type == "cuda":
            gpu_transfer_seconds = _elapsed_cuda_seconds(transfer_events)
            forward_backward_seconds = _elapsed_cuda_seconds(forward_events)
        train_seconds = float(time.perf_counter() - train_start)
        epoch_train_seconds.append(train_seconds)
        epoch_data_loading_seconds.append(data_loading_seconds)
        epoch_gpu_transfer_seconds.append(gpu_transfer_seconds)
        epoch_forward_backward_seconds.append(forward_backward_seconds)
        should_validate = bool(prepared_validation) and (
            epoch == 1
            or epoch == epochs
            or epoch % validation_every == 0
            or reached_max_steps
        )
        validation_loss = None
        validation_seconds = 0.0
        if should_validate:
            _synchronize_device(device)
            validation_start = time.perf_counter()
            validation_loss, validation_epoch_metrics = _evaluate_dataset(
                model,
                prepared_validation,
                config,
                device,
                loss_kwargs,
                cache_on_device=cache_on_device,
                data_loader=validation_loader,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                query_settings=query_settings,
                query_seed=seed,
                query_epoch=epoch,
                augment_queries=query_settings.enabled
                and query_settings.apply_to_validation,
            )
            validation_exact_query_loss = _mean_metric(
                validation_epoch_metrics, "exact_query_loss"
            )
            validation_perturbed_query_loss = _mean_metric(
                validation_epoch_metrics, "perturbed_query_loss"
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
        if validation_loss is not None and early_stopping.enabled:
            if validation_loss < early_stopping_best - early_stopping.min_delta:
                early_stopping_best = validation_loss
                early_stopping_bad_validations = 0
            else:
                early_stopping_bad_validations += 1
                if early_stopping_bad_validations >= early_stopping.patience_validations:
                    stopped_early = True
                    stop_reason = "early_stopping"
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
            "data_loading_seconds": data_loading_seconds,
            "gpu_transfer_seconds": gpu_transfer_seconds,
            "forward_backward_seconds": forward_backward_seconds,
            "total_step_seconds": train_seconds,
            "optimizer_steps_this_epoch": optimizer_steps - steps_at_epoch_start,
            "learning_rate": current_lr,
            "train_exact_query_loss": train_exact_query_loss,
            "train_perturbed_query_loss": train_perturbed_query_loss,
            "validation_exact_query_loss": (
                validation_exact_query_loss if should_validate else None
            ),
            "validation_perturbed_query_loss": (
                validation_perturbed_query_loss if should_validate else None
            ),
        }
        history.append(record)
        if progress:
            val_text = "n/a" if validation_loss is None else f"{validation_loss:.8f}"
            print(
                f"epoch={epoch:04d} train={train_loss:.8f} validation={val_text} "
                f"best={best_selection_loss:.8f} lr={current_lr:.8e}",
                flush=True,
            )
            if train_exact_query_loss is not None:
                print(
                    "query loss "
                    f"exact={train_exact_query_loss:.8f} "
                    f"perturbed={train_perturbed_query_loss:.8f}",
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
                    f"timing data={data_loading_seconds:.2f}s "
                    f"transfer={gpu_transfer_seconds:.2f}s "
                    f"forward_backward={forward_backward_seconds:.2f}s "
                    f"steps_total={train_seconds:.2f}s "
                    f"validation={validation_seconds:.2f}s",
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
        if stopped_early:
            if progress:
                print(
                    "early stopping: "
                    f"{early_stopping_bad_validations} validations without sufficient improvement",
                    flush=True,
                )
            break
        if reached_max_steps:
            stop_reason = "max_optimizer_steps"
            if progress:
                print(f"reached max optimizer steps: {optimizer_steps}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    predictions_path = None if output_path is None else output_path / "predictions"
    # Reuse persistent training workers for the final metric pass. Metric
    # aggregation is keyed by sample_id, so shuffled evaluation order is safe.
    final_train_loader = train_loader
    final_train_loss, train_metrics = _evaluate_dataset(
        model,
        prepared_train,
        config,
        device,
        loss_kwargs,
        None if predictions_path is None else predictions_path / "train",
        cache_on_device=cache_on_device,
        data_loader=final_train_loader,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        query_settings=query_settings,
        query_seed=seed,
        query_epoch=0,
        augment_queries=query_settings.enabled,
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
            data_loader=validation_loader,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            query_settings=query_settings,
            query_seed=seed,
            query_epoch=0,
            augment_queries=query_settings.enabled
            and query_settings.apply_to_validation,
        )
    _synchronize_device(device)
    runtime_seconds = float(initial_loading_seconds + time.perf_counter() - start_time)
    mean_epoch_train_seconds = float(np.mean(epoch_train_seconds))
    mean_validation_seconds = (
        float(np.mean(epoch_validation_seconds)) if epoch_validation_seconds else 0.0
    )
    mean_epoch_data_loading_seconds = float(np.mean(epoch_data_loading_seconds))
    mean_epoch_gpu_transfer_seconds = float(np.mean(epoch_gpu_transfer_seconds))
    mean_epoch_forward_backward_seconds = float(
        np.mean(epoch_forward_backward_seconds)
    )
    mean_optimizer_step_seconds = (
        float(sum(epoch_train_seconds) / optimizer_steps) if optimizer_steps else 0.0
    )
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    peak_cpu_memory_mb = _peak_cpu_memory_mb()
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    if progress and profile_training:
        peak_text = "n/a" if peak_gpu_memory_mb is None else f"{peak_gpu_memory_mb:.2f} MB"
        print(f"mean train epoch time: {mean_epoch_train_seconds:.2f}s", flush=True)
        print(f"mean validation time: {mean_validation_seconds:.2f}s", flush=True)
        print(
            "mean train timing: "
            f"data={mean_epoch_data_loading_seconds:.2f}s "
            f"transfer={mean_epoch_gpu_transfer_seconds:.2f}s "
            f"forward_backward={mean_epoch_forward_backward_seconds:.2f}s "
            f"per_optimizer_step={mean_optimizer_step_seconds:.2f}s",
            flush=True,
        )
        print(f"peak CPU memory: {peak_cpu_memory_mb:.2f} MB", flush=True)
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
            "completed_epochs": len(history),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "amp_enabled": amp_enabled,
            "mean_epoch_data_loading_seconds": mean_epoch_data_loading_seconds,
            "mean_epoch_gpu_transfer_seconds": mean_epoch_gpu_transfer_seconds,
            "mean_epoch_forward_backward_seconds": mean_epoch_forward_backward_seconds,
            "mean_optimizer_step_seconds": mean_optimizer_step_seconds,
            "peak_cpu_memory_mb": peak_cpu_memory_mb,
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
        completed_epochs=len(history),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        amp_enabled=amp_enabled,
        peak_cpu_memory_mb=peak_cpu_memory_mb,
        mean_epoch_data_loading_seconds=mean_epoch_data_loading_seconds,
        mean_epoch_gpu_transfer_seconds=mean_epoch_gpu_transfer_seconds,
        mean_epoch_forward_backward_seconds=mean_epoch_forward_backward_seconds,
        mean_optimizer_step_seconds=mean_optimizer_step_seconds,
    )


def _build_model(
    config: Mapping[str, Any], input_mode_override: str | None, zero_images: bool
) -> LearnedLaplacianModel:
    image_config = config.get("image_encoder", {})
    model_config = config.get("model", {})
    position_config = model_config.get("position_encoding", {})
    if not isinstance(position_config, Mapping):
        raise ValueError("model.position_encoding must be an object.")
    return LearnedLaplacianModel(
        image_feature_dim=int(image_config.get("feature_dim", 32)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_graph_layers=int(model_config.get("num_graph_layers", 3)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_mode=input_mode_override or str(config.get("input_mode", "coarse_plus_multiview")),
        zero_images=zero_images,
        geometry_mode=str(model_config.get("geometry_mode", "legacy")),
        position_num_frequencies=int(position_config.get("num_frequencies", 6)),
        position_include_input=bool(position_config.get("include_input", True)),
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
    if query_augmentation_settings(config).enabled:
        validate_gt_query_contract(static_sample)
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
    prepared: _PreparedObject,
    device: torch.device,
    *,
    config: Mapping[str, Any],
    non_blocking: bool = False,
) -> _PreparedObject:
    moved_sample: dict[str, Any] = {}
    for name, value in prepared.sample.items():
        if not isinstance(value, torch.Tensor):
            moved_sample[name] = value
            continue
        if name == "images" and value.dtype == torch.uint8:
            moved_sample[name] = value.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ).div_(255.0)
        else:
            moved_sample[name] = value.to(device, non_blocking=non_blocking)
    if "images" in moved_sample:
        moved_sample["images"] = _normalize_images(moved_sample["images"], config)
    moved_target = prepared.training_target.to(device, non_blocking=non_blocking)
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
    prepared: _PreparedObject,
    device: torch.device,
    cache_on_device: bool,
    config: Mapping[str, Any],
    non_blocking: bool,
) -> _PreparedObject:
    if cache_on_device:
        return prepared
    return _move_prepared_object_to_device(
        prepared,
        device,
        config=config,
        non_blocking=non_blocking,
    )


def _prepare_item_for_use(
    item: Mapping[str, Any] | _PreparedObject,
    config: Mapping[str, Any],
    device: torch.device,
    cache_on_device: bool,
    non_blocking: bool = False,
) -> _PreparedObject:
    prepared = item if isinstance(item, _PreparedObject) else _prepare_object_static(item, config)
    if "images" not in prepared.sample and prepared.sample.get("image_paths"):
        prepared = _materialize_prepared_images(prepared, dtype=torch.uint8)
    return _prepared_for_use(
        prepared,
        device,
        cache_on_device,
        config,
        non_blocking,
    )


def _materialize_prepared_images(
    prepared: _PreparedObject,
    *,
    dtype: torch.dtype,
) -> _PreparedObject:
    if "images" in prepared.sample or not prepared.sample.get("image_paths"):
        return prepared
    dataset_root = Path(str(prepared.sample["_dataset_root"]))
    image_paths = [
        Path(value) if Path(value).is_absolute() else dataset_root / value
        for value in prepared.sample["image_paths"]
    ]
    images, _ = load_and_resize_images(
        image_paths,
        int(prepared.sample["prepared_image_size"]),
        dtype=dtype,
    )
    materialized_sample = dict(prepared.sample)
    materialized_sample["images"] = images
    return _PreparedObject(
        materialized_sample,
        prepared.training_target,
        prepared.clipped_target_vertices,
    )


def _prepared_from_loader_item(item: Any) -> _PreparedObject:
    if isinstance(item, _PreparedObject):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("Prepared DataLoader items must be mappings.")
    sample = item.get("sample")
    training_target = item.get("training_target")
    if not isinstance(sample, Mapping) or not isinstance(training_target, torch.Tensor):
        raise TypeError("Prepared DataLoader items require sample and training_target.")
    return _PreparedObject(
        dict(sample),
        training_target,
        int(item.get("clipped_target_vertices", 0)),
    )


def _build_prepared_loader(
    items: Sequence[_PreparedObject],
    settings: _DataLoaderSettings,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = _MaterializedPreparedDataset(items)
    sampler = (
        RandomSampler(dataset, generator=generator)
        if shuffle
        else SequentialSampler(dataset)
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "sampler": sampler,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "persistent_workers": settings.persistent_workers,
    }
    if settings.num_workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return DataLoader(**kwargs)


def _normalize_images(images: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    if not images.is_floating_point():
        raise TypeError("Images must be floating point after device transfer.")
    normalization = config.get("image_encoder", {}).get("normalization", {})
    mean_values = normalization.get("mean", [0.0, 0.0, 0.0])
    std_values = normalization.get("std", [1.0, 1.0, 1.0])
    if len(mean_values) != 3 or len(std_values) != 3:
        raise ValueError("image normalization mean and std must contain three values.")
    if any(float(value) <= 0 for value in std_values):
        raise ValueError("image normalization std values must be positive.")
    mean = images.new_tensor(mean_values).view(1, 3, 1, 1)
    std = images.new_tensor(std_values).view(1, 3, 1, 1)
    return (images - mean) / std


def _data_loader_settings(config: Mapping[str, Any]) -> _DataLoaderSettings:
    loading = config.get("data_loading", {})
    num_workers = int(loading.get("num_workers", 0))
    prefetch_factor = int(loading.get("prefetch_factor", 2))
    pin_memory = bool(loading.get("pin_memory", False))
    persistent_workers = bool(loading.get("persistent_workers", False))
    if num_workers < 0:
        raise ValueError("data_loading.num_workers must be non-negative.")
    if prefetch_factor < 1:
        raise ValueError("data_loading.prefetch_factor must be positive.")
    if persistent_workers and num_workers == 0:
        raise ValueError(
            "data_loading.persistent_workers requires data_loading.num_workers > 0."
        )
    return _DataLoaderSettings(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _early_stopping_settings(
    multi_config: Mapping[str, Any],
) -> _EarlyStoppingSettings:
    raw = multi_config.get("early_stopping", {})
    if not isinstance(raw, Mapping):
        raise ValueError("multi_object_training.early_stopping must be an object.")
    enabled = bool(raw.get("enabled", False))
    patience = int(raw.get("patience_validations", 10))
    min_delta = float(raw.get("min_delta", 0.0))
    if patience < 1:
        raise ValueError("early_stopping.patience_validations must be positive.")
    if min_delta < 0:
        raise ValueError("early_stopping.min_delta must be non-negative.")
    return _EarlyStoppingSettings(enabled, patience, min_delta)


def _amp_settings(
    training_config: Mapping[str, Any], device: torch.device
) -> tuple[bool, torch.dtype]:
    raw = training_config.get("amp", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training.amp must be an object.")
    requested = bool(raw.get("enabled", False))
    dtype_name = str(raw.get("dtype", "float16"))
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError("training.amp.dtype must be 'float16' or 'bfloat16'.")
    return requested and device.type == "cuda", dtypes[dtype_name]


def _start_cuda_timing(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _finish_cuda_timing(
    device: torch.device, start: torch.cuda.Event | None
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if device.type != "cuda" or start is None:
        return None, None
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    return start, end


def _elapsed_cuda_seconds(
    events: Iterable[tuple[torch.cuda.Event | None, torch.cuda.Event | None]],
) -> float:
    return float(
        sum(
            start.elapsed_time(end)
            for start, end in events
            if start is not None and end is not None
        )
        / 1000.0
    )


def _peak_cpu_memory_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_kwargs(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_type": str(training.get("loss", "huber")),
        "huber_delta": float(training.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training.get("charbonnier_epsilon", 1e-3)),
    }


def _with_query_augmentation(
    prepared: _PreparedObject,
    settings: QueryAugmentationSettings,
    *,
    base_seed: int,
    epoch: int,
    enabled: bool,
) -> _PreparedObject:
    if not settings.enabled:
        return prepared
    effective = (
        settings
        if enabled
        else QueryAugmentationSettings(
            enabled=False,
            exact_fraction=settings.exact_fraction,
            normal_std_h=settings.normal_std_h,
            tangent_std_h=settings.tangent_std_h,
            max_offset_h=settings.max_offset_h,
            apply_to_validation=settings.apply_to_validation,
            zero_initial_laplacian=settings.zero_initial_laplacian,
        )
    )
    return _PreparedObject(
        sample=apply_query_augmentation(
            prepared.sample, effective, base_seed=base_seed, epoch=epoch
        ),
        training_target=prepared.training_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
    )


def _query_subset_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    exact_mask: torch.Tensor | None,
    loss_kwargs: Mapping[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if exact_mask is None:
        return None, None
    exact_mask = exact_mask.to(dtype=torch.bool, device=prediction.device)

    def subset(mask: torch.Tensor) -> torch.Tensor | None:
        weights = confidence * mask.to(dtype=confidence.dtype)
        return weighted_robust_laplacian_loss(
            prediction, target, weights, **loss_kwargs
        )

    return subset(exact_mask), subset(~exact_mask)


def _mean_optional_tensors(values: Sequence[torch.Tensor]) -> float | None:
    return float(torch.stack(tuple(values)).mean().item()) if values else None


def _mean_metric(metrics: Mapping[str, Mapping[str, Any]], name: str) -> float | None:
    values = [float(item[name]) for item in metrics.values() if item.get(name) is not None]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def _evaluate_dataset(
    model: LearnedLaplacianModel,
    samples: Sequence[Mapping[str, Any] | _PreparedObject],
    config: Mapping[str, Any],
    device: torch.device,
    loss_kwargs: Mapping[str, Any],
    prediction_dir: Path | None = None,
    cache_on_device: bool = True,
    data_loader: Iterable[Any] | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    query_settings: QueryAugmentationSettings | None = None,
    query_seed: int = 7,
    query_epoch: int = 0,
    augment_queries: bool = False,
) -> tuple[float, dict[str, dict[str, Any]]]:
    model.eval()
    target_mode, _ = _target_settings(config)
    losses = []
    metrics: dict[str, dict[str, Any]] = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    items = samples if data_loader is None else data_loader
    non_blocking = _data_loader_settings(config).pin_memory
    for item in items:
        prepared = _prepare_item_for_use(
            _prepared_from_loader_item(item),
            config,
            device,
            cache_on_device,
            non_blocking=non_blocking,
        )
        if query_settings is not None:
            prepared = _with_query_augmentation(
                prepared,
                query_settings,
                base_seed=query_seed,
                epoch=query_epoch,
                enabled=augment_queries,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            prediction = model(prepared.sample).predicted_laplacian
        prediction = prediction.float()
        loss = weighted_robust_laplacian_loss(
            prediction,
            prepared.training_target.float(),
            prepared.sample["target_confidence"].float(),
            **loss_kwargs,
        )
        loss_value = float(loss.item())
        exact_loss, perturbed_loss = _query_subset_losses(
            prediction,
            prepared.training_target.float(),
            prepared.sample["target_confidence"].float(),
            prepared.sample.get("query_is_exact"),
            loss_kwargs,
        )
        losses.append(loss_value)
        valid_mask = prepared.sample["valid_scale_mask"]
        target_metrics = laplacian_prediction_metrics(
            prediction, prepared.training_target.float(), valid_mask=valid_mask
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
            "exact_query_loss": None if exact_loss is None else float(exact_loss.item()),
            "perturbed_query_loss": (
                None if perturbed_loss is None else float(perturbed_loss.item())
            ),
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

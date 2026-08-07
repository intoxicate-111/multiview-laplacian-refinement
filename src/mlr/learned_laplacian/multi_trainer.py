from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import validate_sample
from .losses import (
    confidence_calibration_metrics,
    confidence_reliability_loss,
    laplacian_prediction_metrics,
    weighted_robust_laplacian_loss,
)
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
    mean_epoch_image_decode_resize_seconds: float = 0.0
    mean_train_views_per_sample: float = 0.0
    mean_epoch_decoded_image_bytes: float = 0.0


@dataclass(frozen=True)
class _PreparedObject:
    sample: dict[str, Any]
    training_target: torch.Tensor
    clipped_target_vertices: int
    raw_target: torch.Tensor | None = None
    face_count: int = 0
    image_decode_resize_seconds: float = 0.0
    decoded_image_bytes: int = 0
    used_view_count: int = 0


@dataclass(frozen=True)
class _DataLoaderSettings:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    train_views_per_sample: int | None
    validation_views_per_sample: int | None


@dataclass(frozen=True)
class _EarlyStoppingSettings:
    enabled: bool
    patience_validations: int
    min_delta: float


class _MaterializedPreparedDataset(Dataset[dict[str, Any]]):
    """Materialize only the image tensor requested by a DataLoader worker."""

    def __init__(
        self,
        items: Sequence[_PreparedObject],
        *,
        decode_images: bool,
        views_per_sample: int | None,
        base_seed: int,
        profile_loading: bool,
    ) -> None:
        self.items = items
        self.decode_images = decode_images
        self.views_per_sample = views_per_sample
        self.base_seed = int(base_seed)
        self.profile_loading = profile_loading

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, Any]:
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        prepared = _select_prepared_views(
            self.items[index],
            self.views_per_sample,
            base_seed=self.base_seed,
            epoch=epoch,
        )
        if self.decode_images:
            prepared = _materialize_prepared_images(
                prepared,
                dtype=torch.uint8,
                profile_loading=self.profile_loading,
            )
        return {
            "sample": prepared.sample,
            "training_target": prepared.training_target,
            "clipped_target_vertices": prepared.clipped_target_vertices,
            "image_decode_resize_seconds": prepared.image_decode_resize_seconds,
            "decoded_image_bytes": prepared.decoded_image_bytes,
            "used_view_count": prepared.used_view_count,
        }


class _EpochIndexSampler(Sampler[tuple[int, int]]):
    """Send the epoch with each index so persistent workers remain reproducible."""

    def __init__(
        self,
        size: int,
        *,
        shuffle: bool,
        generator: torch.Generator | None,
    ) -> None:
        self.size = int(size)
        self.shuffle = shuffle
        self.generator = generator
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.size, generator=self.generator).tolist()
        else:
            indices = range(self.size)
        return iter((int(index), self.epoch) for index in indices)

    def __len__(self) -> int:
        return self.size


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
    resume_checkpoint: str | Path | None = None,
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
    decode_images = model.input_mode != "coarse_only" and not model.zero_images
    keep_projection = model.input_mode != "coarse_only"
    training = config.get("training", {})
    multi = config.get("multi_object_training", {})
    epochs = int(multi.get("epochs", 1))
    accumulation_meshes = int(multi.get("gradient_accumulation_meshes", 1))
    validation_every = int(multi.get("validation_every_epochs", 1))
    checkpoint_every = int(multi.get("checkpoint_every_epochs", 0))
    checkpoint_optimizer_steps = tuple(
        sorted({int(value) for value in multi.get("checkpoint_optimizer_steps", ())})
    )
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
    if checkpoint_optimizer_steps and checkpoint_optimizer_steps[0] < 0:
        raise ValueError("checkpoint_optimizer_steps cannot contain negative values.")
    if (
        max_optimizer_steps is not None
        and checkpoint_optimizer_steps
        and checkpoint_optimizer_steps[-1] > max_optimizer_steps
    ):
        raise ValueError(
            "checkpoint_optimizer_steps cannot exceed max_optimizer_steps."
        )
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
    confidence_settings = _confidence_settings(config)
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    resume_payload: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_payload.get("optimizer_steps") is None:
            raise ValueError(
                "resume_checkpoint is not an optimizer-step checkpoint."
            )
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if scheduler is not None and resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])

    start_time = time.perf_counter()
    static_start = time.perf_counter()
    prepared_train = tuple(
        _prepare_object_static(
            _static_sample_at(train_samples, index),
            config,
            keep_image_payload=decode_images,
            keep_projection=keep_projection,
        )
        for index in range(len(train_samples))
    )
    prepared_validation = tuple(
        _prepare_object_static(
            _static_sample_at(validation_samples, index),
            config,
            keep_image_payload=decode_images,
            keep_projection=keep_projection,
        )
        for index in range(len(validation_samples))
    )
    static_preparation_seconds = float(time.perf_counter() - static_start)
    device_cache_seconds = 0.0
    if cache_on_device and decode_images and any(
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
            decode_images=decode_images,
            views_per_sample=loader_settings.train_views_per_sample,
            base_seed=seed,
            profile_loading=profile_training,
        )
        if prepared_validation:
            validation_loader = _build_prepared_loader(
                prepared_validation,
                loader_settings,
                shuffle=False,
                decode_images=decode_images,
                views_per_sample=loader_settings.validation_views_per_sample,
                base_seed=seed,
                profile_loading=profile_training,
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
            f"prefetch_factor={loader_settings.prefetch_factor} "
            f"train_views={loader_settings.train_views_per_sample} "
            f"validation_views={loader_settings.validation_views_per_sample} "
            f"decode_images={decode_images}",
            flush=True,
        )
        print(f"AMP: {amp_enabled} ({amp_dtype})", flush=True)

    resume_state = {} if resume_payload is None else dict(
        resume_payload.get("training_state", {})
    )
    history: list[dict[str, float | int | None]] = list(
        resume_state.get("history", [])
    )
    best_epoch = int(resume_state.get("best_epoch", 0))
    best_selection_loss = float(resume_state.get("best_selection_loss", float("inf")))
    stored_best_state = resume_state.get("best_model_state_dict")
    best_state = (
        copy.deepcopy(model.state_dict())
        if stored_best_state is None
        else stored_best_state
    )
    optimizer_steps = int(
        0 if resume_payload is None else resume_payload["optimizer_steps"]
    )
    start_epoch = int(resume_state.get("next_epoch", 1))
    resume_groups_completed = int(
        resume_state.get("groups_completed_in_epoch", 0)
    )
    shuffle = bool(multi.get("shuffle", True))
    epoch_train_seconds: list[float] = []
    epoch_validation_seconds: list[float] = []
    epoch_data_loading_seconds: list[float] = []
    epoch_gpu_transfer_seconds: list[float] = []
    epoch_forward_backward_seconds: list[float] = []
    epoch_image_decode_resize_seconds: list[float] = []
    epoch_mean_used_view_count: list[float] = []
    epoch_decoded_image_bytes: list[int] = []
    lr_reduction_count = int(resume_state.get("lr_reduction_count", 0))
    early_stopping_best = float(
        resume_state.get("early_stopping_best", float("inf"))
    )
    early_stopping_bad_validations = int(
        resume_state.get("early_stopping_bad_validations", 0)
    )
    stopped_early = False
    stop_reason = "max_epochs"

    groups_per_epoch = math.ceil(len(prepared_train) / accumulation_meshes)
    if output_path is not None and 0 in checkpoint_optimizer_steps and optimizer_steps == 0:
        _save_optimizer_step_checkpoint(
            output_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=0,
            optimizer_steps=0,
            train_loss=None,
            validation_loss=None,
            config=config,
            train_meshes=len(prepared_train),
            validation_meshes=len(prepared_validation),
            next_epoch=1,
            groups_completed_in_epoch=0,
            history=history,
            best_epoch=best_epoch,
            best_selection_loss=best_selection_loss,
            best_state=best_state,
            lr_reduction_count=lr_reduction_count,
            early_stopping_best=early_stopping_best,
            early_stopping_bad_validations=early_stopping_bad_validations,
        )

    for epoch in range(start_epoch, epochs + 1):
        _synchronize_device(device)
        train_start = time.perf_counter()
        train_generator.manual_seed(seed + epoch)
        _set_loader_epoch(train_loader, epoch)
        if train_loader is None:
            order = torch.randperm(len(prepared_train), generator=train_generator).tolist()
            if not shuffle:
                order = list(range(len(prepared_train)))
            epoch_items: Iterable[Any] = (prepared_train[index] for index in order)
        else:
            epoch_items = train_loader
        item_iterator = iter(epoch_items)
        groups_completed_in_epoch = (
            resume_groups_completed if epoch == start_epoch else 0
        )
        items_to_skip = groups_completed_in_epoch * accumulation_meshes
        for _ in range(items_to_skip):
            try:
                next(item_iterator)
            except StopIteration as error:
                raise ValueError(
                    "Resume checkpoint groups_completed_in_epoch exceeds the "
                    "deterministic epoch length."
                ) from error
        model.train()
        mesh_loss_tensors: list[torch.Tensor] = []
        objective_tensors: list[torch.Tensor] = []
        confidence_loss_tensors: list[torch.Tensor] = []
        confidence_value_tensors: list[torch.Tensor] = []
        exact_query_loss_tensors: list[torch.Tensor] = []
        perturbed_query_loss_tensors: list[torch.Tensor] = []
        data_loading_seconds = 0.0
        gpu_transfer_seconds = 0.0
        forward_backward_seconds = 0.0
        image_decode_resize_seconds = 0.0
        decoded_image_bytes = 0
        used_view_count = 0
        loaded_mesh_count = 0
        visible_query_count = 0
        invisible_query_count = 0
        steps_at_epoch_start = optimizer_steps
        transfer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        reached_max_steps = False
        boundary_checkpoint_step: int | None = None
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
                loaded = _prepared_from_loader_item(item)
                if train_loader is None:
                    loaded = _select_prepared_views(
                        loaded,
                        loader_settings.train_views_per_sample,
                        base_seed=seed,
                        epoch=epoch,
                    )
                group.append(loaded)
                image_decode_resize_seconds += loaded.image_decode_resize_seconds
                decoded_image_bytes += loaded.decoded_image_bytes
                used_view_count += loaded.used_view_count
                loaded_mesh_count += 1
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
                    decode_images=decode_images,
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
                    model_output = model(prepared.sample)
                prediction_fp32 = model_output.predicted_laplacian.float()
                prediction_loss = weighted_robust_laplacian_loss(
                    prediction_fp32,
                    prepared.training_target.float(),
                    prepared.sample["target_confidence"].float(),
                    **loss_kwargs,
                )
                confidence_loss = None
                objective = prediction_loss
                if confidence_settings["enabled"]:
                    if model_output.confidence_prediction is None:
                        raise RuntimeError(
                            "confidence.enabled requires a model confidence head."
                        )
                    confidence_prediction = model_output.confidence_prediction.float()
                    confidence_loss = confidence_reliability_loss(
                        confidence_prediction,
                        prediction_fp32,
                        prepared.training_target.float(),
                        prepared.sample["target_confidence"].float(),
                        regularizer=confidence_settings["regularizer"],
                        minimum_confidence=confidence_settings["minimum_confidence"],
                        loss_type=str(loss_kwargs["loss_type"]),
                        huber_delta=float(loss_kwargs["huber_delta"]),
                        charbonnier_epsilon=float(
                            loss_kwargs["charbonnier_epsilon"]
                        ),
                    )
                    objective = prediction_loss + confidence_settings["loss_weight"] * confidence_loss
                    confidence_loss_tensors.append(confidence_loss.detach())
                    confidence_value_tensors.append(confidence_prediction.detach())
                if not torch.isfinite(objective):
                    sample_id = prepared.sample["sample_id"]
                    raise FloatingPointError(
                        f"Training produced a non-finite loss for {sample_id!r} at epoch {epoch}."
                    )
                scaler.scale(objective / len(group)).backward()
                forward_events.append(_finish_cuda_timing(device, forward_event))
                if device.type != "cuda":
                    forward_backward_seconds += time.perf_counter() - forward_start
                mesh_loss_tensors.append(prediction_loss.detach())
                objective_tensors.append(objective.detach())
                visible = model_output.valid_views.any(dim=0)
                visible_query_count += int(visible.sum().item())
                invisible_query_count += int((~visible).sum().item())
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
            groups_completed_in_epoch += 1
            if (
                output_path is not None
                and optimizer_steps in checkpoint_optimizer_steps
            ):
                if groups_completed_in_epoch >= groups_per_epoch:
                    boundary_checkpoint_step = optimizer_steps
                else:
                    _save_optimizer_step_checkpoint(
                        output_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch=epoch,
                        optimizer_steps=optimizer_steps,
                        train_loss=float(torch.stack(mesh_loss_tensors).mean().item()),
                        validation_loss=None,
                        config=config,
                        train_meshes=len(prepared_train),
                        validation_meshes=len(prepared_validation),
                        next_epoch=epoch,
                        groups_completed_in_epoch=groups_completed_in_epoch,
                        history=history,
                        best_epoch=best_epoch,
                        best_selection_loss=best_selection_loss,
                        best_state=best_state,
                        lr_reduction_count=lr_reduction_count,
                        early_stopping_best=early_stopping_best,
                        early_stopping_bad_validations=early_stopping_bad_validations,
                    )

        train_loss = float(torch.stack(mesh_loss_tensors).mean().item())
        train_objective = float(torch.stack(objective_tensors).mean().item())
        train_confidence_loss = _mean_optional_tensors(confidence_loss_tensors)
        train_mean_confidence = (
            float(torch.cat(confidence_value_tensors).mean().item())
            if confidence_value_tensors
            else None
        )
        train_exact_query_loss = _mean_optional_tensors(exact_query_loss_tensors)
        train_perturbed_query_loss = _mean_optional_tensors(perturbed_query_loss_tensors)
        mean_image_decode_resize_seconds = (
            image_decode_resize_seconds / loaded_mesh_count if loaded_mesh_count else 0.0
        )
        mean_used_view_count = used_view_count / loaded_mesh_count if loaded_mesh_count else 0.0
        _synchronize_device(device)
        if device.type == "cuda":
            gpu_transfer_seconds = _elapsed_cuda_seconds(transfer_events)
            forward_backward_seconds = _elapsed_cuda_seconds(forward_events)
        train_seconds = float(time.perf_counter() - train_start)
        epoch_train_seconds.append(train_seconds)
        epoch_data_loading_seconds.append(data_loading_seconds)
        epoch_gpu_transfer_seconds.append(gpu_transfer_seconds)
        epoch_forward_backward_seconds.append(forward_backward_seconds)
        epoch_image_decode_resize_seconds.append(image_decode_resize_seconds)
        epoch_mean_used_view_count.append(mean_used_view_count)
        epoch_decoded_image_bytes.append(decoded_image_bytes)
        should_validate = bool(prepared_validation) and (
            epoch == 1
            or epoch == epochs
            or epoch % validation_every == 0
            or reached_max_steps
        )
        validation_loss = None
        validation_seconds = 0.0
        if should_validate:
            _set_loader_epoch(validation_loader, 0)
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
                views_per_sample=loader_settings.validation_views_per_sample,
            )
            validation_exact_query_loss = _mean_metric(
                validation_epoch_metrics, "exact_query_loss"
            )
            validation_perturbed_query_loss = _mean_metric(
                validation_epoch_metrics, "perturbed_query_loss"
            )
            validation_global_cosine = _mean_nested_metric(
                validation_epoch_metrics, "target_space", "global_cosine"
            )
            validation_high10_cosine = _mean_nested_metric(
                validation_epoch_metrics, "target_space", "top_10_percent_cosine"
            )
            validation_norm_ratio = _mean_nested_metric(
                validation_epoch_metrics,
                "target_space",
                "prediction_to_target_norm_ratio",
            )
            validation_mean_confidence = _mean_nested_metric(
                validation_epoch_metrics, "confidence", "mean"
            )
            validation_confidence_error_correlation = _mean_nested_metric(
                validation_epoch_metrics,
                "confidence",
                "correlation_with_negative_error",
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
                for best_name in ("best.pt", "checkpoint_best.pt"):
                    _save_multi_checkpoint(
                        output_path / best_name,
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
            "train_normalized_laplacian_loss": train_loss,
            "train_objective": train_objective,
            "train_confidence_loss": train_confidence_loss,
            "train_mean_confidence": train_mean_confidence,
            "validation_loss": validation_loss,
            "validation_normalized_laplacian_loss": validation_loss,
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "data_loading_seconds": data_loading_seconds,
            "sample_wait_seconds": data_loading_seconds,
            "image_decode_resize_seconds": image_decode_resize_seconds,
            "mean_image_decode_resize_seconds_per_mesh": mean_image_decode_resize_seconds,
            "mean_used_view_count": mean_used_view_count,
            "decoded_image_bytes": decoded_image_bytes,
            "gpu_transfer_seconds": gpu_transfer_seconds,
            "pin_or_transfer_seconds": gpu_transfer_seconds,
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
            "validation_global_cosine": (
                validation_global_cosine if should_validate else None
            ),
            "validation_high_10_percent_cosine": (
                validation_high10_cosine if should_validate else None
            ),
            "validation_prediction_to_gt_norm_ratio": (
                validation_norm_ratio if should_validate else None
            ),
            "validation_mean_confidence": (
                validation_mean_confidence if should_validate else None
            ),
            "validation_confidence_negative_error_correlation": (
                validation_confidence_error_correlation if should_validate else None
            ),
            "visible_query_count": visible_query_count,
            "invisible_query_count": invisible_query_count,
        }
        history.append(record)
        if boundary_checkpoint_step is not None and output_path is not None:
            _save_optimizer_step_checkpoint(
                output_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                optimizer_steps=boundary_checkpoint_step,
                train_loss=train_loss,
                validation_loss=validation_loss,
                config=config,
                train_meshes=len(prepared_train),
                validation_meshes=len(prepared_validation),
                next_epoch=epoch + 1,
                groups_completed_in_epoch=0,
                history=history,
                best_epoch=best_epoch,
                best_selection_loss=best_selection_loss,
                best_state=best_state,
                lr_reduction_count=lr_reduction_count,
                early_stopping_best=early_stopping_best,
                early_stopping_bad_validations=early_stopping_bad_validations,
            )
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
            if train_mean_confidence is not None:
                print(
                    "confidence "
                    f"mean={train_mean_confidence:.6f} "
                    f"loss={train_confidence_loss:.8f} "
                    f"visible={visible_query_count} invisible={invisible_query_count}",
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
                    f"decode={image_decode_resize_seconds:.2f}s "
                    f"transfer={gpu_transfer_seconds:.2f}s "
                    f"forward_backward={forward_backward_seconds:.2f}s "
                    f"steps_total={train_seconds:.2f}s "
                    f"validation={validation_seconds:.2f}s "
                    f"views={mean_used_view_count:.2f} "
                    f"decoded={decoded_image_bytes / (1024.0 * 1024.0):.2f}MB",
                    flush=True,
                )
        checkpoint_epochs = {
            int(value) for value in multi.get("checkpoint_epochs", ())
        }
        if output_path is not None:
            _save_resumable_checkpoint(
                output_path / "checkpoint_latest.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                train_loss=train_loss,
                validation_loss=validation_loss,
                config=config,
                train_meshes=len(prepared_train),
                validation_meshes=len(prepared_validation),
                history=history,
                best_epoch=best_epoch,
                best_selection_loss=best_selection_loss,
                best_state=best_state,
                lr_reduction_count=lr_reduction_count,
                early_stopping_best=early_stopping_best,
                early_stopping_bad_validations=early_stopping_bad_validations,
            )
        if output_path is not None and (
            (checkpoint_every > 0 and epoch % checkpoint_every == 0)
            or epoch in checkpoint_epochs
        ):
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
    _set_loader_epoch(final_train_loader, 0)
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
        views_per_sample=loader_settings.train_views_per_sample,
    )
    final_validation_loss = None
    validation_metrics: dict[str, dict[str, Any]] = {}
    if prepared_validation:
        _set_loader_epoch(validation_loader, 0)
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
            views_per_sample=loader_settings.validation_views_per_sample,
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
    mean_epoch_image_decode_resize_seconds = float(
        np.mean(epoch_image_decode_resize_seconds)
    )
    mean_train_views_per_sample = float(np.mean(epoch_mean_used_view_count))
    mean_epoch_decoded_image_bytes = float(np.mean(epoch_decoded_image_bytes))
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
            f"decode={mean_epoch_image_decode_resize_seconds:.2f}s "
            f"transfer={mean_epoch_gpu_transfer_seconds:.2f}s "
            f"forward_backward={mean_epoch_forward_backward_seconds:.2f}s "
            f"per_optimizer_step={mean_optimizer_step_seconds:.2f}s "
            f"views={mean_train_views_per_sample:.2f} "
            f"decoded={mean_epoch_decoded_image_bytes / (1024.0 * 1024.0):.2f}MB",
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
            "mean_epoch_image_decode_resize_seconds": mean_epoch_image_decode_resize_seconds,
            "mean_train_views_per_sample": mean_train_views_per_sample,
            "mean_epoch_decoded_image_bytes": mean_epoch_decoded_image_bytes,
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
        mean_epoch_image_decode_resize_seconds=mean_epoch_image_decode_resize_seconds,
        mean_train_views_per_sample=mean_train_views_per_sample,
        mean_epoch_decoded_image_bytes=mean_epoch_decoded_image_bytes,
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
        predict_confidence=bool(config.get("confidence", {}).get("enabled", False)),
    )


def _confidence_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("confidence", {})
    if not isinstance(settings, Mapping):
        raise ValueError("confidence must be an object.")
    result = {
        "enabled": bool(settings.get("enabled", False)),
        "loss_weight": float(settings.get("loss_weight", 1.0)),
        "regularizer": float(settings.get("regularizer", 0.01)),
        "minimum_confidence": float(settings.get("minimum_confidence", 1e-4)),
        "quantile_bins": int(settings.get("quantile_bins", 5)),
    }
    if result["loss_weight"] < 0:
        raise ValueError("confidence.loss_weight must be non-negative.")
    if result["regularizer"] <= 0:
        raise ValueError("confidence.regularizer must be positive.")
    if not 0 < result["minimum_confidence"] < 1:
        raise ValueError("confidence.minimum_confidence must be between zero and one.")
    if result["quantile_bins"] < 2:
        raise ValueError("confidence.quantile_bins must be at least two.")
    return result


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
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    keep_image_payload: bool = True,
    keep_projection: bool = True,
) -> _PreparedObject:
    static_sample = (
        dict(sample) if sample.get("_static_prepared") is True else validate_sample(sample)
    )
    static_sample = _select_renderer_visibility(static_sample, config)
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
    raw_target = static_sample["raw_laplacian_target"]
    face_count = int(static_sample["faces"].shape[0])
    if static_sample.get("prepared_storage_format") == "lazy_image_paths_v1":
        static_sample.pop("images", None)
    static_sample = _prune_sample_for_training(
        static_sample,
        keep_image_payload=keep_image_payload,
        keep_projection=keep_projection,
    )
    return _PreparedObject(
        sample=static_sample,
        training_target=target,
        clipped_target_vertices=clipped_count,
        raw_target=raw_target,
        face_count=face_count,
        used_view_count=int(static_sample["num_views"]),
    )


def _select_renderer_visibility(
    sample: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Select one precomputed renderer mask without reconstructing visibility."""

    result = dict(sample)
    settings = config.get("renderer_visibility")
    if settings is None:
        return result
    if not isinstance(settings, Mapping):
        raise ValueError("renderer_visibility must be an object.")
    condition = str(settings.get("condition", "prepared"))
    if condition == "prepared":
        return result
    if condition == "frustum_only":
        result["visibility"] = None
        return result
    fields = {
        "backface_only": "visibility_backface_only",
        "occlusion_only": "visibility_occlusion_only",
        "backface_and_occlusion": "visibility_backface_and_occlusion",
    }
    field = fields.get(condition)
    if field is None:
        raise ValueError(
            "renderer_visibility.condition must be prepared, frustum_only, "
            "backface_only, occlusion_only, or backface_and_occlusion."
        )
    value = result.get(field)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Renderer visibility condition {condition!r} requires sample field {field!r}."
        )
    result["visibility"] = value
    return result


def _prune_sample_for_training(
    sample: Mapping[str, Any],
    *,
    keep_image_payload: bool,
    keep_projection: bool,
) -> dict[str, Any]:
    """Keep only tensors consumed by forward, loss, augmentation, or metrics."""

    fields = {
        "sample_id",
        "vertices",
        "vertex_normals",
        "initial_laplacian",
        "edge_index",
        "vertex_degree",
        "target_confidence",
        "local_edge_length",
        "valid_scale_mask",
        "query_positions",
        "query_is_exact",
        "position_normalization_center",
        "position_normalization_scale",
    }
    if keep_projection:
        fields.update({"intrinsics", "extrinsics", "visibility"})
    if keep_image_payload:
        fields.update(
            {
                "images",
                "image_paths",
                "prepared_image_size",
                "source_image_size",
                "prepared_storage_format",
                "_dataset_root",
            }
        )
    result = {name: sample[name] for name in fields if name in sample}
    images = sample.get("images")
    if isinstance(images, torch.Tensor):
        num_views, _, image_height, image_width = images.shape
    else:
        num_views = int(sample["intrinsics"].shape[0])
        prepared_size = int(sample.get("prepared_image_size", 0))
        if prepared_size < 1:
            source_size = sample.get("source_image_size")
            if not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
                raise ValueError("Samples without images require prepared or source image size.")
            image_width, image_height = (int(source_size[0]), int(source_size[1]))
        else:
            image_height = image_width = prepared_size
    result["image_height"] = int(image_height)
    result["image_width"] = int(image_width)
    result["num_views"] = int(num_views)
    return result


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
    return _PreparedObject(
        sample=moved_sample,
        training_target=moved_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=prepared.raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=prepared.used_view_count,
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
    decode_images: bool = True,
) -> _PreparedObject:
    prepared = item if isinstance(item, _PreparedObject) else _prepare_object_static(item, config)
    if decode_images and "images" not in prepared.sample and prepared.sample.get("image_paths"):
        prepared = _materialize_prepared_images(
            prepared, dtype=torch.uint8, profile_loading=False
        )
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
    profile_loading: bool = False,
) -> _PreparedObject:
    if "images" in prepared.sample or not prepared.sample.get("image_paths"):
        return prepared
    dataset_root = Path(str(prepared.sample["_dataset_root"]))
    image_paths = [
        Path(value) if Path(value).is_absolute() else dataset_root / value
        for value in prepared.sample["image_paths"]
    ]
    decode_start = time.perf_counter() if profile_loading else 0.0
    images, _ = load_and_resize_images(
        image_paths,
        int(prepared.sample["prepared_image_size"]),
        dtype=dtype,
    )
    materialized_sample = dict(prepared.sample)
    materialized_sample["images"] = images
    return _PreparedObject(
        sample=materialized_sample,
        training_target=prepared.training_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=prepared.raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=(
            time.perf_counter() - decode_start if profile_loading else 0.0
        ),
        decoded_image_bytes=images.numel() * images.element_size(),
        used_view_count=int(images.shape[0]),
    )


def _select_prepared_views(
    prepared: _PreparedObject,
    views_per_sample: int | None,
    *,
    base_seed: int,
    epoch: int,
) -> _PreparedObject:
    sample = prepared.sample
    total_views = int(sample["num_views"])
    if views_per_sample is None or views_per_sample >= total_views:
        if prepared.used_view_count == total_views:
            return prepared
        return _replace_prepared(prepared, used_view_count=total_views)
    digest = hashlib.sha256(str(sample["sample_id"]).encode("utf-8")).digest()
    sample_seed = int.from_bytes(digest[:8], "little", signed=False)
    generator = torch.Generator().manual_seed(
        (sample_seed + int(base_seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    )
    indices = torch.randperm(total_views, generator=generator)[:views_per_sample]
    selected = dict(sample)
    if "image_paths" in selected:
        selected["image_paths"] = [selected["image_paths"][int(i)] for i in indices]
    for name in ("images", "intrinsics", "extrinsics", "visibility"):
        value = selected.get(name)
        if isinstance(value, torch.Tensor):
            selected[name] = value.index_select(0, indices.to(value.device))
    selected["num_views"] = int(views_per_sample)
    return _PreparedObject(
        sample=selected,
        training_target=prepared.training_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=prepared.raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=int(views_per_sample),
    )


def _replace_prepared(
    prepared: _PreparedObject,
    *,
    raw_target: torch.Tensor | None = None,
    face_count: int | None = None,
    used_view_count: int | None = None,
) -> _PreparedObject:
    return _PreparedObject(
        sample=prepared.sample,
        training_target=prepared.training_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=prepared.raw_target if raw_target is None else raw_target,
        face_count=prepared.face_count if face_count is None else face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=(
            prepared.used_view_count if used_view_count is None else used_view_count
        ),
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
        sample=dict(sample),
        training_target=training_target,
        clipped_target_vertices=int(item.get("clipped_target_vertices", 0)),
        image_decode_resize_seconds=float(item.get("image_decode_resize_seconds", 0.0)),
        decoded_image_bytes=int(item.get("decoded_image_bytes", 0)),
        used_view_count=int(item.get("used_view_count", sample.get("num_views", 0))),
    )


def _build_prepared_loader(
    items: Sequence[_PreparedObject],
    settings: _DataLoaderSettings,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
    decode_images: bool = True,
    views_per_sample: int | None = None,
    base_seed: int = 7,
    profile_loading: bool = False,
) -> DataLoader:
    worker_items = tuple(
        _PreparedObject(
            sample=item.sample,
            training_target=item.training_target,
            clipped_target_vertices=item.clipped_target_vertices,
            used_view_count=item.used_view_count,
        )
        for item in items
    )
    dataset = _MaterializedPreparedDataset(
        worker_items,
        decode_images=decode_images,
        views_per_sample=views_per_sample,
        base_seed=base_seed,
        profile_loading=profile_loading,
    )
    sampler = _EpochIndexSampler(
        len(dataset), shuffle=shuffle, generator=generator
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


def _set_loader_epoch(loader: DataLoader | None, epoch: int) -> None:
    if loader is not None and isinstance(loader.sampler, _EpochIndexSampler):
        loader.sampler.set_epoch(epoch)


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
    train_views_per_sample = _optional_positive_int(
        loading.get("train_views_per_sample"), "data_loading.train_views_per_sample"
    )
    validation_views_per_sample = _optional_positive_int(
        loading.get("validation_views_per_sample"),
        "data_loading.validation_views_per_sample",
    )
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
        train_views_per_sample=train_views_per_sample,
        validation_views_per_sample=validation_views_per_sample,
    )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be null or a positive integer.")
    return result


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
        "target_magnitude_weight_lambda": float(
            training.get("target_magnitude_weight_lambda", 0.0)
        ),
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
        raw_target=prepared.raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=prepared.used_view_count,
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


def _mean_nested_metric(
    metrics: Mapping[str, Mapping[str, Any]], group: str, name: str
) -> float | None:
    values = []
    for item in metrics.values():
        nested = item.get(group)
        if isinstance(nested, Mapping) and nested.get(name) is not None:
            values.append(float(nested[name]))
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
    views_per_sample: int | None = None,
) -> tuple[float, dict[str, dict[str, Any]]]:
    model.eval()
    decode_images = model.input_mode != "coarse_only" and not model.zero_images
    target_mode, epsilon = _target_settings(config)
    confidence_settings = _confidence_settings(config)
    losses = []
    metrics: dict[str, dict[str, Any]] = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    items = samples if data_loader is None else data_loader
    source_by_id = {
        str(item.sample["sample_id"]): item
        for item in samples
        if isinstance(item, _PreparedObject)
    }
    non_blocking = _data_loader_settings(config).pin_memory
    for item in items:
        cpu_prepared = _prepared_from_loader_item(item)
        if data_loader is None:
            cpu_prepared = _select_prepared_views(
                cpu_prepared,
                views_per_sample,
                base_seed=query_seed,
                epoch=0,
            )
        source = source_by_id.get(str(cpu_prepared.sample["sample_id"]))
        if cpu_prepared.raw_target is None and source is not None:
            cpu_prepared = _replace_prepared(
                cpu_prepared,
                raw_target=source.raw_target,
                face_count=source.face_count,
            )
        prepared = _prepare_item_for_use(
            cpu_prepared,
            config,
            device,
            cache_on_device,
            non_blocking=non_blocking,
            decode_images=decode_images,
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
            model_output = model(prepared.sample)
        prediction = model_output.predicted_laplacian.float()
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
                prediction, prepared.sample["local_edge_length"], eps=epsilon
            )
        else:
            raw_prediction = prediction
        if prepared.raw_target is None:
            raise ValueError("Evaluation requires an explicit raw Laplacian target.")
        raw_metrics = laplacian_prediction_metrics(
            raw_prediction,
            prepared.raw_target.to(device=raw_prediction.device),
            valid_mask=valid_mask,
        )
        raw_prediction_cpu = raw_prediction.detach().cpu()
        sample_id = str(prepared.sample["sample_id"])
        if sample_id in metrics:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in one dataset split.")
        confidence_metrics = None
        confidence_prediction_cpu = None
        if model_output.confidence_prediction is not None:
            confidence_prediction = model_output.confidence_prediction.float()
            confidence_metrics = confidence_calibration_metrics(
                confidence_prediction,
                prediction,
                prepared.training_target.float(),
                valid_mask=valid_mask,
                quantile_bins=confidence_settings["quantile_bins"],
            )
            confidence_prediction_cpu = confidence_prediction.detach().cpu()
        visible = model_output.valid_views.any(dim=0)
        metrics[sample_id] = {
            "loss": loss_value,
            "exact_query_loss": None if exact_loss is None else float(exact_loss.item()),
            "perturbed_query_loss": (
                None if perturbed_loss is None else float(perturbed_loss.item())
            ),
            "query_perturbation": prepared.sample.get(
                "query_perturbation_diagnostics"
            ),
            "vertex_count": int(prepared.sample["vertices"].shape[0]),
            "face_count": prepared.face_count,
            "view_count": int(prepared.sample["num_views"]),
            "clipped_target_vertices": prepared.clipped_target_vertices,
            "target_space": target_metrics,
            "recovered_raw_space": raw_metrics,
            "confidence": confidence_metrics,
            "visible_query_count": int(visible.sum().item()),
            "invisible_query_count": int((~visible).sum().item()),
        }
        if prediction_dir is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
            np.save(
                prediction_dir / f"{safe_id}_target_space_delta.npy",
                prediction.detach().cpu().numpy(),
            )
            np.save(
                prediction_dir / f"{safe_id}_raw_delta.npy",
                raw_prediction_cpu.numpy(),
            )
            if confidence_prediction_cpu is not None:
                np.save(
                    prediction_dir / f"{safe_id}_confidence.npy",
                    confidence_prediction_cpu.numpy(),
                )
    if not losses:
        raise ValueError("Cannot evaluate an empty dataset split.")
    return float(np.mean(losses)), metrics


def _save_multi_checkpoint(
    path: Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float | None,
    validation_loss: float | None,
    config: Mapping[str, Any],
    train_meshes: int,
    validation_meshes: int,
    *,
    optimizer_steps: int | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "model_config": model.architecture_config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "experiment_config": dict(config),
        "train_meshes": train_meshes,
        "validation_meshes": validation_meshes,
    }
    if optimizer_steps is not None:
        payload.update(
            {
                "optimizer_steps": optimizer_steps,
                "scheduler_state_dict": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "scaler_state_dict": None if scaler is None else scaler.state_dict(),
                "training_state": (
                    None if training_state is None else dict(training_state)
                ),
            }
        )
    torch.save(payload, path)


def _save_optimizer_step_checkpoint(
    output_path: Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    optimizer_steps: int,
    train_loss: float | None,
    validation_loss: float | None,
    config: Mapping[str, Any],
    train_meshes: int,
    validation_meshes: int,
    next_epoch: int,
    groups_completed_in_epoch: int,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_selection_loss: float,
    best_state: Mapping[str, torch.Tensor],
    lr_reduction_count: int,
    early_stopping_best: float,
    early_stopping_bad_validations: int,
) -> Path:
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{optimizer_steps:06d}.pt"
    _save_multi_checkpoint(
        path,
        model,
        optimizer,
        epoch,
        train_loss,
        validation_loss,
        config,
        train_meshes,
        validation_meshes,
        optimizer_steps=optimizer_steps,
        scheduler=scheduler,
        scaler=scaler,
        training_state={
            "next_epoch": int(next_epoch),
            "groups_completed_in_epoch": int(groups_completed_in_epoch),
            "history": [dict(row) for row in history],
            "best_epoch": int(best_epoch),
            "best_selection_loss": float(best_selection_loss),
            "best_model_state_dict": copy.deepcopy(dict(best_state)),
            "lr_reduction_count": int(lr_reduction_count),
            "early_stopping_best": float(early_stopping_best),
            "early_stopping_bad_validations": int(
                early_stopping_bad_validations
            ),
        },
    )
    return path


def _save_resumable_checkpoint(
    path: Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    optimizer_steps: int,
    train_loss: float | None,
    validation_loss: float | None,
    config: Mapping[str, Any],
    train_meshes: int,
    validation_meshes: int,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_selection_loss: float,
    best_state: Mapping[str, torch.Tensor],
    lr_reduction_count: int,
    early_stopping_best: float,
    early_stopping_bad_validations: int,
) -> None:
    _save_multi_checkpoint(
        path,
        model,
        optimizer,
        epoch,
        train_loss,
        validation_loss,
        config,
        train_meshes,
        validation_meshes,
        optimizer_steps=optimizer_steps,
        scheduler=scheduler,
        scaler=scaler,
        training_state={
            "next_epoch": int(epoch + 1),
            "groups_completed_in_epoch": 0,
            "history": [dict(row) for row in history],
            "best_epoch": int(best_epoch),
            "best_selection_loss": float(best_selection_loss),
            "best_model_state_dict": copy.deepcopy(dict(best_state)),
            "lr_reduction_count": int(lr_reduction_count),
            "early_stopping_best": float(early_stopping_best),
            "early_stopping_bad_validations": int(
                early_stopping_bad_validations
            ),
        },
    )

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
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import resolve_lazy_image_paths, validate_sample
from .controlled_displacement import (
    CURRENT_GRAPH_LAPLACIAN,
    DIRECT_VERTEX_DISPLACEMENT,
    displacement_target,
    prediction_semantics,
)
from .distributed import (
    DistributedContext,
    current_distributed_context,
    reduce_scalar,
)
from .cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
    differentiable_cotangent_sparse_recovery,
    differentiable_cotangent_sparse_recovery_with_audit,
)
from .differentiable_sparse_recovery import (
    ConjugateGradientAudit,
    differentiable_regularized_sparse_recovery,
    differentiable_regularized_sparse_recovery_with_audit,
)
from .hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
    differentiable_hard_anchor_sparse_recovery,
    differentiable_hard_anchor_sparse_recovery_with_audit,
)
from .losses import (
    confidence_calibration_metrics,
    confidence_reliability_loss,
    laplacian_prediction_metrics,
    weighted_robust_laplacian_loss,
)
from .local_query_jitter import (
    LocalQueryJitterSettings,
    apply_local_query_jitter,
    local_query_jitter_settings,
    validate_local_query_jitter_contract,
)
from .model import LearnedLaplacianModel
from .evaluation import evaluate_mesh_geometry
from mlr.data import Mesh
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
from .two_branch_hybrid import (
    TwoBranchPretrainedHybridModel,
    load_specialist_checkpoint,
)
from .trainer import _resolve_device, _seed_everything
from .vertex_sampling import sample_training_vertices, vertex_sampling_settings


OUTPUT_REPRESENTATION_LOSS = "output_representation"
RAW_LAPLACIAN_LOSS = "raw_laplacian"
PREDICTION_LOSS_SPACES = {OUTPUT_REPRESENTATION_LOSS, RAW_LAPLACIAN_LOSS}
RECOVERY_PRIMARY_SUPERVISION = {"prediction_space", "oriented_face_normals"}
RECOVERY_ANCHOR_MODES = {"initial_vertices", "cached_frozen_vertices"}
MULTIPROCESSING_SHARING_STRATEGIES = {"file_descriptor", "file_system"}


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
    continuation_optimizer_steps: int
    device: str
    runtime_seconds: float
    peak_gpu_memory_mb: float | None
    target_mode: str
    prediction_semantics: str
    prediction_loss_space: str
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
    distributed_world_size: int = 1
    cuda_transfer_overlap_enabled: bool = False


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
    # Loss-side only. This tensor is deliberately never inserted into ``sample``,
    # which is the only mapping passed to the predictor.
    clean_vertices: torch.Tensor | None = None
    recovery_anchor_vertices: torch.Tensor | None = None


@dataclass(frozen=True)
class _DataLoaderSettings:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    multiprocessing_sharing_strategy: str
    cuda_prefetch: bool
    train_views_per_sample: int | None
    validation_views_per_sample: int | None


@dataclass(frozen=True)
class _EarlyStoppingSettings:
    enabled: bool
    patience_validations: int
    min_delta: float


@dataclass(frozen=True)
class _RecoveryAwareGeometrySettings:
    enabled: bool
    regularization: float
    beta: float
    prediction_loss_weight: float
    maximum_iterations: int
    tolerance: float
    runtime_diagnostics: bool
    compute_dtype: str
    adaptive_lambda: bool
    solver_mode: str
    primary_supervision: str
    normal_epsilon: float
    anchor_mode: str


@dataclass(frozen=True)
class _HybridSingleGeometrySettings:
    enabled: bool
    regularization: float
    maximum_iterations: int
    tolerance: float
    runtime_diagnostics: bool
    compute_dtype: str
    validation_surface_samples: int
    operator: str
    cotangent_relative_area_epsilon: float


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
            "raw_target": prepared.raw_target,
            "clean_vertices": prepared.clean_vertices,
            "recovery_anchor_vertices": prepared.recovery_anchor_vertices,
            "clipped_target_vertices": prepared.clipped_target_vertices,
            "face_count": prepared.face_count,
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
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.size = int(size)
        self.shuffle = shuffle
        self.generator = generator
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.size < 1:
            raise ValueError("Distributed epoch sampler requires a non-empty dataset.")
        if self.world_size < 1:
            raise ValueError("world_size must be positive.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size).")
        self.num_samples = math.ceil(self.size / self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.size, generator=self.generator).tolist()
        else:
            indices = list(range(self.size))
        if len(indices) < self.total_size:
            repeats = math.ceil((self.total_size - len(indices)) / len(indices))
            indices += (indices * repeats)[: self.total_size - len(indices)]
        rank_indices = indices[self.rank : self.total_size : self.world_size]
        return iter((int(index), self.epoch) for index in rank_indices)

    def __len__(self) -> int:
        return self.num_samples


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
    reset_resume_tracking: bool = False,
    initialization_checkpoint: str | Path | None = None,
) -> MultiObjectTrainingResult:
    """Train one shared model over ragged mesh samples, one mesh forward at a time."""

    if len(train_samples) < 1:
        raise ValueError("train_samples must contain at least one mesh.")
    validation_samples = validation_samples or ()
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    device = _resolve_device(device_override or str(config.get("device", "cpu")))
    distributed = current_distributed_context(device)
    if distributed.enabled and distributed.device != device:
        raise ValueError(
            f"Distributed process uses {distributed.device}, but training resolved {device}."
        )
    is_main_process = distributed.is_main
    progress = progress and is_main_process
    target_mode, epsilon = _target_settings(config)
    output_semantics = prediction_semantics(config)
    query_settings = query_augmentation_settings(config)
    local_jitter_settings = local_query_jitter_settings(config)
    vertex_sampling = vertex_sampling_settings(config)
    if query_settings.enabled and local_jitter_settings.enabled:
        raise ValueError(
            "query_training and local_query_jitter cannot be enabled together."
        )
    if query_settings.enabled and str(
        config.get("model", {}).get("geometry_mode", "legacy")
    ) != QUERY_FOURIER_GEOMETRY_MODE:
        raise ValueError(
            "query_training.enabled=true requires model.geometry_mode='query_fourier'."
        )
    base_model = _build_model(config, input_mode_override, zero_images).to(device)
    if resume_checkpoint is not None and initialization_checkpoint is not None:
        raise ValueError(
            "resume_checkpoint and initialization_checkpoint are mutually exclusive."
        )
    if initialization_checkpoint is not None:
        _load_initialization_checkpoint(base_model, initialization_checkpoint, device)
    trainable_scope = str(
        config.get("training", {}).get("trainable_parameter_scope", "all")
    )
    if trainable_scope == "dynamic_residual_expert_only":
        _freeze_except_dynamic_residual_expert(base_model)
    elif trainable_scope != "all":
        raise ValueError(
            "training.trainable_parameter_scope must be 'all' or "
            "'dynamic_residual_expert_only'."
        )
    if base_model.input_mode == "coarse_only" or base_model.zero_images:
        base_model.image_encoder.requires_grad_(False)
    decode_images = base_model.input_mode != "coarse_only" and not base_model.zero_images
    keep_projection = base_model.input_mode != "coarse_only"
    model: nn.Module
    if distributed.enabled:
        ddp_arguments: dict[str, Any] = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_arguments.update(
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
            )
        model = DistributedDataParallel(base_model, **ddp_arguments)
    else:
        model = base_model
    training = config.get("training", {})
    prediction_loss_space = _prediction_loss_space(training)
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
    report_every_optimizer_steps_value = multi.get("report_every_optimizer_steps")
    report_every_optimizer_steps = (
        None
        if report_every_optimizer_steps_value is None
        else int(report_every_optimizer_steps_value)
    )
    cache_on_device = bool(multi.get("cache_prepared_samples_on_device", False))
    profile_training = bool(multi.get("profile_training", False))
    early_stopping = _early_stopping_settings(multi)
    loader_settings = _data_loader_settings(config)
    _configure_multiprocessing_sharing(loader_settings)
    if early_stopping.enabled and not validation_samples:
        raise ValueError("Early stopping requires at least one validation sample.")
    if epochs < 1 or accumulation_meshes < 1 or validation_every < 1:
        raise ValueError(
            "epochs, gradient_accumulation_meshes, and validation_every_epochs must be positive."
        )
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError("max_optimizer_steps must be positive when provided.")
    if report_every_optimizer_steps is not None and report_every_optimizer_steps < 1:
        raise ValueError(
            "report_every_optimizer_steps must be positive when provided."
        )
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
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Training has no trainable parameters.")
    optimizer = torch.optim.Adam(
        trainable_parameters,
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
    recovery_aware = _recovery_aware_geometry_settings(config)
    hybrid_single = _hybrid_single_geometry_settings(config)
    direct_vertex_runtime_diagnostics = bool(
        training.get("direct_vertex_runtime_diagnostics", False)
    )
    if recovery_aware.enabled:
        if confidence_settings["enabled"]:
            raise ValueError(
                "recovery-aware geometry supervision requires confidence.enabled=false."
            )
        if vertex_sampling.mode != "full":
            raise ValueError(
                "recovery-aware geometry supervision requires full vertex sampling."
            )
        if recovery_aware.adaptive_lambda != bool(
            base_model.recovery_lambda_head_enabled
        ):
            raise ValueError(
                "training.recovery_aware_geometry_loss.adaptive_lambda must match "
                "model.recovery_lambda_head.enabled."
            )
    if hybrid_single.enabled:
        if recovery_aware.enabled:
            raise ValueError("Hybrid single-geometry training cannot enable recovery-aware auxiliary loss.")
        if confidence_settings["enabled"]:
            raise ValueError("Hybrid single-geometry training requires confidence.enabled=false.")
        if not base_model.hybrid_direct_head_enabled:
            raise ValueError("Hybrid single-geometry training requires model.hybrid_direct_head.enabled=true.")
        if base_model.recovery_lambda_head_enabled:
            raise ValueError("Hybrid single-geometry training forbids adaptive lambda.")
        if vertex_sampling.mode != "full":
            raise ValueError("Hybrid single-geometry training requires all vertices.")
        if output_semantics != CURRENT_GRAPH_LAPLACIAN:
            raise ValueError("Hybrid Laplacian branch must retain current-graph Laplacian semantics.")
    if output_semantics == DIRECT_VERTEX_DISPLACEMENT:
        if str(loss_kwargs["loss_type"]) != "mse":
            raise ValueError("direct vertex displacement training requires MSE.")
        if prediction_loss_space != OUTPUT_REPRESENTATION_LOSS:
            raise ValueError(
                "direct vertex displacement training requires output-representation loss."
            )
        if confidence_settings["enabled"]:
            raise ValueError(
                "direct vertex displacement training requires confidence.enabled=false."
            )
        if recovery_aware.enabled:
            raise ValueError(
                "direct vertex displacement training cannot use recovery-aware integration."
            )
        if vertex_sampling.mode != "full":
            raise ValueError("direct vertex displacement training requires all vertices.")
        if float(loss_kwargs["target_magnitude_weight_lambda"]) != 0.0:
            raise ValueError("direct vertex displacement training cannot use target weighting.")
    output_path = (
        None if output_dir is None or not is_main_process else Path(output_dir)
    )
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    resume_payload: dict[str, Any] | None = None
    if reset_resume_tracking and resume_checkpoint is None:
        raise ValueError("reset_resume_tracking requires resume_checkpoint.")
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_payload.get("optimizer_steps") is None:
            raise ValueError(
                "resume_checkpoint is not an optimizer-step checkpoint."
            )
        allow_lambda_head_extension = bool(
            training.get("allow_resume_with_new_recovery_lambda_head", False)
        )
        if allow_lambda_head_extension:
            incompatible = base_model.load_state_dict(
                resume_payload["model_state_dict"], strict=False
            )
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith("recovery_lambda_head.")
            ]
            if incompatible.unexpected_keys or disallowed_missing:
                raise ValueError(
                    "Recovery-lambda continuation checkpoint is incompatible: "
                    f"missing={disallowed_missing}, "
                    f"unexpected={list(incompatible.unexpected_keys)}."
                )
            if not incompatible.missing_keys:
                raise ValueError(
                    "allow_resume_with_new_recovery_lambda_head was set but the "
                    "checkpoint is not missing a recovery lambda head."
                )
            old_optimizer = resume_payload["optimizer_state_dict"]
            new_optimizer = optimizer.state_dict()
            if len(old_optimizer["param_groups"]) != 1 or len(new_optimizer["param_groups"]) != 1:
                raise ValueError("Lambda-head extension requires one optimizer parameter group.")
            old_ids = list(old_optimizer["param_groups"][0]["params"])
            new_ids = list(new_optimizer["param_groups"][0]["params"])
            if len(new_ids) <= len(old_ids):
                raise ValueError("Lambda-head extension did not add optimizer parameters.")
            remapped_state = {
                new_id: copy.deepcopy(old_optimizer["state"][old_id])
                for old_id, new_id in zip(old_ids, new_ids)
                if old_id in old_optimizer["state"]
            }
            remapped_group = copy.deepcopy(old_optimizer["param_groups"][0])
            remapped_group["params"] = new_ids
            optimizer.load_state_dict(
                {"state": remapped_state, "param_groups": [remapped_group]}
            )
        else:
            base_model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if scheduler is not None and resume_payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
    initial_learning_rate = float(optimizer.param_groups[0]["lr"])

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
    initial_loading_seconds = reduce_scalar(
        initial_loading_seconds, distributed, reduction="max"
    )
    static_preparation_seconds = reduce_scalar(
        static_preparation_seconds, distributed, reduction="max"
    )
    device_cache_seconds = reduce_scalar(
        device_cache_seconds, distributed, reduction="max"
    )
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
            rank=distributed.rank,
            world_size=distributed.world_size,
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
    cuda_transfer_overlap_enabled = _cuda_transfer_overlap_enabled(
        device,
        cache_on_device=cache_on_device,
        settings=loader_settings,
    )
    cuda_transfer_stream = (
        torch.cuda.Stream(device=device) if cuda_transfer_overlap_enabled else None
    )
    cuda_compute_stream = (
        torch.cuda.current_stream(device) if cuda_transfer_overlap_enabled else None
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
            "multiprocessing_sharing_strategy="
            f"{loader_settings.multiprocessing_sharing_strategy} "
            f"train_views={loader_settings.train_views_per_sample} "
            f"validation_views={loader_settings.validation_views_per_sample} "
            f"decode_images={decode_images} "
            f"cuda_prefetch={cuda_transfer_overlap_enabled}",
            flush=True,
        )
        print(f"AMP: {amp_enabled} ({amp_dtype})", flush=True)

    resume_state = {} if resume_payload is None else dict(
        resume_payload.get("training_state", {})
    )
    if reset_resume_tracking:
        resume_state = {
            "next_epoch": resume_state.get("next_epoch", 1),
            "groups_completed_in_epoch": resume_state.get(
                "groups_completed_in_epoch", 0
            ),
            "distributed_world_size": resume_state.get(
                "distributed_world_size", distributed.world_size
            ),
        }
    history: list[dict[str, float | int | None]] = list(
        resume_state.get("history", [])
    )
    best_epoch = int(resume_state.get("best_epoch", 0))
    best_selection_loss = float(resume_state.get("best_selection_loss", float("inf")))
    stored_best_state = resume_state.get("best_model_state_dict")
    best_state = (
        copy.deepcopy(base_model.state_dict())
        if stored_best_state is None
        else stored_best_state
    )
    optimizer_steps = int(
        0 if resume_payload is None else resume_payload["optimizer_steps"]
    )
    starting_optimizer_steps = optimizer_steps
    step_history: list[dict[str, float | int | None]] = []
    if output_path is not None:
        step_history_path = output_path / "training_step_history.json"
        if step_history_path.is_file():
            loaded_step_history = json.loads(
                step_history_path.read_text(encoding="utf-8")
            )
            if not isinstance(loaded_step_history, list):
                raise ValueError("training_step_history.json must contain a list.")
            step_history = loaded_step_history
        elif report_every_optimizer_steps is not None:
            previous_step = 0
            for epoch_record in history:
                recorded_step = int(epoch_record.get("optimizer_steps", 0))
                if recorded_step <= 0:
                    continue
                step_count = recorded_step - previous_step
                train_seconds = float(epoch_record.get("train_seconds", 0.0))
                step_history.append(
                    {
                        "optimizer_steps": recorded_step,
                        "interval_start_optimizer_steps": previous_step,
                        "optimizer_steps_in_interval": step_count,
                        "progress_percent": (
                            100.0 * recorded_step / max_optimizer_steps
                            if max_optimizer_steps is not None
                            else None
                        ),
                        "train_loss": epoch_record.get("train_loss"),
                        "train_objective": epoch_record.get("train_objective"),
                        "train_confidence_loss": epoch_record.get(
                            "train_confidence_loss"
                        ),
                        "validation_loss": epoch_record.get("validation_loss"),
                        "validation_seconds": epoch_record.get(
                            "validation_seconds"
                        ),
                        "learning_rate": epoch_record.get("learning_rate"),
                        "interval_seconds": train_seconds,
                        "optimizer_steps_per_second": (
                            step_count / train_seconds if train_seconds > 0 else None
                        ),
                    }
                )
                previous_step = recorded_step
            if step_history:
                _write_step_history(output_path, step_history)
    next_report_step = (
        None
        if report_every_optimizer_steps is None
        else (
            optimizer_steps // report_every_optimizer_steps + 1
        )
        * report_every_optimizer_steps
    )
    report_mesh_loss_tensors: list[torch.Tensor] = []
    report_objective_tensors: list[torch.Tensor] = []
    report_confidence_loss_tensors: list[torch.Tensor] = []
    report_refine_loss_tensors: list[torch.Tensor] = []
    report_pcg_iterations: list[float] = []
    report_pcg_relative_residuals: list[float] = []
    report_pcg_failed_solves = 0
    report_prediction_gradient_norms: list[float] = []
    report_direct_gradient_norms: list[float] = []
    report_prediction_head_gradient_norms: list[float] = []
    report_direct_head_gradient_norms: list[float] = []
    report_b_backbone_gradient_norms: list[float] = []
    report_e_backbone_gradient_norms: list[float] = []
    report_image_encoder_gradient_norms: list[float] = []
    report_graph_block_gradient_norms: list[float] = []
    report_prediction_displacement_rms: list[float] = []
    report_prediction_displacement_mean: list[float] = []
    report_laplacian_output_rms: list[float] = []
    report_recovery_lambda_values: list[float] = []
    report_recovery_lambda_gradient_norms: list[float] = []
    report_recovery_lambda_head_gradient_norms: list[float] = []
    report_nonfinite_counts = 0
    report_started_at = time.perf_counter()
    start_epoch = int(resume_state.get("next_epoch", 1))
    resume_groups_completed = int(
        resume_state.get("groups_completed_in_epoch", 0)
    )
    resume_world_size = int(
        resume_state.get("distributed_world_size", distributed.world_size)
    )
    if resume_groups_completed and resume_world_size != distributed.world_size:
        raise ValueError(
            "A mid-epoch distributed checkpoint must resume with the same world size."
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

    local_train_meshes = math.ceil(len(prepared_train) / distributed.world_size)
    groups_per_epoch = math.ceil(local_train_meshes / accumulation_meshes)
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
            epoch_sampler = _EpochIndexSampler(
                len(prepared_train),
                shuffle=shuffle,
                generator=train_generator,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
            epoch_sampler.set_epoch(epoch)
            order = [index for index, _ in epoch_sampler]
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
        refine_loss_tensors: list[torch.Tensor] = []
        exact_query_loss_tensors: list[torch.Tensor] = []
        perturbed_query_loss_tensors: list[torch.Tensor] = []
        local_jitter_mean_ratios: list[float] = []
        local_jitter_max_ratios: list[float] = []
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
            diagnostic_predictions: list[torch.Tensor] = []
            diagnostic_direct_predictions: list[torch.Tensor] = []
            diagnostic_recovery_lambdas: list[torch.Tensor] = []
            gradient_scale = float(scaler.get_scale())
            pending_cuda_transfer = None
            if cuda_transfer_stream is not None:
                pending_cuda_transfer = _enqueue_cuda_transfer(
                    group[0],
                    config,
                    device,
                    cache_on_device=cache_on_device,
                    decode_images=decode_images,
                    transfer_stream=cuda_transfer_stream,
                )
            for group_index, cpu_prepared in enumerate(group):
                if isinstance(model, DistributedDataParallel):
                    # Delay gradient synchronization until the final mesh in the
                    # local accumulation group. This preserves the global batch
                    # definition while avoiding one all-reduce per mesh.
                    model.require_backward_grad_sync = group_index == len(group) - 1
                if pending_cuda_transfer is not None:
                    if cuda_transfer_stream is None or cuda_compute_stream is None:
                        raise RuntimeError("CUDA transfer pipeline is not initialized.")
                    prepared, transfer_event_pair = pending_cuda_transfer
                    _, transfer_end = transfer_event_pair
                    if transfer_end is None:
                        raise RuntimeError("CUDA transfer completion event is missing.")
                    cuda_compute_stream.wait_event(transfer_end)
                    _record_prepared_stream(prepared, cuda_compute_stream)
                    transfer_events.append(transfer_event_pair)
                    pending_cuda_transfer = (
                        _enqueue_cuda_transfer(
                            group[group_index + 1],
                            config,
                            device,
                            cache_on_device=cache_on_device,
                            decode_images=decode_images,
                            transfer_stream=cuda_transfer_stream,
                        )
                        if group_index + 1 < len(group)
                        else None
                    )
                else:
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
                    transfer_events.append(
                        _finish_cuda_timing(device, transfer_event)
                    )
                    if device.type != "cuda":
                        gpu_transfer_seconds += time.perf_counter() - transfer_start
                prepared = _with_query_augmentation(
                    prepared,
                    query_settings,
                    base_seed=seed,
                    epoch=epoch,
                    enabled=query_settings.enabled,
                )
                prepared = _with_local_query_jitter(
                    prepared,
                    local_jitter_settings,
                    base_seed=seed,
                    epoch=epoch,
                )
                local_jitter_diagnostics = prepared.sample.get(
                    "local_query_jitter_diagnostics"
                )
                if isinstance(local_jitter_diagnostics, Mapping):
                    local_jitter_mean_ratios.append(
                        float(local_jitter_diagnostics["mean_offset_norm_over_h"])
                    )
                    local_jitter_max_ratios.append(
                        float(local_jitter_diagnostics["max_offset_norm_over_h"])
                    )
                forward_start = time.perf_counter()
                forward_event = _start_cuda_timing(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    model_output = model(prepared.sample)
                prediction_fp32 = model_output.predicted_laplacian.float()
                direct_prediction_fp32 = model_output.direct_vertex_displacement_prediction
                if hybrid_single.enabled:
                    if direct_prediction_fp32 is None:
                        raise RuntimeError("Hybrid training requires the direct branch output.")
                    direct_prediction_fp32 = direct_prediction_fp32.float()
                elif direct_prediction_fp32 is not None:
                    raise RuntimeError("A hybrid direct output is present while hybrid loss is disabled.")
                recovery_regularization: float | torch.Tensor = (
                    recovery_aware.regularization
                )
                if recovery_aware.adaptive_lambda:
                    if model_output.recovery_lambda is None:
                        raise RuntimeError(
                            "Adaptive recovery requires a predicted mesh-level lambda."
                        )
                    recovery_regularization = model_output.recovery_lambda
                    recovery_regularization.retain_grad()
                    diagnostic_recovery_lambdas.append(recovery_regularization)
                    report_recovery_lambda_values.append(
                        float(recovery_regularization.detach().cpu())
                    )
                    report_nonfinite_counts += int(
                        (~torch.isfinite(recovery_regularization.detach())).sum().item()
                    )
                elif model_output.recovery_lambda is not None:
                    raise RuntimeError(
                        "Fixed-lambda recovery must not instantiate the lambda head."
                    )
                runtime_gradient_diagnostics = (
                    recovery_aware.enabled and recovery_aware.runtime_diagnostics
                ) or direct_vertex_runtime_diagnostics or (
                    hybrid_single.enabled and hybrid_single.runtime_diagnostics
                )
                if runtime_gradient_diagnostics:
                    prediction_fp32.retain_grad()
                    diagnostic_predictions.append(prediction_fp32)
                    report_nonfinite_counts += int(
                        (~torch.isfinite(prediction_fp32.detach())).sum().item()
                    )
                    if hybrid_single.enabled:
                        assert direct_prediction_fp32 is not None
                        direct_prediction_fp32.retain_grad()
                        diagnostic_direct_predictions.append(direct_prediction_fp32)
                        report_nonfinite_counts += int(
                            (~torch.isfinite(direct_prediction_fp32.detach())).sum().item()
                        )
                        report_laplacian_output_rms.append(
                            float(torch.sqrt(prediction_fp32.detach().square().mean()).item())
                        )
                        direct_magnitude = torch.linalg.vector_norm(
                            direct_prediction_fp32.detach(), dim=-1
                        )
                        report_prediction_displacement_rms.append(
                            float(torch.sqrt(direct_magnitude.square().mean()).item())
                        )
                        report_prediction_displacement_mean.append(
                            float(direct_magnitude.mean().item())
                        )
                if direct_vertex_runtime_diagnostics:
                    displacement_magnitude = torch.linalg.vector_norm(
                        prediction_fp32.detach(), dim=-1
                    )
                    report_prediction_displacement_rms.append(
                        float(torch.sqrt(displacement_magnitude.square().mean()).item())
                    )
                    report_prediction_displacement_mean.append(
                        float(displacement_magnitude.mean().item())
                    )
                full_loss_prediction, full_loss_target = _prediction_loss_inputs(
                    prediction_fp32,
                    prepared,
                    output_semantics=output_semantics,
                    target_mode=target_mode,
                    epsilon=epsilon,
                    prediction_loss_space=prediction_loss_space,
                )
                sampled_vertices = sample_training_vertices(
                    full_loss_target,
                    prepared.sample["valid_scale_mask"],
                    vertex_sampling,
                    sample_id=str(prepared.sample["sample_id"]),
                    base_seed=seed,
                    epoch=epoch,
                )
                loss_prediction = full_loss_prediction
                loss_target = full_loss_target
                loss_weight = prepared.sample["target_confidence"].float()
                if sampled_vertices.indices is not None:
                    loss_prediction = loss_prediction.index_select(
                        0, sampled_vertices.indices
                    )
                    loss_target = loss_target.index_select(0, sampled_vertices.indices)
                    loss_weight = loss_weight.index_select(0, sampled_vertices.indices)
                if hybrid_single.enabled:
                    # Both branch targets are deliberately absent.  The scalar
                    # is replaced below by the final recovered-geometry loss.
                    prediction_loss = prediction_fp32.new_zeros(())
                elif output_semantics == DIRECT_VERTEX_DISPLACEMENT:
                    # Exact Arm-E objective: mean_i ||delta_v_pred-delta_v_gt||_2^2.
                    prediction_loss = _direct_vertex_residual_mse(
                        loss_prediction, loss_target
                    )
                else:
                    prediction_loss = weighted_robust_laplacian_loss(
                        loss_prediction,
                        loss_target,
                        loss_weight,
                        **loss_kwargs,
                    )
                confidence_loss = None
                refine_loss = None
                objective = prediction_loss
                if confidence_settings["enabled"]:
                    if model_output.confidence_prediction is None:
                        raise RuntimeError(
                            "confidence.enabled requires a model confidence head."
                        )
                    confidence_prediction = model_output.confidence_prediction.float()
                    loss_confidence_prediction = confidence_prediction
                    if sampled_vertices.indices is not None:
                        loss_confidence_prediction = confidence_prediction.index_select(
                            0, sampled_vertices.indices
                        )
                    confidence_loss = confidence_reliability_loss(
                        loss_confidence_prediction,
                        loss_prediction,
                        loss_target,
                        loss_weight,
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
                if recovery_aware.enabled:
                    if recovery_aware.runtime_diagnostics:
                        try:
                            refine_loss, recovered_vertices, pcg_audit = (
                                _recovery_refine_loss_with_audit(
                                    prediction_fp32,
                                    prepared,
                                    recovery_aware,
                                    regularization=recovery_regularization,
                                )
                            )
                        except RuntimeError as error:
                            report_pcg_failed_solves += 1
                            if output_path is not None:
                                failure_path = output_path / (
                                    "recovery_runtime_failure_"
                                    f"rank{distributed.rank}.json"
                                )
                                failure_path.write_text(
                                    json.dumps(
                                        {
                                            "optimizer_steps": optimizer_steps,
                                            "epoch": epoch,
                                            "rank": distributed.rank,
                                            "sample_id": str(
                                                prepared.sample["sample_id"]
                                            ),
                                            "lambda": float(
                                                torch.as_tensor(
                                                    recovery_regularization
                                                ).detach().cpu()
                                            ),
                                            "solver_mode": recovery_aware.solver_mode,
                                            "hard_anchor_count": int(
                                                prepared.sample.get(
                                                    "hard_anchor_indices",
                                                    torch.empty(0),
                                                ).numel()
                                            ),
                                            "maximum_iterations": recovery_aware.maximum_iterations,
                                            "tolerance": recovery_aware.tolerance,
                                            "error": str(error),
                                        },
                                        indent=2,
                                        sort_keys=True,
                                    )
                                    + "\n",
                                    encoding="utf-8",
                                )
                            raise
                        report_pcg_iterations.append(float(pcg_audit.iterations))
                        report_pcg_relative_residuals.append(
                            float(pcg_audit.relative_residual)
                        )
                        report_nonfinite_counts += int(
                            (~torch.isfinite(recovered_vertices.detach())).sum().item()
                        )
                    else:
                        refine_loss, recovered_vertices = _recovery_refine_loss(
                            prediction_fp32,
                            prepared,
                            recovery_aware,
                            regularization=recovery_regularization,
                        )
                    if recovery_aware.primary_supervision == "oriented_face_normals":
                        clean_vertices = prepared.clean_vertices
                        if clean_vertices is None:
                            raise RuntimeError(
                                "Oriented face-normal supervision requires clean vertices."
                            )
                        prediction_loss = _area_weighted_oriented_face_normal_loss(
                            recovered_vertices,
                            clean_vertices,
                            prepared.sample["faces"],
                            epsilon=recovery_aware.normal_epsilon,
                        )
                    objective = (
                        recovery_aware.prediction_loss_weight * prediction_loss
                        + recovery_aware.beta * refine_loss
                    )
                    refine_loss_tensors.append(refine_loss.detach())
                if hybrid_single.enabled:
                    assert direct_prediction_fp32 is not None
                    try:
                        refine_loss, recovered_vertices, pcg_audit, _ = (
                            _hybrid_single_geometry_loss(
                                prediction_fp32,
                                direct_prediction_fp32,
                                prepared,
                                hybrid_single,
                                with_audit=hybrid_single.runtime_diagnostics,
                            )
                        )
                    except RuntimeError:
                        report_pcg_failed_solves += 1
                        raise
                    objective = refine_loss
                    prediction_loss = refine_loss
                    refine_loss_tensors.append(refine_loss.detach())
                    if pcg_audit is not None:
                        report_pcg_iterations.append(float(pcg_audit.iterations))
                        report_pcg_relative_residuals.append(float(pcg_audit.relative_residual))
                    report_nonfinite_counts += int(
                        (~torch.isfinite(recovered_vertices.detach())).sum().item()
                    )
                if not torch.isfinite(objective):
                    sample_id = prepared.sample["sample_id"]
                    raise FloatingPointError(
                        f"Training produced a non-finite loss for {sample_id!r} at epoch {epoch}."
                    )
                try:
                    scaler.scale(objective / len(group)).backward()
                except RuntimeError as error:
                    raise RuntimeError(
                        "Training backward failed for "
                        f"sample_id={prepared.sample['sample_id']!r}, "
                        f"epoch={epoch}, optimizer_steps={optimizer_steps}, "
                        f"rank={distributed.rank}, group_index={group_index}."
                    ) from error
                forward_events.append(_finish_cuda_timing(device, forward_event))
                if device.type != "cuda":
                    forward_backward_seconds += time.perf_counter() - forward_start
                mesh_loss_tensors.append(prediction_loss.detach())
                objective_tensors.append(objective.detach())
                report_mesh_loss_tensors.append(prediction_loss.detach())
                report_objective_tensors.append(objective.detach())
                if confidence_loss is not None:
                    report_confidence_loss_tensors.append(confidence_loss.detach())
                if refine_loss is not None:
                    report_refine_loss_tensors.append(refine_loss.detach())
                visible = model_output.valid_views.any(dim=0)
                visible_query_count += int(visible.sum().item())
                invisible_query_count += int((~visible).sum().item())
                with torch.no_grad():
                    exact_loss, perturbed_loss = (None, None) if hybrid_single.enabled else _query_subset_losses(
                        full_loss_prediction,
                        full_loss_target,
                        prepared.sample["target_confidence"].float(),
                        prepared.sample.get("query_is_exact"),
                        loss_kwargs,
                    )
                if exact_loss is not None:
                    exact_query_loss_tensors.append(exact_loss.detach())
                if perturbed_loss is not None:
                    perturbed_query_loss_tensors.append(perturbed_loss.detach())
            runtime_gradient_diagnostics = (
                recovery_aware.enabled and recovery_aware.runtime_diagnostics
            ) or direct_vertex_runtime_diagnostics or (
                hybrid_single.enabled and hybrid_single.runtime_diagnostics
            )
            if gradient_clip > 0 or runtime_gradient_diagnostics:
                scaler.unscale_(optimizer)
            if runtime_gradient_diagnostics:
                for diagnostic_prediction in diagnostic_predictions:
                    prediction_gradient = diagnostic_prediction.grad
                    if prediction_gradient is None:
                        raise RuntimeError(
                            "Recovery runtime diagnostic did not retain delta_pred gradient."
                        )
                    unscaled_prediction_gradient = prediction_gradient / gradient_scale
                    report_prediction_gradient_norms.append(
                        float(torch.linalg.vector_norm(unscaled_prediction_gradient).item())
                    )
                    report_nonfinite_counts += int(
                        (~torch.isfinite(unscaled_prediction_gradient)).sum().item()
                    )
                for diagnostic_direct in diagnostic_direct_predictions:
                    direct_gradient = diagnostic_direct.grad
                    if direct_gradient is None:
                        raise RuntimeError("Hybrid runtime diagnostic lost the V_direct gradient.")
                    unscaled_direct_gradient = direct_gradient / gradient_scale
                    report_direct_gradient_norms.append(
                        float(torch.linalg.vector_norm(unscaled_direct_gradient).item())
                    )
                    report_nonfinite_counts += int(
                        (~torch.isfinite(unscaled_direct_gradient)).sum().item()
                    )
                prediction_head_parameters = tuple(
                    _unwrap_model(model).predictor.output_mlp.parameters()
                )
                finite_head_squares = torch.zeros((), device=device)
                for parameter in prediction_head_parameters:
                    if parameter.grad is None:
                        continue
                    report_nonfinite_counts += int(
                        (~torch.isfinite(parameter.grad)).sum().item()
                    )
                    finite_head_squares = finite_head_squares + parameter.grad.float().square().sum()
                report_prediction_head_gradient_norms.append(
                    float(torch.sqrt(finite_head_squares).item())
                )
                unwrapped_model = _unwrap_model(model)
                direct_head = unwrapped_model.hybrid_direct_head
                if (
                    direct_head is None
                    and getattr(unwrapped_model, "direct_predictor", None) is not None
                ):
                    direct_head = unwrapped_model.direct_predictor.output_mlp
                if direct_head is not None:
                    direct_head_norm, direct_head_nonfinite = _parameter_gradient_diagnostics(
                        direct_head.parameters(), device
                    )
                    report_direct_head_gradient_norms.append(direct_head_norm)
                    report_nonfinite_counts += direct_head_nonfinite
                if isinstance(unwrapped_model, TwoBranchPretrainedHybridModel):
                    branch_groups = unwrapped_model.branch_parameter_groups()
                    b_backbone_norm, b_backbone_nonfinite = (
                        _parameter_gradient_diagnostics(
                            branch_groups["b_backbone"], device
                        )
                    )
                    e_backbone_norm, e_backbone_nonfinite = (
                        _parameter_gradient_diagnostics(
                            branch_groups["e_backbone"], device
                        )
                    )
                    report_b_backbone_gradient_norms.append(b_backbone_norm)
                    report_e_backbone_gradient_norms.append(e_backbone_norm)
                    report_nonfinite_counts += (
                        b_backbone_nonfinite + e_backbone_nonfinite
                    )
                elif getattr(
                    unwrapped_model, "split_geometry_towers_enabled", False
                ):
                    direct_predictor = unwrapped_model.direct_predictor
                    if direct_predictor is None:
                        raise RuntimeError("Split geometry Direct tower is missing.")
                    lap_backbone = tuple(
                        unwrapped_model.predictor.input_mlp.parameters()
                    ) + tuple(unwrapped_model.predictor.blocks.parameters())
                    direct_backbone = tuple(
                        direct_predictor.input_mlp.parameters()
                    ) + tuple(direct_predictor.blocks.parameters())
                    lap_backbone_norm, lap_backbone_nonfinite = (
                        _parameter_gradient_diagnostics(lap_backbone, device)
                    )
                    direct_backbone_norm, direct_backbone_nonfinite = (
                        _parameter_gradient_diagnostics(direct_backbone, device)
                    )
                    report_b_backbone_gradient_norms.append(lap_backbone_norm)
                    report_e_backbone_gradient_norms.append(
                        direct_backbone_norm
                    )
                    report_nonfinite_counts += (
                        lap_backbone_nonfinite + direct_backbone_nonfinite
                    )
                for recovery_lambda in diagnostic_recovery_lambdas:
                    if recovery_lambda.grad is None:
                        raise RuntimeError(
                            "Adaptive lambda diagnostic did not retain a gradient."
                        )
                    unscaled_lambda_gradient = recovery_lambda.grad / gradient_scale
                    report_recovery_lambda_gradient_norms.append(
                        abs(float(unscaled_lambda_gradient.detach().cpu()))
                    )
                    report_nonfinite_counts += int(
                        (~torch.isfinite(unscaled_lambda_gradient)).sum().item()
                    )
                lambda_head = _unwrap_model(model).recovery_lambda_head
                if lambda_head is not None:
                    lambda_head_norm, lambda_head_nonfinite = (
                        _parameter_gradient_diagnostics(lambda_head.parameters(), device)
                    )
                    report_recovery_lambda_head_gradient_norms.append(
                        lambda_head_norm
                    )
                    report_nonfinite_counts += lambda_head_nonfinite
                if direct_vertex_runtime_diagnostics:
                    unwrapped = _unwrap_model(model)
                    image_norm, image_nonfinite = _parameter_gradient_diagnostics(
                        unwrapped.image_encoder.parameters(), device
                    )
                    graph_norm, graph_nonfinite = _parameter_gradient_diagnostics(
                        unwrapped.predictor.blocks.parameters(), device
                    )
                    report_image_encoder_gradient_norms.append(image_norm)
                    report_graph_block_gradient_norms.append(graph_norm)
                    report_nonfinite_counts += image_nonfinite + graph_nonfinite
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            groups_completed_in_epoch += 1
            should_report_step = (
                next_report_step is not None
                and (
                    optimizer_steps >= next_report_step
                    or (
                        max_optimizer_steps is not None
                        and optimizer_steps >= max_optimizer_steps
                    )
                )
            )
            if should_report_step:
                rolling_train_loss = reduce_scalar(
                    float(
                        torch.stack(report_mesh_loss_tensors)
                        .mean()
                        .item()
                    ),
                    distributed,
                    reduction="mean",
                )
                rolling_objective = reduce_scalar(
                    float(
                        torch.stack(
                            report_objective_tensors
                        )
                        .mean()
                        .item()
                    ),
                    distributed,
                    reduction="mean",
                )
                rolling_confidence_loss = _distributed_optional_mean(
                    _mean_optional_tensors(
                        report_confidence_loss_tensors
                    ),
                    distributed,
                )
                rolling_refine_loss = _distributed_optional_mean(
                    _mean_optional_tensors(report_refine_loss_tensors),
                    distributed,
                )
                pcg_iterations_mean = _distributed_optional_mean(
                    float(np.mean(report_pcg_iterations))
                    if report_pcg_iterations
                    else None,
                    distributed,
                )
                pcg_iterations_max = (
                    reduce_scalar(
                        max(report_pcg_iterations), distributed, reduction="max"
                    )
                    if report_pcg_iterations
                    else None
                )
                pcg_relative_residual_mean = _distributed_optional_mean(
                    float(np.mean(report_pcg_relative_residuals))
                    if report_pcg_relative_residuals
                    else None,
                    distributed,
                )
                pcg_relative_residual_max = (
                    reduce_scalar(
                        max(report_pcg_relative_residuals),
                        distributed,
                        reduction="max",
                    )
                    if report_pcg_relative_residuals
                    else None
                )
                pcg_failed_solves = int(
                    reduce_scalar(
                        report_pcg_failed_solves, distributed, reduction="sum"
                    )
                )
                prediction_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_prediction_gradient_norms))
                    if report_prediction_gradient_norms
                    else None,
                    distributed,
                )
                direct_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_direct_gradient_norms))
                    if report_direct_gradient_norms
                    else None,
                    distributed,
                )
                prediction_head_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_prediction_head_gradient_norms))
                    if report_prediction_head_gradient_norms
                    else None,
                    distributed,
                )
                direct_head_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_direct_head_gradient_norms))
                    if report_direct_head_gradient_norms
                    else None,
                    distributed,
                )
                b_backbone_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_b_backbone_gradient_norms))
                    if report_b_backbone_gradient_norms
                    else None,
                    distributed,
                )
                e_backbone_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_e_backbone_gradient_norms))
                    if report_e_backbone_gradient_norms
                    else None,
                    distributed,
                )
                image_encoder_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_image_encoder_gradient_norms))
                    if report_image_encoder_gradient_norms
                    else None,
                    distributed,
                )
                graph_block_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_graph_block_gradient_norms))
                    if report_graph_block_gradient_norms
                    else None,
                    distributed,
                )
                prediction_displacement_rms = _distributed_optional_mean(
                    float(np.mean(report_prediction_displacement_rms))
                    if report_prediction_displacement_rms
                    else None,
                    distributed,
                )
                prediction_displacement_mean = _distributed_optional_mean(
                    float(np.mean(report_prediction_displacement_mean))
                    if report_prediction_displacement_mean
                    else None,
                    distributed,
                )
                laplacian_output_rms = _distributed_optional_mean(
                    float(np.mean(report_laplacian_output_rms))
                    if report_laplacian_output_rms
                    else None,
                    distributed,
                )
                recovery_lambda_mean = _distributed_optional_mean(
                    float(np.mean(report_recovery_lambda_values))
                    if report_recovery_lambda_values
                    else None,
                    distributed,
                )
                recovery_lambda_min = (
                    reduce_scalar(
                        min(report_recovery_lambda_values),
                        distributed,
                        reduction="min",
                    )
                    if report_recovery_lambda_values
                    else None
                )
                recovery_lambda_max = (
                    reduce_scalar(
                        max(report_recovery_lambda_values),
                        distributed,
                        reduction="max",
                    )
                    if report_recovery_lambda_values
                    else None
                )
                recovery_lambda_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_recovery_lambda_gradient_norms))
                    if report_recovery_lambda_gradient_norms
                    else None,
                    distributed,
                )
                recovery_lambda_head_gradient_norm = _distributed_optional_mean(
                    float(np.mean(report_recovery_lambda_head_gradient_norms))
                    if report_recovery_lambda_head_gradient_norms
                    else None,
                    distributed,
                )
                nonfinite_count = int(
                    reduce_scalar(
                        report_nonfinite_counts, distributed, reduction="sum"
                    )
                )
                peak_gpu_memory_mb = (
                    reduce_scalar(
                        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                        distributed,
                        reduction="max",
                    )
                    if device.type == "cuda"
                    else None
                )
                report_seconds = reduce_scalar(
                    time.perf_counter() - report_started_at,
                    distributed,
                    reduction="max",
                )
                report_start_step = max(
                    starting_optimizer_steps,
                    optimizer_steps - report_every_optimizer_steps,
                )
                report_step_count = optimizer_steps - report_start_step
                progress_percent = (
                    100.0 * optimizer_steps / max_optimizer_steps
                    if max_optimizer_steps is not None
                    else None
                )
                report_record: dict[str, float | int | None] = {
                    "optimizer_steps": optimizer_steps,
                    "interval_start_optimizer_steps": report_start_step,
                    "optimizer_steps_in_interval": report_step_count,
                    "progress_percent": progress_percent,
                    "train_loss": rolling_train_loss,
                    "train_operator_normal_loss": (
                        rolling_train_loss
                        if recovery_aware.primary_supervision
                        == "oriented_face_normals"
                        else None
                    ),
                    "train_objective": rolling_objective,
                    "train_confidence_loss": rolling_confidence_loss,
                    "train_recovery_refine_loss": rolling_refine_loss,
                    "pcg_iterations_mean": pcg_iterations_mean,
                    "pcg_iterations_max": pcg_iterations_max,
                    "pcg_relative_residual_mean": pcg_relative_residual_mean,
                    "pcg_relative_residual_max": pcg_relative_residual_max,
                    "pcg_failed_solves": pcg_failed_solves,
                    "delta_pred_gradient_norm": prediction_gradient_norm,
                    "v_direct_gradient_norm": direct_gradient_norm,
                    "delta_v_gradient_norm": (
                        prediction_gradient_norm
                        if output_semantics == DIRECT_VERTEX_DISPLACEMENT
                        else None
                    ),
                    "image_encoder_gradient_norm": image_encoder_gradient_norm,
                    "graph_block_gradient_norm": graph_block_gradient_norm,
                    "prediction_head_gradient_norm": prediction_head_gradient_norm,
                    "direct_head_gradient_norm": direct_head_gradient_norm,
                    "b_laplacian_head_gradient_norm": prediction_head_gradient_norm,
                    "b_backbone_gradient_norm": b_backbone_gradient_norm,
                    "e_direct_head_gradient_norm": direct_head_gradient_norm,
                    "e_backbone_gradient_norm": e_backbone_gradient_norm,
                    "laplacian_output_rms": laplacian_output_rms,
                    "prediction_displacement_rms": prediction_displacement_rms,
                    "prediction_displacement_mean": prediction_displacement_mean,
                    "recovery_lambda_mean": recovery_lambda_mean,
                    "recovery_lambda_min": recovery_lambda_min,
                    "recovery_lambda_max": recovery_lambda_max,
                    "recovery_lambda_gradient_norm": recovery_lambda_gradient_norm,
                    "recovery_lambda_head_gradient_norm": recovery_lambda_head_gradient_norm,
                    "nan_inf_count": nonfinite_count,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                    "validation_loss": None,
                    "validation_hybrid_chamfer": None,
                    "validation_seconds": None,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "interval_seconds": report_seconds,
                    "optimizer_steps_per_second": (
                        report_step_count / report_seconds
                        if report_seconds > 0
                        else None
                    ),
                }
                if output_path is not None:
                    step_history.append(report_record)
                    _write_step_history(output_path, step_history)
                if progress:
                    progress_text = (
                        "n/a"
                        if progress_percent is None
                        else f"{progress_percent:.2f}%"
                    )
                    confidence_text = (
                        "n/a"
                        if rolling_confidence_loss is None
                        else f"{rolling_confidence_loss:.8f}"
                    )
                    refine_text = (
                        "n/a"
                        if rolling_refine_loss is None
                        else f"{rolling_refine_loss:.8f}"
                    )
                    pcg_text = (
                        "n/a"
                        if pcg_iterations_mean is None
                        else (
                            f"{pcg_iterations_mean:.2f}/"
                            f"{float(pcg_iterations_max):.0f}"
                        )
                    )
                    residual_text = (
                        "n/a"
                        if pcg_relative_residual_max is None
                        else f"{float(pcg_relative_residual_max):.3e}"
                    )
                    print(
                        "step progress "
                        f"progress={progress_text} "
                        f"step={optimizer_steps} "
                        f"train={rolling_train_loss:.8f} "
                        f"objective={rolling_objective:.8f} "
                        f"confidence={confidence_text} "
                        f"refine={refine_text} "
                        f"pcg_iter_mean_max={pcg_text} "
                        f"pcg_residual_max={residual_text} "
                        f"pcg_failed={pcg_failed_solves} "
                        f"delta_grad={prediction_gradient_norm} "
                        f"v_direct_grad={direct_gradient_norm} "
                        f"image_grad={image_encoder_gradient_norm} "
                        f"graph_grad={graph_block_gradient_norm} "
                        f"head_grad={prediction_head_gradient_norm} "
                        f"direct_head_grad={direct_head_gradient_norm} "
                        f"b_backbone_grad={b_backbone_gradient_norm} "
                        f"e_backbone_grad={e_backbone_gradient_norm} "
                        f"laplacian_rms={laplacian_output_rms} "
                        f"recovery_lambda={recovery_lambda_mean} "
                        f"lambda_grad={recovery_lambda_gradient_norm} "
                        f"lambda_head_grad={recovery_lambda_head_gradient_norm} "
                        f"displacement_rms={prediction_displacement_rms} "
                        f"nan_inf={nonfinite_count} "
                        f"peak_gpu_mb={peak_gpu_memory_mb} "
                        f"lr={optimizer.param_groups[0]['lr']:.8e} "
                        f"seconds={report_seconds:.2f}",
                        flush=True,
                    )
                report_mesh_loss_tensors.clear()
                report_objective_tensors.clear()
                report_confidence_loss_tensors.clear()
                report_refine_loss_tensors.clear()
                report_pcg_iterations.clear()
                report_pcg_relative_residuals.clear()
                report_pcg_failed_solves = 0
                report_prediction_gradient_norms.clear()
                report_direct_gradient_norms.clear()
                report_prediction_head_gradient_norms.clear()
                report_direct_head_gradient_norms.clear()
                report_b_backbone_gradient_norms.clear()
                report_e_backbone_gradient_norms.clear()
                report_image_encoder_gradient_norms.clear()
                report_graph_block_gradient_norms.clear()
                report_prediction_displacement_rms.clear()
                report_prediction_displacement_mean.clear()
                report_laplacian_output_rms.clear()
                report_recovery_lambda_values.clear()
                report_recovery_lambda_gradient_norms.clear()
                report_recovery_lambda_head_gradient_norms.clear()
                report_nonfinite_counts = 0
                report_started_at = time.perf_counter()
                while (
                    next_report_step is not None
                    and next_report_step <= optimizer_steps
                ):
                    next_report_step += report_every_optimizer_steps
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

        train_loss = reduce_scalar(
            float(torch.stack(mesh_loss_tensors).mean().item()),
            distributed,
            reduction="mean",
        )
        train_objective = reduce_scalar(
            float(torch.stack(objective_tensors).mean().item()),
            distributed,
            reduction="mean",
        )
        train_confidence_loss = _distributed_optional_mean(
            _mean_optional_tensors(confidence_loss_tensors), distributed
        )
        train_refine_loss = _distributed_optional_mean(
            _mean_optional_tensors(refine_loss_tensors), distributed
        )
        train_mean_confidence = _distributed_optional_mean(
            (
                float(torch.cat(confidence_value_tensors).mean().item())
                if confidence_value_tensors
                else None
            ),
            distributed,
        )
        train_exact_query_loss = _distributed_optional_mean(
            _mean_optional_tensors(exact_query_loss_tensors), distributed
        )
        train_perturbed_query_loss = _distributed_optional_mean(
            _mean_optional_tensors(perturbed_query_loss_tensors), distributed
        )
        train_local_jitter_mean_ratio = _distributed_optional_mean(
            (
                float(np.mean(local_jitter_mean_ratios))
                if local_jitter_mean_ratios
                else None
            ),
            distributed,
        )
        train_local_jitter_max_ratio = _distributed_optional_mean(
            (
                float(max(local_jitter_max_ratios))
                if local_jitter_max_ratios
                else None
            ),
            distributed,
        )
        global_loaded_mesh_count = int(
            reduce_scalar(loaded_mesh_count, distributed, reduction="sum")
        )
        global_used_view_count = reduce_scalar(
            used_view_count, distributed, reduction="sum"
        )
        decoded_image_bytes = int(
            reduce_scalar(decoded_image_bytes, distributed, reduction="sum")
        )
        image_decode_resize_seconds = reduce_scalar(
            image_decode_resize_seconds, distributed, reduction="max"
        )
        visible_query_count = int(
            reduce_scalar(visible_query_count, distributed, reduction="sum")
        )
        invisible_query_count = int(
            reduce_scalar(invisible_query_count, distributed, reduction="sum")
        )
        mean_image_decode_resize_seconds = (
            image_decode_resize_seconds / loaded_mesh_count
            if loaded_mesh_count
            else 0.0
        )
        mean_used_view_count = (
            global_used_view_count / global_loaded_mesh_count
            if global_loaded_mesh_count
            else 0.0
        )
        _synchronize_device(device)
        if device.type == "cuda":
            gpu_transfer_seconds = _elapsed_cuda_seconds(transfer_events)
            forward_backward_seconds = _elapsed_cuda_seconds(forward_events)
        data_loading_seconds = reduce_scalar(
            data_loading_seconds, distributed, reduction="max"
        )
        gpu_transfer_seconds = reduce_scalar(
            gpu_transfer_seconds, distributed, reduction="max"
        )
        forward_backward_seconds = reduce_scalar(
            forward_backward_seconds, distributed, reduction="max"
        )
        train_seconds = reduce_scalar(
            time.perf_counter() - train_start, distributed, reduction="max"
        )
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
        validation_laplacian_loss = None
        validation_operator_normal_loss = None
        validation_refine_loss = None
        validation_recovered_vertex_rms = None
        validation_hybrid_chamfer = None
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
            validation_laplacian_loss = _mean_metric(
                validation_epoch_metrics, "laplacian_loss"
            )
            validation_operator_normal_loss = _mean_metric(
                validation_epoch_metrics, "operator_normal_loss"
            )
            validation_refine_loss = _mean_metric(
                validation_epoch_metrics, "recovery_refine_loss"
            )
            validation_recovered_vertex_rms = _mean_metric(
                validation_epoch_metrics, "recovered_vertex_rms"
            )
            validation_hybrid_chamfer = _mean_metric(
                validation_epoch_metrics, "hybrid_chamfer"
            )
            validation_hybrid_chamfer = _distributed_optional_mean(
                validation_hybrid_chamfer, distributed
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
            validation_loss = reduce_scalar(
                validation_loss, distributed, reduction="mean"
            )
            validation_seconds = reduce_scalar(
                time.perf_counter() - validation_start,
                distributed,
                reduction="max",
            )
            epoch_validation_seconds.append(validation_seconds)
            if (
                output_path is not None
                and step_history
                and step_history[-1]["optimizer_steps"] == optimizer_steps
            ):
                step_history[-1]["validation_loss"] = validation_loss
                step_history[-1]["validation_hybrid_chamfer"] = (
                    validation_hybrid_chamfer
                )
                step_history[-1]["validation_seconds"] = validation_seconds
                _write_step_history(output_path, step_history)
                if progress:
                    progress_percent = step_history[-1]["progress_percent"]
                    progress_text = (
                        "n/a"
                        if progress_percent is None
                        else f"{float(progress_percent):.2f}%"
                    )
                    print(
                        "step validation "
                        f"progress={progress_text} "
                        f"step={optimizer_steps} "
                        f"validation={validation_loss:.8f} "
                        f"hybrid_chamfer={validation_hybrid_chamfer} "
                        f"seconds={validation_seconds:.2f}",
                        flush=True,
                    )
        selection_loss = (
            validation_hybrid_chamfer
            if hybrid_single.enabled and prepared_validation
            else (validation_loss if prepared_validation else train_loss)
        )
        if selection_loss is not None and selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_epoch = epoch
            best_state = copy.deepcopy(base_model.state_dict())
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
            "train_operator_normal_loss": (
                train_loss
                if recovery_aware.primary_supervision == "oriented_face_normals"
                else None
            ),
            "train_objective": train_objective,
            "train_confidence_loss": train_confidence_loss,
            "train_recovery_refine_loss": train_refine_loss,
            "train_mean_confidence": train_mean_confidence,
            "validation_loss": validation_loss,
            "validation_normalized_laplacian_loss": validation_laplacian_loss,
            "validation_operator_normal_loss": validation_operator_normal_loss,
            "validation_recovery_refine_loss": validation_refine_loss,
            "validation_recovered_vertex_rms": validation_recovered_vertex_rms,
            "validation_hybrid_chamfer": validation_hybrid_chamfer,
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
            "train_local_jitter_mean_offset_norm_over_h": train_local_jitter_mean_ratio,
            "train_local_jitter_max_offset_norm_over_h": train_local_jitter_max_ratio,
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
        if progress and report_every_optimizer_steps is None:
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

    base_model.load_state_dict(best_state)
    base_model.eval()
    predictions_path = None if output_path is None else output_path / "predictions"
    # A distributed training loader contains only the current rank's shard.
    # Evaluate the complete split on every rank so rank 0 writes complete
    # per-object metrics and prediction files.
    final_train_loader = None if distributed.enabled else train_loader
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
    final_train_loss = reduce_scalar(
        final_train_loss, distributed, reduction="mean"
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
        final_validation_loss = reduce_scalar(
            final_validation_loss, distributed, reduction="mean"
        )
    _synchronize_device(device)
    runtime_seconds = reduce_scalar(
        initial_loading_seconds + time.perf_counter() - start_time,
        distributed,
        reduction="max",
    )
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
    continuation_optimizer_steps = optimizer_steps - starting_optimizer_steps
    mean_optimizer_step_seconds = (
        float(sum(epoch_train_seconds) / continuation_optimizer_steps)
        if continuation_optimizer_steps
        else 0.0
    )
    final_learning_rate = float(optimizer.param_groups[0]["lr"])
    peak_cpu_memory_mb = _peak_cpu_memory_mb()
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    peak_cpu_memory_mb = reduce_scalar(
        peak_cpu_memory_mb, distributed, reduction="max"
    )
    if peak_gpu_memory_mb is not None:
        peak_gpu_memory_mb = reduce_scalar(
            peak_gpu_memory_mb, distributed, reduction="max"
        )
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
            "starting_optimizer_steps": starting_optimizer_steps,
            "continuation_optimizer_steps": continuation_optimizer_steps,
            "train_meshes": len(prepared_train),
            "validation_meshes": len(prepared_validation),
            "target_mode": target_mode,
            "prediction_semantics": output_semantics,
            "prediction_loss_space": prediction_loss_space,
            "target_scaling_epsilon": epsilon,
            "device": str(device),
            "distributed_world_size": distributed.world_size,
            "global_batch_meshes": distributed.world_size * accumulation_meshes,
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
            "cuda_transfer_overlap_enabled": cuda_transfer_overlap_enabled,
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
        model=base_model,
        history=history,
        best_epoch=best_epoch,
        best_selection_loss=best_selection_loss,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        per_object_metrics=per_object_metrics,
        optimizer_steps=optimizer_steps,
        continuation_optimizer_steps=continuation_optimizer_steps,
        device=str(device),
        runtime_seconds=runtime_seconds,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        target_mode=target_mode,
        prediction_semantics=output_semantics,
        prediction_loss_space=prediction_loss_space,
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
        distributed_world_size=distributed.world_size,
        cuda_transfer_overlap_enabled=cuda_transfer_overlap_enabled,
    )


def _write_step_history(
    output_path: Path, history: list[dict[str, float | int | None]]
) -> None:
    (output_path / "training_step_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


def _build_model(
    config: Mapping[str, Any], input_mode_override: str | None, zero_images: bool
) -> LearnedLaplacianModel:
    model_config = config.get("model", {})
    two_branch = model_config.get("two_branch_pretrained_hybrid", {})
    if not isinstance(two_branch, Mapping):
        raise ValueError("model.two_branch_pretrained_hybrid must be an object.")
    if bool(two_branch.get("enabled", False)):
        b_checkpoint = two_branch.get("arm_b_checkpoint")
        e_checkpoint = two_branch.get("arm_e_checkpoint")
        if not b_checkpoint or not e_checkpoint:
            raise ValueError(
                "Two-branch continuous hybrid requires arm_b_checkpoint and "
                "arm_e_checkpoint."
            )
        arm_b = _build_single_model(config, input_mode_override, zero_images)
        arm_e = _build_single_model(config, input_mode_override, zero_images)
        load_specialist_checkpoint(arm_b, b_checkpoint)
        load_specialist_checkpoint(arm_e, e_checkpoint)
        return TwoBranchPretrainedHybridModel(
            arm_b,
            arm_e,
            arm_b_checkpoint=b_checkpoint,
            arm_e_checkpoint=e_checkpoint,
        )
    return _build_single_model(config, input_mode_override, zero_images)


def _build_single_model(
    config: Mapping[str, Any], input_mode_override: str | None, zero_images: bool
) -> LearnedLaplacianModel:
    image_config = config.get("image_encoder", {})
    feature_construction = image_config.get("feature_construction", {})
    if not isinstance(feature_construction, Mapping):
        raise ValueError("image_encoder.feature_construction must be an object.")
    model_config = config.get("model", {})
    position_config = model_config.get("position_encoding", {})
    if not isinstance(position_config, Mapping):
        raise ValueError("model.position_encoding must be an object.")
    expert_config = model_config.get("oracle_residual_expert", {})
    if not isinstance(expert_config, Mapping):
        raise ValueError("model.oracle_residual_expert must be an object.")
    dynamic_expert_config = model_config.get("dynamic_residual_expert", {})
    if not isinstance(dynamic_expert_config, Mapping):
        raise ValueError("model.dynamic_residual_expert must be an object.")
    recovery_lambda_config = model_config.get("recovery_lambda_head", {})
    if not isinstance(recovery_lambda_config, Mapping):
        raise ValueError("model.recovery_lambda_head must be an object.")
    hybrid_direct_config = model_config.get("hybrid_direct_head", {})
    if not isinstance(hybrid_direct_config, Mapping):
        raise ValueError("model.hybrid_direct_head must be an object.")
    split_geometry_config = model_config.get("split_geometry_towers", {})
    if not isinstance(split_geometry_config, Mapping):
        raise ValueError("model.split_geometry_towers must be an object.")
    return LearnedLaplacianModel(
        image_feature_dim=int(image_config.get("feature_dim", 32)),
        image_first_stride=int(image_config.get("first_stride", 2)),
        image_second_stride=int(image_config.get("second_stride", 2)),
        image_feature_construction_mode=str(
            feature_construction.get("mode", "original")
        ),
        image_gaussian_kernel_size=int(
            feature_construction.get("kernel_size", 5)
        ),
        image_gaussian_sigma=float(feature_construction.get("sigma", 1.0)),
        image_view_chunk_size=(
            None
            if image_config.get("view_chunk_size") is None
            else int(image_config["view_chunk_size"])
        ),
        image_gradient_checkpointing=bool(
            image_config.get("gradient_checkpointing", False)
        ),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_graph_layers=int(model_config.get("num_graph_layers", 3)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_mode=input_mode_override or str(config.get("input_mode", "coarse_plus_multiview")),
        zero_images=zero_images,
        geometry_mode=str(model_config.get("geometry_mode", "legacy")),
        position_num_frequencies=int(position_config.get("num_frequencies", 6)),
        position_include_input=bool(position_config.get("include_input", True)),
        predict_confidence=bool(config.get("confidence", {}).get("enabled", False)),
        oracle_residual_expert_enabled=bool(expert_config.get("enabled", False)),
        oracle_residual_expert_hidden_dim=int(expert_config.get("hidden_dim", 32)),
        dynamic_residual_expert_enabled=bool(
            dynamic_expert_config.get("enabled", False)
        ),
        dynamic_residual_expert_hidden_dim=int(
            dynamic_expert_config.get("residual_hidden_dim", 32)
        ),
        dynamic_gate_hidden_dim=int(
            dynamic_expert_config.get("gate_hidden_dim", 32)
        ),
        dynamic_gate_initial_bias=float(
            dynamic_expert_config.get("gate_initial_bias", 0.1)
        ),
        recovery_lambda_head_enabled=bool(
            recovery_lambda_config.get("enabled", False)
        ),
        recovery_lambda_head_hidden_dim=int(
            recovery_lambda_config.get("hidden_dim", 16)
        ),
        recovery_lambda_minimum=float(
            recovery_lambda_config.get("lambda_min", 1e-3)
        ),
        recovery_lambda_maximum=float(
            recovery_lambda_config.get("lambda_max", 1e-1)
        ),
        recovery_lambda_initial=float(
            recovery_lambda_config.get("lambda_initial", 1e-2)
        ),
        hybrid_direct_head_enabled=bool(hybrid_direct_config.get("enabled", False)),
        split_geometry_towers_enabled=bool(
            split_geometry_config.get("enabled", False)
        ),
    )


def _load_initialization_checkpoint(
    model: LearnedLaplacianModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("initialization_checkpoint has no model_state_dict.")
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "dynamic_residual_expert.",
        "dynamic_gate_head.",
        "recovery_lambda_head.",
    )
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if unexpected or disallowed_missing:
        raise ValueError(
            "Initialization checkpoint is not base-model compatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}."
        )


def _freeze_except_dynamic_residual_expert(model: LearnedLaplacianModel) -> None:
    if model.dynamic_residual_expert is None or model.dynamic_gate_head is None:
        raise ValueError(
            "dynamic_residual_expert_only requires model.dynamic_residual_expert.enabled=true."
        )
    model.requires_grad_(False)
    model.dynamic_residual_expert.requires_grad_(True)
    model.dynamic_gate_head.requires_grad_(True)


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


def _prediction_loss_space(training_config: Mapping[str, Any]) -> str:
    value = str(
        training_config.get("prediction_loss_space", OUTPUT_REPRESENTATION_LOSS)
    )
    if value not in PREDICTION_LOSS_SPACES:
        raise ValueError(
            "training.prediction_loss_space must be one of "
            f"{sorted(PREDICTION_LOSS_SPACES)}."
        )
    return value


def _prediction_loss_inputs(
    prediction: torch.Tensor,
    prepared: _PreparedObject,
    *,
    output_semantics: str = CURRENT_GRAPH_LAPLACIAN,
    target_mode: str,
    epsilon: float,
    prediction_loss_space: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prediction and target in the configured loss space.

    The model output representation remains controlled by ``target_mode``.
    Converting a normalized output to raw space here changes only the tensors
    entering the prediction/confidence losses; it does not alter the model or
    the stored current-graph target.
    """

    if output_semantics == DIRECT_VERTEX_DISPLACEMENT:
        if prediction_loss_space != OUTPUT_REPRESENTATION_LOSS:
            raise ValueError(
                "direct_vertex_displacement requires "
                "training.prediction_loss_space='output_representation'."
            )
        return prediction, prepared.training_target.float()
    if output_semantics != CURRENT_GRAPH_LAPLACIAN:
        raise ValueError(f"Unsupported prediction semantics: {output_semantics!r}.")
    if prediction_loss_space == OUTPUT_REPRESENTATION_LOSS:
        return prediction, prepared.training_target.float()
    if prediction_loss_space != RAW_LAPLACIAN_LOSS:
        raise ValueError(f"Unsupported prediction loss space: {prediction_loss_space!r}.")
    if prepared.raw_target is None:
        raise ValueError("Raw-space prediction loss requires prepared.raw_target.")
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        raw_prediction = denormalize_laplacian_by_edge_scale(
            prediction,
            prepared.sample["local_edge_length"],
            eps=epsilon,
        )
    elif target_mode == RAW_LAPLACIAN:
        raw_prediction = prediction
    else:
        raise ValueError(f"Unsupported target mode: {target_mode!r}.")
    raw_target = prepared.raw_target.to(
        device=raw_prediction.device, dtype=raw_prediction.dtype
    )
    return raw_prediction, raw_target


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
    recovery_aware = _recovery_aware_geometry_settings(config)
    hybrid_single = _hybrid_single_geometry_settings(config)
    if query_augmentation_settings(config).enabled:
        validate_gt_query_contract(static_sample)
    if local_query_jitter_settings(config).enabled:
        validate_local_query_jitter_contract(static_sample)
    target_mode, epsilon = _target_settings(config)
    output_semantics = prediction_semantics(config)
    target = (
        displacement_target(static_sample)
        if output_semantics == DIRECT_VERTEX_DISPLACEMENT
        else static_sample["raw_laplacian_target"]
    )
    if (
        output_semantics == CURRENT_GRAPH_LAPLACIAN
        and target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN
    ):
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
    raw_target = (
        static_sample["raw_laplacian_target"]
        if output_semantics == CURRENT_GRAPH_LAPLACIAN
        else None
    )
    expert_config = config.get("model", {}).get("oracle_residual_expert", {})
    if not isinstance(expert_config, Mapping):
        raise ValueError("model.oracle_residual_expert must be an object.")
    if bool(expert_config.get("enabled", False)):
        top_fraction = float(expert_config.get("top_fraction", 0.10))
        static_sample["oracle_high_signal_mask"] = _oracle_top_magnitude_mask(
            target, static_sample["valid_scale_mask"], top_fraction
        )
    if recovery_aware.enabled and recovery_aware.solver_mode == "hard_anchor_lambda0":
        static_sample["hard_anchor_indices"] = deterministic_component_anchor_indices(
            static_sample["edge_index"], int(static_sample["vertices"].shape[0])
        )
    recovery_anchor_vertices = None
    if (
        recovery_aware.enabled
        and recovery_aware.anchor_mode == "cached_frozen_vertices"
    ):
        anchor = static_sample.get("recovery_anchor_vertices")
        if not isinstance(anchor, torch.Tensor) or tuple(anchor.shape) != tuple(
            static_sample["vertices"].shape
        ):
            raise ValueError(
                "cached_frozen_vertices recovery requires a same-index "
                "recovery_anchor_vertices tensor."
            )
        recovery_anchor_vertices = anchor.detach()
    if (
        hybrid_single.enabled
        and hybrid_single.operator == "symmetric_cotangent_stiffness"
    ):
        cotangent_edges, cotangent_weights, cotangent_diagonal, _ = (
            build_symmetric_cotangent_stiffness(
                static_sample["vertices"],
                static_sample["faces"],
                relative_area_epsilon=hybrid_single.cotangent_relative_area_epsilon,
            )
        )
        static_sample["cotangent_edge_index"] = cotangent_edges
        static_sample["cotangent_edge_weight"] = cotangent_weights
        static_sample["cotangent_diagonal"] = cotangent_diagonal
    face_count = int(static_sample["faces"].shape[0])
    clean_vertices = None
    if recovery_aware.enabled or hybrid_single.enabled:
        clean_vertices = static_sample.get("clean_reference_vertices")
        clean_faces = static_sample.get("clean_reference_faces")
        compatibility = config.get("dataset", {}).get(
            "clean_reference_compatibility"
        )
        if (
            compatibility == "gt_vertices_same_topology"
            and clean_vertices is None
            and clean_faces is None
        ):
            legacy_vertices = static_sample.get("gt_vertices")
            legacy_faces = static_sample.get("gt_faces")
            target_positions = static_sample.get("target_positions")
            if (
                not isinstance(legacy_vertices, torch.Tensor)
                or not isinstance(legacy_faces, torch.Tensor)
                or not isinstance(target_positions, torch.Tensor)
                or not torch.equal(legacy_faces, static_sample["faces"])
                or not torch.equal(target_positions, legacy_vertices)
            ):
                raise ValueError(
                    "gt_vertices_same_topology compatibility requires exact "
                    "gt/input connectivity and target_positions == gt_vertices."
                )
            clean_vertices = legacy_vertices
            clean_faces = legacy_faces
        if not isinstance(clean_vertices, torch.Tensor) or tuple(
            clean_vertices.shape
        ) != tuple(static_sample["vertices"].shape):
            raise ValueError(
                "Recovery-aware loss requires same-index clean_reference_vertices."
            )
        if not isinstance(clean_faces, torch.Tensor) or not torch.equal(
            clean_faces, static_sample["faces"]
        ):
            raise ValueError(
                "Recovery-aware loss requires clean/input face connectivity to match exactly."
            )
    if static_sample.get("prepared_storage_format") == "lazy_image_paths_v1":
        static_sample.pop("images", None)
    static_sample = _prune_sample_for_training(
        static_sample,
        keep_image_payload=keep_image_payload,
        keep_projection=keep_projection,
    )
    if hybrid_single.enabled or (
        recovery_aware.enabled
        and recovery_aware.primary_supervision == "oriented_face_normals"
    ):
        # Hybrid validation and Arm-F loss need the shared current connectivity.
        # The equality contract above has already proved that these ordered
        # face triplets also define the clean reference; no second GT topology
        # is exposed to the predictor.
        static_sample["faces"] = clean_faces
    return _PreparedObject(
        sample=static_sample,
        training_target=target,
        clipped_target_vertices=clipped_count,
        raw_target=raw_target,
        face_count=face_count,
        used_view_count=int(static_sample["num_views"]),
        clean_vertices=clean_vertices,
        recovery_anchor_vertices=recovery_anchor_vertices,
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
        "oracle_high_signal_mask",
        "hard_anchor_indices",
        "cotangent_edge_index",
        "cotangent_edge_weight",
        "cotangent_diagonal",
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


def _oracle_top_magnitude_mask(
    target: torch.Tensor, valid_mask: torch.Tensor, top_fraction: float
) -> torch.Tensor:
    """Per-mesh clean-target oracle gate for the residual-expert diagnostic."""

    if not 0.0 < top_fraction < 1.0:
        raise ValueError("oracle residual expert top_fraction must be between zero and one.")
    valid = valid_mask.to(dtype=torch.bool, device=target.device)
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_indices.numel() < 1:
        raise ValueError("oracle residual expert requires at least one valid vertex.")
    magnitude = torch.linalg.vector_norm(target.float(), dim=-1)
    descending = valid_indices[
        torch.argsort(magnitude[valid_indices], descending=True, stable=True)
    ]
    selected_count = max(1, int(round(top_fraction * int(valid_indices.numel()))))
    result = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
    result[descending[:selected_count]] = True
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
    moved_raw_target = (
        None
        if prepared.raw_target is None
        else prepared.raw_target.to(device, non_blocking=non_blocking)
    )
    moved_clean_vertices = (
        None
        if prepared.clean_vertices is None
        else prepared.clean_vertices.to(device, non_blocking=non_blocking)
    )
    moved_recovery_anchor_vertices = (
        None
        if prepared.recovery_anchor_vertices is None
        else prepared.recovery_anchor_vertices.to(device, non_blocking=non_blocking)
    )
    return _PreparedObject(
        sample=moved_sample,
        training_target=moved_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=moved_raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=prepared.used_view_count,
        clean_vertices=moved_clean_vertices,
        recovery_anchor_vertices=moved_recovery_anchor_vertices,
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


def _enqueue_cuda_transfer(
    prepared: _PreparedObject,
    config: Mapping[str, Any],
    device: torch.device,
    *,
    cache_on_device: bool,
    decode_images: bool,
    transfer_stream: torch.cuda.Stream,
) -> tuple[
    _PreparedObject,
    tuple[torch.cuda.Event | None, torch.cuda.Event | None],
]:
    """Queue one non-blocking H2D transfer on the dedicated copy stream."""

    with torch.cuda.stream(transfer_stream):
        transfer_start = _start_cuda_timing(device)
        moved = _prepare_item_for_use(
            prepared,
            config,
            device,
            cache_on_device,
            non_blocking=True,
            decode_images=decode_images,
        )
        transfer_events = _finish_cuda_timing(device, transfer_start)
    return moved, transfer_events


def _record_prepared_stream(
    prepared: _PreparedObject, stream: torch.cuda.Stream
) -> None:
    """Keep copy-stream allocations alive until compute-stream work completes."""

    _record_value_stream(prepared.sample, stream)
    _record_value_stream(prepared.training_target, stream)
    _record_value_stream(prepared.raw_target, stream)
    _record_value_stream(prepared.clean_vertices, stream)
    _record_value_stream(prepared.recovery_anchor_vertices, stream)


def _record_value_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            _record_value_stream(nested, stream)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _record_value_stream(nested, stream)


def _materialize_prepared_images(
    prepared: _PreparedObject,
    *,
    dtype: torch.dtype,
    profile_loading: bool = False,
) -> _PreparedObject:
    if "images" in prepared.sample or not prepared.sample.get("image_paths"):
        return prepared
    dataset_root = Path(str(prepared.sample["_dataset_root"]))
    image_paths = resolve_lazy_image_paths(
        prepared.sample["image_paths"], dataset_root
    )
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
        clean_vertices=prepared.clean_vertices,
        recovery_anchor_vertices=prepared.recovery_anchor_vertices,
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
        clean_vertices=prepared.clean_vertices,
        recovery_anchor_vertices=prepared.recovery_anchor_vertices,
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
        clean_vertices=prepared.clean_vertices,
        recovery_anchor_vertices=prepared.recovery_anchor_vertices,
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
        raw_target=(
            item.get("raw_target")
            if isinstance(item.get("raw_target"), torch.Tensor)
            else None
        ),
        face_count=int(item.get("face_count", 0)),
        image_decode_resize_seconds=float(item.get("image_decode_resize_seconds", 0.0)),
        decoded_image_bytes=int(item.get("decoded_image_bytes", 0)),
        used_view_count=int(item.get("used_view_count", sample.get("num_views", 0))),
        clean_vertices=(
            item.get("clean_vertices")
            if isinstance(item.get("clean_vertices"), torch.Tensor)
            else None
        ),
        recovery_anchor_vertices=(
            item.get("recovery_anchor_vertices")
            if isinstance(item.get("recovery_anchor_vertices"), torch.Tensor)
            else None
        ),
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
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    worker_items = tuple(
        _PreparedObject(
            sample=item.sample,
            training_target=item.training_target,
            clipped_target_vertices=item.clipped_target_vertices,
            raw_target=item.raw_target,
            face_count=item.face_count,
            used_view_count=item.used_view_count,
            clean_vertices=item.clean_vertices,
            recovery_anchor_vertices=item.recovery_anchor_vertices,
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
        len(dataset),
        shuffle=shuffle,
        generator=generator,
        rank=rank,
        world_size=world_size,
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
    multiprocessing_sharing_strategy = str(
        loading.get("multiprocessing_sharing_strategy", "file_descriptor")
    )
    cuda_prefetch = bool(loading.get("cuda_prefetch", True))
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
    if multiprocessing_sharing_strategy not in MULTIPROCESSING_SHARING_STRATEGIES:
        allowed = ", ".join(sorted(MULTIPROCESSING_SHARING_STRATEGIES))
        raise ValueError(
            "data_loading.multiprocessing_sharing_strategy must be one of "
            f"{allowed}."
        )
    return _DataLoaderSettings(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_sharing_strategy=multiprocessing_sharing_strategy,
        cuda_prefetch=cuda_prefetch,
        train_views_per_sample=train_views_per_sample,
        validation_views_per_sample=validation_views_per_sample,
    )


def _configure_multiprocessing_sharing(settings: _DataLoaderSettings) -> None:
    """Avoid per-tensor file descriptor growth in worker-backed DataLoaders."""

    if settings.num_workers == 0:
        return
    if (
        torch.multiprocessing.get_sharing_strategy()
        != settings.multiprocessing_sharing_strategy
    ):
        torch.multiprocessing.set_sharing_strategy(
            settings.multiprocessing_sharing_strategy
        )


def _cuda_transfer_overlap_enabled(
    device: torch.device,
    *,
    cache_on_device: bool,
    settings: _DataLoaderSettings,
) -> bool:
    return bool(
        device.type == "cuda"
        and not cache_on_device
        and settings.pin_memory
        and settings.cuda_prefetch
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


def _direct_vertex_residual_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Exact Arm-E loss ``mean_i ||delta_v_pred-delta_v_gt||_2^2``."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("direct vertex prediction and target must have shape [N, 3].")
    return (prediction - target).square().sum(dim=-1).mean()


def _loss_kwargs(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss_type": str(training.get("loss", "huber")),
        "huber_delta": float(training.get("huber_delta", 0.01)),
        "charbonnier_epsilon": float(training.get("charbonnier_epsilon", 1e-3)),
        "target_magnitude_weight_lambda": float(
            training.get("target_magnitude_weight_lambda", 0.0)
        ),
    }


def _recovery_aware_geometry_settings(
    config: Mapping[str, Any],
) -> _RecoveryAwareGeometrySettings:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object.")
    raw = training.get("recovery_aware_geometry_loss", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training.recovery_aware_geometry_loss must be an object.")
    settings = _RecoveryAwareGeometrySettings(
        enabled=bool(raw.get("enabled", False)),
        regularization=float(raw.get("lambda", 1e-2)),
        beta=float(raw.get("beta", 0.0)),
        prediction_loss_weight=float(raw.get("prediction_loss_weight", 1.0)),
        maximum_iterations=int(raw.get("maximum_iterations", 128)),
        tolerance=float(raw.get("tolerance", 1e-5)),
        runtime_diagnostics=bool(raw.get("runtime_diagnostics", False)),
        compute_dtype=str(raw.get("compute_dtype", "float32")),
        adaptive_lambda=bool(raw.get("adaptive_lambda", False)),
        solver_mode=str(raw.get("solver", "regularized_sparse")),
        primary_supervision=str(raw.get("primary_supervision", "prediction_space")),
        normal_epsilon=float(raw.get("normal_epsilon", 1e-12)),
        anchor_mode=str(raw.get("anchor_mode", "initial_vertices")),
    )
    if settings.solver_mode not in {"regularized_sparse", "hard_anchor_lambda0"}:
        raise ValueError(
            "recovery-aware solver must be regularized_sparse or hard_anchor_lambda0."
        )
    if settings.solver_mode == "regularized_sparse":
        if settings.regularization <= 0:
            raise ValueError("regularized recovery-aware geometry lambda must be positive.")
    elif settings.regularization != 0:
        raise ValueError("hard_anchor_lambda0 requires lambda exactly equal to zero.")
    if settings.solver_mode == "hard_anchor_lambda0" and settings.adaptive_lambda:
        raise ValueError("hard_anchor_lambda0 cannot use adaptive lambda.")
    if settings.beta < 0:
        raise ValueError("recovery-aware geometry beta must be non-negative.")
    if settings.prediction_loss_weight < 0:
        raise ValueError(
            "recovery-aware prediction_loss_weight must be non-negative."
        )
    if settings.enabled and settings.beta <= 0:
        raise ValueError("enabled recovery-aware geometry loss requires beta > 0.")
    if settings.maximum_iterations < 1:
        raise ValueError("recovery-aware maximum_iterations must be positive.")
    if settings.tolerance <= 0:
        raise ValueError("recovery-aware tolerance must be positive.")
    if settings.compute_dtype not in {"float32", "float64"}:
        raise ValueError("recovery-aware compute_dtype must be float32 or float64.")
    if settings.primary_supervision not in RECOVERY_PRIMARY_SUPERVISION:
        raise ValueError(
            "recovery-aware primary_supervision must be prediction_space or "
            "oriented_face_normals."
        )
    if settings.normal_epsilon <= 0:
        raise ValueError("recovery-aware normal_epsilon must be positive.")
    if settings.anchor_mode not in RECOVERY_ANCHOR_MODES:
        raise ValueError(
            "recovery-aware anchor_mode must be initial_vertices or "
            "cached_frozen_vertices."
        )
    if (
        settings.solver_mode == "hard_anchor_lambda0"
        and settings.anchor_mode != "initial_vertices"
    ):
        raise ValueError("hard_anchor_lambda0 only supports initial_vertices.")
    return settings


def _hybrid_single_geometry_settings(
    config: Mapping[str, Any],
) -> _HybridSingleGeometrySettings:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object.")
    raw = training.get("hybrid_single_geometry_loss", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training.hybrid_single_geometry_loss must be an object.")
    settings = _HybridSingleGeometrySettings(
        enabled=bool(raw.get("enabled", False)),
        regularization=float(raw.get("lambda", 3e-2)),
        maximum_iterations=int(raw.get("maximum_iterations", 2048)),
        tolerance=float(raw.get("tolerance", 1e-6)),
        runtime_diagnostics=bool(raw.get("runtime_diagnostics", False)),
        compute_dtype=str(raw.get("compute_dtype", "float64")),
        validation_surface_samples=int(raw.get("validation_surface_samples", 3000)),
        operator=str(raw.get("operator", "uniform_random_walk")),
        cotangent_relative_area_epsilon=float(
            raw.get("cotangent_relative_area_epsilon", 1e-12)
        ),
    )
    if settings.regularization <= 0:
        raise ValueError("hybrid single-geometry lambda must be positive.")
    if settings.maximum_iterations < 1 or settings.tolerance <= 0:
        raise ValueError("hybrid PCG iteration budget and tolerance must be positive.")
    if settings.compute_dtype not in {"float32", "float64"}:
        raise ValueError("hybrid compute_dtype must be float32 or float64.")
    if settings.validation_surface_samples < 1:
        raise ValueError("hybrid validation_surface_samples must be positive.")
    if settings.operator not in {
        "uniform_random_walk",
        "symmetric_cotangent_stiffness",
    }:
        raise ValueError(
            "hybrid operator must be uniform_random_walk or "
            "symmetric_cotangent_stiffness."
        )
    if settings.cotangent_relative_area_epsilon <= 0:
        raise ValueError("cotangent_relative_area_epsilon must be positive.")
    return settings


def _hybrid_single_geometry_loss(
    laplacian_prediction: torch.Tensor,
    direct_displacement_prediction: torch.Tensor,
    prepared: _PreparedObject,
    settings: _HybridSingleGeometrySettings,
    *,
    with_audit: bool,
) -> tuple[torch.Tensor, torch.Tensor, ConjugateGradientAudit | None, torch.Tensor]:
    clean_vertices = prepared.clean_vertices
    if clean_vertices is None:
        raise RuntimeError("Hybrid single-geometry loss requires loss-side clean vertices.")
    if direct_displacement_prediction.shape != laplacian_prediction.shape:
        raise ValueError("Hybrid Laplacian and direct outputs must both have shape [N, 3].")
    sample = prepared.sample
    solve_dtype = torch.float64 if settings.compute_dtype == "float64" else torch.float32
    v_direct = sample["vertices"].to(dtype=solve_dtype) + direct_displacement_prediction.to(
        dtype=solve_dtype
    )
    arguments = dict(
        regularization=settings.regularization,
        maximum_iterations=settings.maximum_iterations,
        tolerance=settings.tolerance,
    )
    if settings.operator == "uniform_random_walk":
        if with_audit:
            recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
                laplacian_prediction.to(dtype=solve_dtype),
                v_direct,
                sample["edge_index"],
                sample["vertex_degree"].to(dtype=solve_dtype),
                **arguments,
            )
        else:
            recovered = differentiable_regularized_sparse_recovery(
                laplacian_prediction.to(dtype=solve_dtype),
                v_direct,
                sample["edge_index"],
                sample["vertex_degree"].to(dtype=solve_dtype),
                **arguments,
            )
            audit = None
    else:
        cotangent_arguments = (
            sample["cotangent_edge_index"],
            sample["cotangent_edge_weight"].to(dtype=solve_dtype),
            sample["cotangent_diagonal"].to(dtype=solve_dtype),
        )
        if with_audit:
            recovered, audit = differentiable_cotangent_sparse_recovery_with_audit(
                laplacian_prediction.to(dtype=solve_dtype),
                v_direct,
                *cotangent_arguments,
                **arguments,
            )
        else:
            recovered = differentiable_cotangent_sparse_recovery(
                laplacian_prediction.to(dtype=solve_dtype),
                v_direct,
                *cotangent_arguments,
                **arguments,
            )
            audit = None
    loss = (
        (recovered - clean_vertices.to(dtype=solve_dtype))
        .square()
        .sum(dim=-1)
        .mean()
    )
    return loss, recovered, audit, v_direct


def _recovery_refine_loss(
    prediction: torch.Tensor,
    prepared: _PreparedObject,
    settings: _RecoveryAwareGeometrySettings,
    regularization: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_vertices = prepared.clean_vertices
    if clean_vertices is None:
        raise RuntimeError("Recovery-aware loss requires loss-side clean vertices.")
    sample = prepared.sample
    solve_dtype = torch.float64 if settings.compute_dtype == "float64" else torch.float32
    if settings.solver_mode == "hard_anchor_lambda0":
        anchors = sample.get("hard_anchor_indices")
        if not isinstance(anchors, torch.Tensor):
            raise RuntimeError("hard_anchor_lambda0 requires graph-derived anchors.")
        recovered = differentiable_hard_anchor_sparse_recovery(
            prediction.to(dtype=solve_dtype),
            sample["vertices"].to(dtype=solve_dtype),
            sample["edge_index"],
            sample["vertex_degree"].to(dtype=solve_dtype),
            anchors,
            maximum_iterations=settings.maximum_iterations,
            tolerance=settings.tolerance,
        )
    else:
        anchor = _recovery_anchor(prepared, settings, solve_dtype)
        recovered = differentiable_regularized_sparse_recovery(
            prediction.to(dtype=solve_dtype),
            anchor,
            sample["edge_index"],
            sample["vertex_degree"].to(dtype=solve_dtype),
            regularization=(
                settings.regularization if regularization is None else regularization
            ),
            maximum_iterations=settings.maximum_iterations,
            tolerance=settings.tolerance,
        )
    refine_loss = (
        (recovered - clean_vertices.to(dtype=solve_dtype))
        .square()
        .sum(dim=-1)
        .mean()
    )
    return refine_loss, recovered


def _recovery_refine_loss_with_audit(
    prediction: torch.Tensor,
    prepared: _PreparedObject,
    settings: _RecoveryAwareGeometrySettings,
    regularization: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ConjugateGradientAudit]:
    clean_vertices = prepared.clean_vertices
    if clean_vertices is None:
        raise RuntimeError("Recovery-aware loss requires loss-side clean vertices.")
    sample = prepared.sample
    solve_dtype = torch.float64 if settings.compute_dtype == "float64" else torch.float32
    if settings.solver_mode == "hard_anchor_lambda0":
        anchors = sample.get("hard_anchor_indices")
        if not isinstance(anchors, torch.Tensor):
            raise RuntimeError("hard_anchor_lambda0 requires graph-derived anchors.")
        recovered, audit = differentiable_hard_anchor_sparse_recovery_with_audit(
            prediction.to(dtype=solve_dtype),
            sample["vertices"].to(dtype=solve_dtype),
            sample["edge_index"],
            sample["vertex_degree"].to(dtype=solve_dtype),
            anchors,
            maximum_iterations=settings.maximum_iterations,
            tolerance=settings.tolerance,
        )
    else:
        anchor = _recovery_anchor(prepared, settings, solve_dtype)
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            prediction.to(dtype=solve_dtype),
            anchor,
            sample["edge_index"],
            sample["vertex_degree"].to(dtype=solve_dtype),
            regularization=(
                settings.regularization if regularization is None else regularization
            ),
            maximum_iterations=settings.maximum_iterations,
            tolerance=settings.tolerance,
        )
    refine_loss = (
        (recovered - clean_vertices.to(dtype=solve_dtype))
        .square()
        .sum(dim=-1)
        .mean()
    )
    return refine_loss, recovered, audit


def _recovery_anchor(
    prepared: _PreparedObject,
    settings: _RecoveryAwareGeometrySettings,
    solve_dtype: torch.dtype,
) -> torch.Tensor:
    """Return the loss-side recovery anchor without exposing it to the predictor."""

    sample = prepared.sample
    if settings.anchor_mode == "initial_vertices":
        return sample["vertices"].to(dtype=solve_dtype)
    anchor = prepared.recovery_anchor_vertices
    if not isinstance(anchor, torch.Tensor):
        raise RuntimeError(
            "cached_frozen_vertices recovery requires recovery_anchor_vertices."
        )
    if tuple(anchor.shape) != tuple(sample["vertices"].shape):
        raise RuntimeError("Frozen recovery anchor shape differs from input vertices.")
    # The frozen positional prediction is loss-side data.  Detaching here is a
    # second guard in addition to the on-disk cache and frozen-model audit.
    return anchor.detach().to(dtype=solve_dtype)


def _area_weighted_oriented_face_normal_loss(
    predicted_vertices: torch.Tensor,
    clean_vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """GT-area-weighted, orientation-sensitive face-normal cosine loss."""
    if predicted_vertices.ndim != 2 or predicted_vertices.shape[1] != 3:
        raise ValueError("predicted_vertices must have shape [N, 3].")
    if clean_vertices.shape != predicted_vertices.shape:
        raise ValueError("clean and predicted vertices must have identical shape.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if faces.numel() == 0:
        raise ValueError("face-normal supervision requires at least one face.")
    if epsilon <= 0:
        raise ValueError("normal epsilon must be positive.")
    face_index = faces.to(device=predicted_vertices.device, dtype=torch.long)
    if int(face_index.min()) < 0 or int(face_index.max()) >= predicted_vertices.shape[0]:
        raise ValueError("faces contain an out-of-range vertex index.")

    predicted_triangles = predicted_vertices[face_index]
    clean = clean_vertices.to(
        device=predicted_vertices.device, dtype=predicted_vertices.dtype
    )
    clean_triangles = clean[face_index]
    predicted_cross = torch.cross(
        predicted_triangles[:, 1] - predicted_triangles[:, 0],
        predicted_triangles[:, 2] - predicted_triangles[:, 0],
        dim=-1,
    )
    clean_cross = torch.cross(
        clean_triangles[:, 1] - clean_triangles[:, 0],
        clean_triangles[:, 2] - clean_triangles[:, 0],
        dim=-1,
    )
    predicted_normals = predicted_cross / (
        torch.linalg.vector_norm(predicted_cross, dim=-1, keepdim=True) + epsilon
    )
    clean_cross_norm = torch.linalg.vector_norm(clean_cross, dim=-1, keepdim=True)
    clean_normals = clean_cross / (clean_cross_norm + epsilon)
    # The loss weights are fixed GT areas; predicted areas never enter them.
    gt_area = (0.5 * clean_cross_norm.squeeze(-1)).detach()
    area_sum = gt_area.sum()
    if not bool(torch.isfinite(area_sum)) or float(area_sum) <= 0:
        raise ValueError("clean mesh has non-finite or zero total triangle area.")
    weights = gt_area / area_sum
    oriented_cosine = (predicted_normals * clean_normals).sum(dim=-1)
    return (weights * (1.0 - oriented_cosine)).sum()


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
        clean_vertices=prepared.clean_vertices,
        recovery_anchor_vertices=prepared.recovery_anchor_vertices,
    )


def _with_local_query_jitter(
    prepared: _PreparedObject,
    settings: LocalQueryJitterSettings,
    *,
    base_seed: int,
    epoch: int,
) -> _PreparedObject:
    if not settings.enabled:
        return prepared
    return _PreparedObject(
        sample=apply_local_query_jitter(
            prepared.sample, settings, base_seed=base_seed, epoch=epoch
        ),
        training_target=prepared.training_target,
        clipped_target_vertices=prepared.clipped_target_vertices,
        raw_target=prepared.raw_target,
        face_count=prepared.face_count,
        image_decode_resize_seconds=prepared.image_decode_resize_seconds,
        decoded_image_bytes=prepared.decoded_image_bytes,
        used_view_count=prepared.used_view_count,
        clean_vertices=prepared.clean_vertices,
        recovery_anchor_vertices=prepared.recovery_anchor_vertices,
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


def _distributed_optional_mean(
    value: float | None, context: DistributedContext
) -> float | None:
    present = reduce_scalar(
        0 if value is None else 1, context, reduction="sum"
    )
    if present == 0:
        return None
    if present != context.world_size:
        raise RuntimeError("Optional distributed metric is missing on a subset of ranks.")
    return reduce_scalar(float(value), context, reduction="mean")


def _unwrap_model(model: nn.Module) -> LearnedLaplacianModel:
    result = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(result, LearnedLaplacianModel):
        raise TypeError("Expected a LearnedLaplacianModel or its DDP wrapper.")
    return result


def _parameter_gradient_diagnostics(
    parameters: Iterable[torch.nn.Parameter], device: torch.device
) -> tuple[float, int]:
    """Return an observational gradient norm and non-finite count."""

    squared = torch.zeros((), device=device)
    nonfinite = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        nonfinite += int((~torch.isfinite(gradient)).sum().item())
        squared = squared + gradient.float().square().sum()
    return float(torch.sqrt(squared).item()), nonfinite


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
    model: nn.Module,
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
    base_model = _unwrap_model(model)
    decode_images = (
        base_model.input_mode != "coarse_only" and not base_model.zero_images
    )
    target_mode, epsilon = _target_settings(config)
    output_semantics = prediction_semantics(config)
    prediction_loss_space = _prediction_loss_space(config.get("training", {}))
    confidence_settings = _confidence_settings(config)
    recovery_aware = _recovery_aware_geometry_settings(config)
    hybrid_single = _hybrid_single_geometry_settings(config)
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
        direct_prediction = model_output.direct_vertex_displacement_prediction
        if hybrid_single.enabled:
            if direct_prediction is None:
                raise RuntimeError("Hybrid evaluation requires the direct branch output.")
            direct_prediction = direct_prediction.float()
        loss_prediction, loss_target = _prediction_loss_inputs(
            prediction,
            prepared,
            output_semantics=output_semantics,
            target_mode=target_mode,
            epsilon=epsilon,
            prediction_loss_space=prediction_loss_space,
        )
        if output_semantics == DIRECT_VERTEX_DISPLACEMENT:
            loss = _direct_vertex_residual_mse(loss_prediction, loss_target)
        else:
            loss = weighted_robust_laplacian_loss(
                loss_prediction,
                loss_target,
                prepared.sample["target_confidence"].float(),
                **loss_kwargs,
            )
        prediction_space_loss = loss
        refine_loss = None
        operator_normal_loss = None
        recovered_vertex_rms = None
        recovered_vertices = None
        hybrid_geometry = None
        objective = loss
        if recovery_aware.enabled:
            refine_loss, recovered_vertices = _recovery_refine_loss(
                prediction, prepared, recovery_aware
            )
            recovered_vertex_rms = torch.sqrt(refine_loss)
            if recovery_aware.primary_supervision == "oriented_face_normals":
                clean_vertices = prepared.clean_vertices
                if clean_vertices is None:
                    raise RuntimeError(
                        "Oriented face-normal validation requires clean vertices."
                    )
                operator_normal_loss = _area_weighted_oriented_face_normal_loss(
                    recovered_vertices,
                    clean_vertices,
                    prepared.sample["faces"],
                    epsilon=recovery_aware.normal_epsilon,
                )
                loss = operator_normal_loss
            objective = (
                recovery_aware.prediction_loss_weight * loss
                + recovery_aware.beta * refine_loss
            )
        if hybrid_single.enabled:
            assert direct_prediction is not None
            refine_loss, recovered_vertices, _, v_direct = _hybrid_single_geometry_loss(
                prediction,
                direct_prediction,
                prepared,
                hybrid_single,
                with_audit=False,
            )
            recovered_vertex_rms = torch.sqrt(refine_loss)
            objective = refine_loss
            faces_np = prepared.sample["faces"].detach().cpu().numpy().astype(np.int64)
            clean_vertices = prepared.clean_vertices
            if clean_vertices is None:
                raise RuntimeError("Hybrid evaluation requires clean vertices.")
            hybrid_geometry = evaluate_mesh_geometry(
                Mesh(
                    recovered_vertices.detach().cpu().numpy(), faces_np
                ).ensure_normals(),
                Mesh(clean_vertices.detach().cpu().numpy(), faces_np).ensure_normals(),
                surface_samples=hybrid_single.validation_surface_samples,
                seed=7,
                fscore_threshold=0.01,
            )
        loss_value = float(objective.item())
        laplacian_loss_value = float(prediction_space_loss.item())
        exact_loss, perturbed_loss = _query_subset_losses(
            loss_prediction,
            loss_target,
            prepared.sample["target_confidence"].float(),
            prepared.sample.get("query_is_exact"),
            loss_kwargs,
        )
        losses.append(loss_value)
        valid_mask = prepared.sample["valid_scale_mask"]
        target_metrics = laplacian_prediction_metrics(
            prediction, prepared.training_target.float(), valid_mask=valid_mask
        )
        if output_semantics == DIRECT_VERTEX_DISPLACEMENT:
            raw_prediction = prediction
            raw_metrics = None
        elif target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
            raw_prediction = denormalize_laplacian_by_edge_scale(
                prediction, prepared.sample["local_edge_length"], eps=epsilon
            )
        else:
            raw_prediction = prediction
        if output_semantics == CURRENT_GRAPH_LAPLACIAN:
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
                loss_prediction,
                loss_target,
                valid_mask=valid_mask,
                quantile_bins=confidence_settings["quantile_bins"],
            )
            confidence_prediction_cpu = confidence_prediction.detach().cpu()
        visible = model_output.valid_views.any(dim=0)
        metrics[sample_id] = {
            "loss": loss_value,
            "laplacian_loss": laplacian_loss_value,
            "recovery_refine_loss": (
                None if refine_loss is None else float(refine_loss.item())
            ),
            "operator_normal_loss": (
                None
                if operator_normal_loss is None
                else float(operator_normal_loss.item())
            ),
            "recovered_vertex_rms": (
                None
                if recovered_vertex_rms is None
                else float(recovered_vertex_rms.item())
            ),
            "hybrid_chamfer": (
                None if hybrid_geometry is None else float(hybrid_geometry["chamfer"])
            ),
            "hybrid_geometry": hybrid_geometry,
            "prediction_semantics": output_semantics,
            "prediction_loss_space": prediction_loss_space,
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
            if output_semantics == DIRECT_VERTEX_DISPLACEMENT:
                np.save(
                    prediction_dir / f"{safe_id}_displacement.npy",
                    raw_prediction_cpu.numpy(),
                )
            else:
                np.save(
                    prediction_dir / f"{safe_id}_raw_delta.npy",
                    raw_prediction_cpu.numpy(),
                )
            if confidence_prediction_cpu is not None:
                np.save(
                    prediction_dir / f"{safe_id}_confidence.npy",
                    confidence_prediction_cpu.numpy(),
                )
            if hybrid_single.enabled:
                assert direct_prediction is not None and recovered_vertices is not None
                np.save(
                    prediction_dir / f"{safe_id}_direct_displacement.npy",
                    direct_prediction.detach().cpu().numpy(),
                )
                np.save(
                    prediction_dir / f"{safe_id}_hybrid_vertices.npy",
                    recovered_vertices.detach().cpu().numpy(),
                )
    if not losses:
        raise ValueError("Cannot evaluate an empty dataset split.")
    return float(np.mean(losses)), metrics


def _save_multi_checkpoint(
    path: Path,
    model: nn.Module,
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
    base_model = _unwrap_model(model)
    payload = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "model_config": base_model.architecture_config(),
        "model_state_dict": base_model.state_dict(),
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
    model: nn.Module,
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
            "distributed_world_size": current_distributed_context().world_size,
        },
    )
    return path


def _save_resumable_checkpoint(
    path: Path,
    model: nn.Module,
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
            "distributed_world_size": current_distributed_context().world_size,
        },
    )

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None
    device: torch.device = torch.device("cpu")
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str) -> DistributedContext:
    """Initialize an env:// process group when launched by torchrun."""

    world_size = _environment_integer("WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive.")
    if world_size == 1:
        return DistributedContext(device=_single_process_device(requested_device))
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")

    rank = _environment_integer("RANK", required=True)
    local_rank = _environment_integer("LOCAL_RANK", required=True)
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK={rank} is outside WORLD_SIZE={world_size}.")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requires CUDA on every rank.")
        device_count = torch.cuda.device_count()
        if not 0 <= local_rank < device_count:
            raise ValueError(
                f"LOCAL_RANK={local_rank} is outside the {device_count} visible CUDA devices."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        default_backend = "nccl"
    elif requested_device == "cpu":
        device = torch.device("cpu")
        default_backend = "gloo"
    else:
        raise ValueError("Distributed training supports only CPU and CUDA devices.")

    backend = os.environ.get("MLR_DISTRIBUTED_BACKEND", default_backend)
    initialized_here = False
    if not dist.is_initialized():
        timeout_seconds = _environment_integer("MLR_DISTRIBUTED_TIMEOUT_SECONDS", 1800)
        if timeout_seconds < 1:
            raise ValueError("MLR_DISTRIBUTED_TIMEOUT_SECONDS must be positive.")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
        initialized_here = True
    actual_rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    if actual_rank != rank or actual_world_size != world_size:
        raise RuntimeError(
            "Initialized process-group identity does not match torchrun environment."
        )
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=str(dist.get_backend()),
        device=device,
        initialized_here=initialized_here,
    )


def current_distributed_context(device: torch.device | None = None) -> DistributedContext:
    if not dist.is_available() or not dist.is_initialized():
        return DistributedContext(device=device or torch.device("cpu"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _environment_integer("LOCAL_RANK", 0)
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=str(dist.get_backend()),
        device=device or _current_device(local_rank),
    )


def distributed_barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(
    value: float | int,
    context: DistributedContext,
    *,
    reduction: str,
) -> float:
    if not context.enabled:
        return float(value)
    operations = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    if reduction not in operations:
        raise ValueError("reduction must be sum, mean, max, or min.")
    tensor = torch.tensor(float(value), dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=operations[reduction])
    if reduction == "mean":
        tensor /= context.world_size
    return float(tensor.item())


def broadcast_scalar(
    value: float | int,
    context: DistributedContext,
    *,
    source: int = 0,
) -> float:
    if not context.enabled:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=context.device)
    dist.broadcast(tensor, src=source)
    return float(tensor.item())


def _single_process_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")
    return device


def _current_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _environment_integer(
    name: str,
    default: int | None = None,
    *,
    required: bool = False,
) -> int:
    raw = os.environ.get(name)
    if raw is None:
        if required:
            raise RuntimeError(f"{name} is required for distributed training.")
        if default is None:
            raise RuntimeError(f"{name} is not set.")
        return int(default)
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from error

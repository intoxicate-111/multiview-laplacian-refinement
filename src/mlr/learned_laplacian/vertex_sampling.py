from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch


FULL_VERTEX_EXPOSURE = "full"
HIGH_LAPLACIAN_MIXTURE = "high_laplacian_mixture_v1"
LAPLACIAN_MAGNITUDE_MIXTURE = "laplacian_magnitude_mixture_v1"


@dataclass(frozen=True)
class VertexSamplingSettings:
    mode: str
    sample_count_ratio: float
    uniform_fraction: float
    top_10_fraction: float
    top_1_to_10_fraction: float
    top_1_fraction: float
    bottom_90_fraction: float


@dataclass(frozen=True)
class VertexSamplingResult:
    indices: torch.Tensor | None
    diagnostics: dict[str, int | float | str]


def vertex_sampling_settings(config: Mapping[str, Any]) -> VertexSamplingSettings:
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object.")
    raw = training.get("vertex_sampling", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training.vertex_sampling must be an object.")
    settings = VertexSamplingSettings(
        mode=str(raw.get("mode", FULL_VERTEX_EXPOSURE)),
        sample_count_ratio=float(raw.get("sample_count_ratio", 1.0)),
        uniform_fraction=float(raw.get("uniform_fraction", 0.5)),
        top_10_fraction=float(raw.get("top_10_fraction", 0.25)),
        top_1_to_10_fraction=float(raw.get("top_1_to_10_fraction", 0.25)),
        top_1_fraction=float(raw.get("top_1_fraction", 0.0)),
        bottom_90_fraction=float(raw.get("bottom_90_fraction", 0.0)),
    )
    if settings.mode not in {
        FULL_VERTEX_EXPOSURE,
        HIGH_LAPLACIAN_MIXTURE,
        LAPLACIAN_MAGNITUDE_MIXTURE,
    }:
        raise ValueError(
            "training.vertex_sampling.mode must be 'full', "
            f"{HIGH_LAPLACIAN_MIXTURE!r}, or {LAPLACIAN_MAGNITUDE_MIXTURE!r}."
        )
    if settings.sample_count_ratio <= 0:
        raise ValueError("training.vertex_sampling.sample_count_ratio must be positive.")
    fractions = (
        settings.uniform_fraction,
        settings.top_10_fraction,
        settings.top_1_to_10_fraction,
        settings.top_1_fraction,
        settings.bottom_90_fraction,
    )
    if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("training.vertex_sampling fractions must be non-negative and sum to 1.")
    return settings


def sample_training_vertices(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    settings: VertexSamplingSettings,
    *,
    sample_id: str,
    base_seed: int,
    epoch: int,
) -> VertexSamplingResult:
    """Select loss rows without changing the forward graph or loss formula."""

    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target must have shape [N, 3].")
    if tuple(valid_mask.shape) != (target.shape[0],):
        raise ValueError("valid_mask must have shape [N].")
    valid_mask = valid_mask.to(device=target.device, dtype=torch.bool)
    if settings.mode == FULL_VERTEX_EXPOSURE:
        valid_count = int(valid_mask.sum().item())
        return VertexSamplingResult(
            indices=None,
            diagnostics={
                "mode": FULL_VERTEX_EXPOSURE,
                "valid_vertex_count": valid_count,
                "selected_row_count": valid_count,
            },
        )
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_indices.numel() < 2:
        raise ValueError("Vertex sampling requires at least two valid vertices.")

    magnitude = torch.linalg.vector_norm(target.float(), dim=-1)
    order = valid_indices[torch.argsort(magnitude[valid_indices], descending=True)]
    valid_count = int(order.numel())
    top_10_count = min(max(2, int(round(0.10 * valid_count))), valid_count)
    top_1_count = min(max(1, int(round(0.01 * valid_count))), top_10_count - 1)
    top_10 = order[:top_10_count]
    top_1 = order[:top_1_count]
    top_1_to_10 = order[top_1_count:top_10_count]
    bottom_90 = order[top_10_count:]
    if top_1_to_10.numel() == 0:
        raise ValueError("Vertex sampling requires a non-empty top 1-10% group.")

    total = max(1, int(round(settings.sample_count_ratio * valid_count)))
    fractions = (
        settings.uniform_fraction,
        settings.top_10_fraction,
        settings.top_1_to_10_fraction,
        settings.top_1_fraction,
        settings.bottom_90_fraction,
    )
    draw_counts = [int(round(fraction * total)) for fraction in fractions]
    residual_slot = max(index for index, fraction in enumerate(fractions) if fraction > 0)
    draw_counts[residual_slot] += total - sum(draw_counts)
    (
        uniform_count,
        top_10_draw_count,
        top_1_to_10_draw_count,
        top_1_draw_count,
        bottom_90_draw_count,
    ) = draw_counts
    generator = torch.Generator(device=target.device)
    generator.manual_seed(_sample_epoch_seed(sample_id, base_seed, epoch))

    def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return pool.new_empty((0,), dtype=torch.long)
        if pool.numel() == 0:
            raise ValueError("Configured vertex-sampling pool is empty.")
        positions = torch.randint(
            int(pool.numel()),
            (count,),
            generator=generator,
            device=target.device,
        )
        return pool.index_select(0, positions)

    indices = torch.cat(
        (
            draw(valid_indices, uniform_count),
            draw(top_10, top_10_draw_count),
            draw(top_1_to_10, top_1_to_10_draw_count),
            draw(top_1, top_1_draw_count),
            draw(bottom_90, bottom_90_draw_count),
        )
    )
    shuffle = torch.randperm(indices.numel(), generator=generator, device=target.device)
    indices = indices.index_select(0, shuffle)
    return VertexSamplingResult(
        indices=indices,
        diagnostics={
            "mode": settings.mode,
            "valid_vertex_count": valid_count,
            "selected_row_count": int(indices.numel()),
            "uniform_draw_count": uniform_count,
            "top_10_draw_count": top_10_draw_count,
            "top_1_to_10_draw_count": top_1_to_10_draw_count,
            "top_1_draw_count": top_1_draw_count,
            "bottom_90_draw_count": bottom_90_draw_count,
            "unique_selected_vertices": int(torch.unique(indices).numel()),
        },
    )


def _sample_epoch_seed(sample_id: str, base_seed: int, epoch: int) -> int:
    digest = hashlib.sha256(("vertex_sampling:" + sample_id).encode("utf-8")).digest()
    sample_component = int.from_bytes(digest[:8], "little", signed=False)
    return int(
        (sample_component + int(base_seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    )

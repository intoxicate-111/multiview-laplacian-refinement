from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch


FIXED_SYNTHETIC_CURRENT_MODE = "fixed_synthetic_current_graph_v1"
DEFAULT_STD_H = 0.003
CLIP_SIGMA = 3.0
MAX_STD_H = 0.01


@dataclass(frozen=True)
class LocalQueryJitterSettings:
    enabled: bool
    std_h: float


def local_query_jitter_settings(
    config: Mapping[str, Any],
) -> LocalQueryJitterSettings:
    raw = config.get("local_query_jitter", {})
    if not isinstance(raw, Mapping):
        raise ValueError("local_query_jitter must be an object.")
    settings = LocalQueryJitterSettings(
        enabled=bool(raw.get("enabled", False)),
        std_h=float(raw.get("std_h", DEFAULT_STD_H)),
    )
    if settings.std_h < 0:
        raise ValueError("local_query_jitter.std_h must be non-negative.")
    if settings.std_h > MAX_STD_H:
        raise ValueError(
            f"local_query_jitter.std_h must not exceed {MAX_STD_H}."
        )
    if settings.enabled and settings.std_h == 0:
        raise ValueError("Enabled local query jitter requires std_h > 0.")
    if raw.get("scope", "training_only") != "training_only":
        raise ValueError("local_query_jitter.scope must be 'training_only'.")
    if raw.get(
        "distribution", "isotropic_gaussian_xyz_clipped_at_3sigma"
    ) != "isotropic_gaussian_xyz_clipped_at_3sigma":
        raise ValueError("Unsupported local_query_jitter.distribution.")
    declared_maximum = float(raw.get("maximum_offset_h", CLIP_SIGMA * settings.std_h))
    if abs(declared_maximum - CLIP_SIGMA * settings.std_h) > 1e-12:
        raise ValueError("local_query_jitter.maximum_offset_h must equal 3 * std_h.")
    return settings


def validate_local_query_jitter_contract(sample: Mapping[str, Any]) -> None:
    """Require the fixed current graph/proxy-target dataset used by this ablation."""

    metadata = sample.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get(
        "query_training_mode"
    ) != FIXED_SYNTHETIC_CURRENT_MODE:
        raise ValueError(
            "local_query_jitter.enabled=true requires samples prepared with "
            f"query_training_mode={FIXED_SYNTHETIC_CURRENT_MODE!r}."
        )
    if metadata.get("proxy_definition") != (
        "P_proxy=source_gt_vertices_with_exact_same_topology"
    ):
        raise ValueError("Local query jitter requires the frozen stored proxy contract.")
    if metadata.get("target_constructor") != "delta_target=L_current@P_proxy":
        raise ValueError("Local query jitter requires delta_target=L_current@P_proxy.")
    for field in (
        "vertices",
        "initial_laplacian",
        "local_edge_length",
        "raw_laplacian_target",
        "normalized_laplacian_target",
    ):
        if not isinstance(sample.get(field), torch.Tensor):
            raise ValueError(f"Local query jitter requires tensor field {field!r}.")


def apply_local_query_jitter(
    sample: Mapping[str, Any],
    settings: LocalQueryJitterSettings,
    *,
    base_seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Jitter only query positions; graph, scale, operator, proxy, and targets stay fixed."""

    result = dict(sample)
    vertices = sample["vertices"]
    if not settings.enabled:
        return result

    generator = torch.Generator(device=vertices.device)
    generator.manual_seed(_sample_epoch_seed(str(sample["sample_id"]), base_seed, epoch))
    noise = torch.randn(
        vertices.shape,
        generator=generator,
        device=vertices.device,
        dtype=torch.float32,
    ) * settings.std_h
    noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    max_ratio = CLIP_SIGMA * settings.std_h
    noise = noise * (max_ratio / noise_norm.clamp_min(1e-12)).clamp_max(1.0)

    local_h = sample["local_edge_length"].to(
        device=vertices.device, dtype=torch.float32
    ).unsqueeze(-1)
    valid = torch.isfinite(local_h) & (local_h > 0)
    valid_scale_mask = sample.get("valid_scale_mask")
    if isinstance(valid_scale_mask, torch.Tensor):
        valid &= valid_scale_mask.to(
            device=vertices.device, dtype=torch.bool
        ).unsqueeze(-1)
    local_h = torch.where(valid, local_h, torch.zeros_like(local_h))
    typed_offsets = (noise * local_h).to(dtype=vertices.dtype)
    query_positions = vertices + typed_offsets
    allowed_ratio = max_ratio + 1e-7
    for _ in range(8):
        realised = torch.linalg.vector_norm(
            (query_positions - vertices).float(), dim=-1, keepdim=True
        ) / local_h.clamp_min(1e-12)
        rounded_outside = valid & (realised > allowed_ratio)
        typed_offsets = torch.where(rounded_outside, typed_offsets * 0.5, typed_offsets)
        query_positions = vertices + typed_offsets
    realised = torch.linalg.vector_norm(
        (query_positions - vertices).float(), dim=-1, keepdim=True
    ) / local_h.clamp_min(1e-12)
    rounded_outside = valid & (realised > allowed_ratio)
    query_positions = torch.where(rounded_outside, vertices, query_positions)
    actual_offsets = query_positions - vertices
    ratios = torch.linalg.vector_norm(actual_offsets.float(), dim=-1)[valid.squeeze(-1)]
    ratios = ratios / local_h.squeeze(-1)[valid.squeeze(-1)]
    result["query_positions"] = query_positions
    result["local_query_offsets"] = actual_offsets
    result["local_query_jitter_diagnostics"] = {
        "distribution": "isotropic_gaussian_xyz_clipped_at_3sigma",
        "std_h": settings.std_h,
        "clip_h": max_ratio,
        "mean_offset_norm_over_h": float(ratios.double().mean().item()),
        "p95_offset_norm_over_h": float(torch.quantile(ratios.double(), 0.95).item()),
        "max_offset_norm_over_h": float(ratios.double().max().item()),
        "valid_vertices": int(valid.sum().item()),
    }
    return result


def _sample_epoch_seed(sample_id: str, base_seed: int, epoch: int) -> int:
    digest = hashlib.sha256(("local-query-jitter:" + sample_id).encode("utf-8")).digest()
    sample_component = int.from_bytes(digest[:8], "little", signed=False)
    return int(
        (sample_component + int(base_seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    )

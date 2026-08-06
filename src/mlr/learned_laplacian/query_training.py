from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch


GT_QUERY_TRAINING_MODE = "gt_vertex_perturbation_v1"
QUERY_FOURIER_GEOMETRY_MODE = "query_fourier"


@dataclass(frozen=True)
class QueryAugmentationSettings:
    enabled: bool
    exact_fraction: float
    normal_std_h: float
    tangent_std_h: float
    max_offset_h: float
    apply_to_validation: bool
    zero_initial_laplacian: bool


def query_augmentation_settings(config: Mapping[str, Any]) -> QueryAugmentationSettings:
    raw = config.get("query_training", {})
    if not isinstance(raw, Mapping):
        raise ValueError("query_training must be an object.")
    settings = QueryAugmentationSettings(
        enabled=bool(raw.get("enabled", False)),
        exact_fraction=float(raw.get("exact_fraction", 0.2)),
        normal_std_h=float(raw.get("normal_std_h", 0.1)),
        tangent_std_h=float(raw.get("tangent_std_h", 0.1)),
        max_offset_h=float(raw.get("max_offset_h", 0.3)),
        apply_to_validation=bool(raw.get("apply_to_validation", True)),
        zero_initial_laplacian=bool(raw.get("zero_initial_laplacian", True)),
    )
    if not 0.0 < settings.exact_fraction < 1.0:
        raise ValueError("query_training.exact_fraction must lie strictly between 0 and 1.")
    if settings.normal_std_h < 0 or settings.tangent_std_h < 0:
        raise ValueError("query perturbation standard deviations must be non-negative.")
    if settings.max_offset_h <= 0:
        raise ValueError("query_training.max_offset_h must be positive.")
    if settings.enabled and settings.normal_std_h == 0 and settings.tangent_std_h == 0:
        raise ValueError("Enabled query perturbation requires a non-zero displacement scale.")
    return settings


def validate_gt_query_contract(sample: Mapping[str, Any]) -> None:
    """Reject coarse/expanded samples when GT-query training is requested."""

    metadata = sample.get("metadata", {})
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("query_training_mode") != GT_QUERY_TRAINING_MODE
    ):
        raise ValueError(
            "query_training.enabled=true requires samples prepared with "
            f"query_training_mode={GT_QUERY_TRAINING_MODE!r}; convert the manifest first."
        )
    vertices = sample["vertices"]
    target_positions = sample.get("target_positions")
    gt_vertices = sample.get("gt_vertices")
    gt_faces = sample.get("gt_faces")
    if not isinstance(target_positions, torch.Tensor) or not torch.equal(
        vertices, target_positions
    ):
        raise ValueError("GT-query samples require vertices == target_positions.")
    if not isinstance(gt_vertices, torch.Tensor) or not torch.equal(vertices, gt_vertices):
        raise ValueError("GT-query samples require vertices == gt_vertices.")
    if not isinstance(gt_faces, torch.Tensor) or not torch.equal(sample["faces"], gt_faces):
        raise ValueError("GT-query samples require faces == gt_faces.")
    if torch.count_nonzero(sample["initial_laplacian"]).item() != 0:
        raise ValueError("GT-query samples must store a zero initial_laplacian to prevent leakage.")


def apply_query_augmentation(
    sample: Mapping[str, Any],
    settings: QueryAugmentationSettings,
    *,
    base_seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Create mixed exact/perturbed queries without modifying graph or targets."""

    result = dict(sample)
    vertices = sample["vertices"]
    num_vertices = int(vertices.shape[0])
    if num_vertices < 2:
        raise ValueError("Mixed exact/perturbed queries require at least two vertices.")
    if not settings.enabled:
        result["query_positions"] = vertices
        result["query_is_exact"] = torch.ones(
            num_vertices, dtype=torch.bool, device=vertices.device
        )
        if settings.zero_initial_laplacian:
            result["initial_laplacian"] = torch.zeros_like(sample["initial_laplacian"])
        return result

    generator = torch.Generator(device=vertices.device)
    generator.manual_seed(_sample_epoch_seed(str(sample["sample_id"]), base_seed, epoch))
    normals = torch.nn.functional.normalize(
        sample["vertex_normals"].float(), dim=-1, eps=1e-8
    )
    tangent = torch.randn(
        (num_vertices, 3),
        generator=generator,
        device=vertices.device,
        dtype=torch.float32,
    )
    tangent = tangent - (tangent * normals).sum(dim=-1, keepdim=True) * normals
    tangent = torch.nn.functional.normalize(tangent, dim=-1, eps=1e-8)
    normal_amount = torch.randn(
        (num_vertices, 1),
        generator=generator,
        device=vertices.device,
        dtype=torch.float32,
    ) * settings.normal_std_h
    tangent_amount = torch.randn(
        (num_vertices, 1),
        generator=generator,
        device=vertices.device,
        dtype=torch.float32,
    ) * settings.tangent_std_h
    local_h = sample["local_edge_length"].float().unsqueeze(-1)
    offsets = (normal_amount * normals + tangent_amount * tangent) * local_h
    maximum = settings.max_offset_h * local_h
    offset_norm = torch.linalg.vector_norm(offsets, dim=-1, keepdim=True)
    offsets = offsets * (maximum / offset_norm.clamp_min(1e-12)).clamp_max(1.0)

    exact_count = min(
        max(int(round(num_vertices * settings.exact_fraction)), 1), num_vertices - 1
    )
    exact_indices = torch.randperm(
        num_vertices, generator=generator, device=vertices.device
    )[:exact_count]
    exact_mask = torch.zeros(num_vertices, dtype=torch.bool, device=vertices.device)
    exact_mask[exact_indices] = True
    offsets[exact_mask] = 0.0

    result["query_positions"] = vertices + offsets.to(dtype=vertices.dtype)
    result["query_offsets"] = offsets.to(dtype=vertices.dtype)
    result["query_is_exact"] = exact_mask
    if settings.zero_initial_laplacian:
        result["initial_laplacian"] = torch.zeros_like(sample["initial_laplacian"])
    return result


def _sample_epoch_seed(sample_id: str, base_seed: int, epoch: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    sample_component = int.from_bytes(digest[:8], "little", signed=False)
    return int(
        (sample_component + int(base_seed) + 1_000_003 * int(epoch)) % (2**63 - 1)
    )

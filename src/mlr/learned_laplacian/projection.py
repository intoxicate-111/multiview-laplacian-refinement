from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ProjectionResult:
    """Projected vertices under the documented CV camera convention.

    Extrinsics are right-handed world-to-camera 4x4 transforms. Camera +X is
    image-right, +Y is image-down, and +Z is forward. Pixel coordinates use a
    top-left origin. ``grid`` follows grid_sample with align_corners=True, so
    pixel (0, 0) maps to (-1, -1) and (W-1, H-1) maps to (1, 1).
    """

    pixels: torch.Tensor
    grid: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor


def project_vertices(
    vertices: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: tuple[int, int],
    visibility: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> ProjectionResult:
    """Project world-space vertices into each view and reject invalid samples."""

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3].")
    views = intrinsics.shape[0]
    if tuple(intrinsics.shape) != (views, 3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3].")
    if tuple(extrinsics.shape) != (views, 4, 4):
        raise ValueError("extrinsics must have shape [V, 4, 4].")
    height, width = image_size
    if height < 1 or width < 1:
        raise ValueError("image_size must contain positive height and width.")

    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    homogeneous = torch.cat((vertices, ones), dim=1)
    camera_h = torch.einsum("vij,nj->vni", extrinsics, homogeneous)
    camera = camera_h[..., :3]
    depth = camera[..., 2]
    safe_depth = torch.where(depth.abs() > eps, depth, torch.ones_like(depth))
    pixels_h = torch.einsum("vij,vnj->vni", intrinsics, camera)
    pixels = pixels_h[..., :2] / safe_depth.unsqueeze(-1)

    u = pixels[..., 0]
    v = pixels[..., 1]
    x_grid = torch.zeros_like(u) if width == 1 else 2.0 * u / float(width - 1) - 1.0
    y_grid = torch.zeros_like(v) if height == 1 else 2.0 * v / float(height - 1) - 1.0
    grid = torch.stack((x_grid, y_grid), dim=-1)
    valid = (depth > eps) & (u >= 0.0) & (u <= width - 1) & (v >= 0.0) & (v <= height - 1)
    if visibility is not None:
        if tuple(visibility.shape) != tuple(valid.shape):
            raise ValueError(
                f"visibility must have shape {tuple(valid.shape)}, got {tuple(visibility.shape)}."
            )
        valid = valid & visibility.to(dtype=torch.bool)
    return ProjectionResult(pixels=pixels, grid=grid, depth=depth, valid=valid)


def sample_vertex_features(
    feature_maps: torch.Tensor,
    vertices: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: tuple[int, int],
    visibility: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ProjectionResult]:
    """Return differentiably sampled features [V, N, C] and their valid mask."""

    if feature_maps.ndim != 4:
        raise ValueError("feature_maps must have shape [V, C, Hf, Wf].")
    projection = project_vertices(
        vertices,
        intrinsics,
        extrinsics,
        image_size=image_size,
        visibility=visibility,
    )
    if feature_maps.shape[0] != projection.grid.shape[0]:
        raise ValueError("feature_maps and camera tensors must have the same number of views.")
    sampled = F.grid_sample(
        feature_maps,
        projection.grid.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2)
    sampled = sampled * projection.valid.unsqueeze(-1).to(sampled.dtype)
    return sampled, projection.valid, projection

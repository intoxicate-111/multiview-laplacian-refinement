from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import Array, Camera, Mesh, VisibilityCache, normalize_rows
from .laplacian import compute_laplacian_target
from .refinement import RefinementConfig, refine_mesh_with_laplacian


class PseudoSurfaceEstimator(Protocol):
    def estimate(
        self,
        mesh: Mesh,
        images: list[Array] | None,
        cameras: list[Camera],
        masks: list[Array] | None,
        visibility: VisibilityCache | None,
    ) -> tuple[Array, Array]:
        ...


@dataclass
class IdentityPseudoSurfaceEstimator:
    min_confidence: float = 1.0

    def estimate(
        self,
        mesh: Mesh,
        images: list[Array] | None,
        cameras: list[Camera],
        masks: list[Array] | None,
        visibility: VisibilityCache | None,
    ) -> tuple[Array, Array]:
        confidence = _visibility_confidence(mesh, visibility)
        confidence = np.maximum(confidence, self.min_confidence)
        return np.array(mesh.vertices, copy=True), confidence


@dataclass
class SilhouetteNormalPseudoSurfaceEstimator:
    """Rule-based pseudo target from projected mask occupancy.

    This is intentionally simple: vertices projected outside visible silhouettes are nudged
    inward along a normal/view direction proxy, and confidence comes from view agreement.
    It is a placeholder baseline for later photometric and learned estimators.
    """

    step_size: float = 0.01
    outside_threshold: float = 0.5

    def estimate(
        self,
        mesh: Mesh,
        images: list[Array] | None,
        cameras: list[Camera],
        masks: list[Array] | None,
        visibility: VisibilityCache | None,
    ) -> tuple[Array, Array]:
        if masks is None or not cameras:
            return IdentityPseudoSurfaceEstimator().estimate(mesh, images, cameras, masks, visibility)

        mesh.ensure_normals()
        votes = np.zeros(mesh.num_vertices, dtype=np.float64)
        total = np.zeros(mesh.num_vertices, dtype=np.float64)
        direction = np.zeros_like(mesh.vertices)

        for camera, mask in zip(cameras, masks, strict=True):
            pixels, depth = camera.project(mesh.vertices)
            inside_image = _inside_mask_extent(mask, pixels) & (depth > 1e-8)
            occupied = np.zeros(mesh.num_vertices, dtype=bool)
            occupied[inside_image] = _sample_mask(mask, pixels[inside_image])
            total += inside_image.astype(np.float64)
            votes += occupied.astype(np.float64)
            to_camera = normalize_rows(camera.center[None, :] - mesh.vertices)
            direction += to_camera * ((inside_image & ~occupied).astype(np.float64)[:, None])

        occupancy_ratio = np.divide(votes, np.maximum(total, 1.0))
        outside = occupancy_ratio < self.outside_threshold
        direction = normalize_rows(direction + mesh.normals)
        p_star = np.array(mesh.vertices, copy=True)
        p_star[outside] = p_star[outside] + self.step_size * direction[outside]
        confidence = np.clip(total / max(1, len(cameras)), 0.0, 1.0)
        confidence[outside] *= 0.5
        return p_star, confidence


def estimate_pseudo_surface(
    mesh: Mesh,
    images: list[Array] | None,
    cameras: list[Camera],
    masks: list[Array] | None,
    visibility: VisibilityCache | None,
    estimator: PseudoSurfaceEstimator | None = None,
) -> tuple[Array, Array]:
    estimator = estimator or IdentityPseudoSurfaceEstimator()
    return estimator.estimate(mesh, images, cameras, masks, visibility)


def refine_from_pseudo_surface(
    mesh: Mesh,
    p_star: Array,
    confidence: Array,
    operator_type: str = "uniform",
    anchors: Array | None = None,
    config: RefinementConfig | None = None,
):
    delta_pseudo = compute_laplacian_target(p_star, mesh.faces, operator_type)
    return refine_mesh_with_laplacian(mesh, delta_pseudo, confidence, anchors, config)


def _visibility_confidence(mesh: Mesh, visibility: VisibilityCache | None) -> Array:
    if visibility is None:
        return np.ones(mesh.num_vertices, dtype=np.float64)
    confidence = visibility.vertex_confidence
    max_conf = confidence.max(initial=0.0)
    if max_conf > 0:
        confidence = confidence / max_conf
    return confidence


def _inside_mask_extent(mask: Array, pixels: Array) -> Array:
    height, width = mask.shape[:2]
    u = pixels[:, 0]
    v = pixels[:, 1]
    return (u >= 0) & (v >= 0) & (u < width) & (v < height)


def _sample_mask(mask: Array, pixels: Array) -> Array:
    height, width = mask.shape[:2]
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    return mask[v, u] > 0

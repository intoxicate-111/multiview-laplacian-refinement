from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Camera:
    """Pinhole camera with world-to-camera extrinsics."""

    intrinsics: Array
    rotation: Array
    translation: Array
    image_size: tuple[int, int] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intrinsics", np.asarray(self.intrinsics, dtype=np.float64))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=np.float64))
        object.__setattr__(self, "translation", np.asarray(self.translation, dtype=np.float64).reshape(3))
        if self.intrinsics.shape != (3, 3):
            raise ValueError("Camera intrinsics must have shape (3, 3).")
        if self.rotation.shape != (3, 3):
            raise ValueError("Camera rotation must have shape (3, 3).")

    @property
    def center(self) -> Array:
        return -self.rotation.T @ self.translation

    def world_to_camera(self, points: Array) -> Array:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.rotation.T + self.translation[None, :]

    def project(self, points: Array) -> tuple[Array, Array]:
        cam = self.world_to_camera(points)
        z = cam[:, 2]
        safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        pixels_h = cam @ self.intrinsics.T
        pixels = pixels_h[:, :2] / safe_z[:, None]
        return pixels, z


@dataclass
class VisibilityCache:
    vertex_view_visible: Array
    vertex_view_weight: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertex_view_visible = np.asarray(self.vertex_view_visible, dtype=bool)
        if self.vertex_view_weight is not None:
            self.vertex_view_weight = np.asarray(self.vertex_view_weight, dtype=np.float64)
            if self.vertex_view_weight.shape != self.vertex_view_visible.shape:
                raise ValueError("Visibility weights must match visibility shape.")

    @property
    def vertex_confidence(self) -> Array:
        if self.vertex_view_weight is not None:
            weights = self.vertex_view_weight * self.vertex_view_visible
            return np.clip(weights.sum(axis=1), 0.0, None)
        return self.vertex_view_visible.sum(axis=1).astype(np.float64)


@dataclass
class Mesh:
    vertices: Array
    faces: Array
    normals: Array | None = None
    visibility: VisibilityCache | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Mesh vertices must have shape (N, 3).")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Mesh faces must have shape (M, 3).")
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
            if self.normals.shape != self.vertices.shape:
                raise ValueError("Mesh normals must have shape (N, 3).")

    def with_vertices(self, vertices: Array, *, recompute_normals: bool = True) -> "Mesh":
        normals = None if recompute_normals else self.normals
        new_mesh = replace(self, vertices=np.asarray(vertices, dtype=np.float64), normals=normals)
        if recompute_normals:
            new_mesh.normals = new_mesh.compute_vertex_normals()
        return new_mesh

    def compute_vertex_normals(self) -> Array:
        normals = np.zeros_like(self.vertices)
        tris = self.vertices[self.faces]
        face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(lengths, 1e-12)
        for corner in range(3):
            np.add.at(normals, self.faces[:, corner], face_normals)
        return normalize_rows(normals)

    def ensure_normals(self) -> "Mesh":
        if self.normals is None:
            self.normals = self.compute_vertex_normals()
        return self

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class ReconstructionInput:
    image_paths: list[Path]
    cameras: list[Camera]
    mask_paths: list[Path] | None = None
    gt_mesh_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.image_paths) != len(self.cameras):
            raise ValueError("Number of images and cameras must match.")
        if self.mask_paths is not None and len(self.mask_paths) != len(self.image_paths):
            raise ValueError("Number of masks and images must match.")


def normalize_rows(values: Array, eps: float = 1e-12) -> Array:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)

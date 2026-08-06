from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from mlr.data import Camera, Mesh
from mlr.synthetic import (
    SyntheticRenderConfig,
    _is_front_facing_projected,
    render_mesh_face_ids,
)


VISIBILITY_CONDITIONS = (
    "frustum_only",
    "backface_only",
    "occlusion_only",
    "backface_and_occlusion",
)


@dataclass(frozen=True)
class RendererVisibilityResult:
    """Per-view masks produced from renderer-native triangle visibility."""

    frustum_valid: np.ndarray
    backface_visible: np.ndarray
    occlusion_visible: np.ndarray
    backface_and_occlusion_visible: np.ndarray
    neighborhood_radius: int
    backend: str
    front_face_winding: str
    front_face_counts: np.ndarray
    back_face_counts: np.ndarray
    two_sided_pixel_counts: np.ndarray
    culled_pixel_counts: np.ndarray

    def condition(self, name: str) -> np.ndarray:
        if name == "frustum_only":
            return np.ones_like(self.frustum_valid)
        if name == "backface_only":
            return self.backface_visible
        if name == "occlusion_only":
            return self.occlusion_visible
        if name == "backface_and_occlusion":
            return self.backface_and_occlusion_visible
        raise ValueError(f"Unknown renderer visibility condition: {name!r}.")


def compute_renderer_visibility(
    mesh: Mesh,
    cameras: Sequence[Camera],
    config: SyntheticRenderConfig,
    *,
    neighborhood_radius: int = 1,
) -> RendererVisibilityResult:
    """Compute four controlled visibility conditions with the RGB renderer.

    Occlusion masks come from depth-tested face-ID buffers. Back-face-only uses
    the exact projected winding convention configured on that renderer, without
    applying inter-surface depth rejection. The combined condition uses a
    depth-tested face-ID pass with renderer face culling enabled.
    """

    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative.")
    if config.backend not in {"cpu", "opengl"}:
        raise ValueError("Renderer visibility supports cpu or opengl backends.")
    num_views = len(cameras)
    num_vertices = mesh.num_vertices
    frustum = np.zeros((num_views, num_vertices), dtype=bool)
    backface = np.zeros_like(frustum)
    occlusion = np.zeros_like(frustum)
    combined = np.zeros_like(frustum)
    front_counts = np.zeros(num_views, dtype=np.int64)
    back_counts = np.zeros(num_views, dtype=np.int64)
    two_sided_pixels = np.zeros(num_views, dtype=np.int64)
    culled_pixels = np.zeros(num_views, dtype=np.int64)
    two_sided = replace(config, backface_culling=False)
    culled = replace(config, backface_culling=True)

    for view_index, camera in enumerate(cameras):
        frustum[view_index] = frustum_valid(mesh.vertices, camera)
        backface[view_index], front_counts[view_index], back_counts[view_index] = (
            projected_backface_visibility_with_counts(
                mesh, camera, config.front_face_winding
            )
        )
        two_sided_ids = render_mesh_face_ids(mesh, camera, two_sided)
        culled_ids = render_mesh_face_ids(mesh, camera, culled)
        two_sided_pixels[view_index] = int(np.count_nonzero(two_sided_ids >= 0))
        culled_pixels[view_index] = int(np.count_nonzero(culled_ids >= 0))
        occlusion[view_index] = vertex_visibility_from_face_id_buffer(
            mesh.vertices,
            mesh.faces,
            camera,
            two_sided_ids,
            neighborhood_radius=neighborhood_radius,
        )
        combined[view_index] = vertex_visibility_from_face_id_buffer(
            mesh.vertices,
            mesh.faces,
            camera,
            culled_ids,
            neighborhood_radius=neighborhood_radius,
        )

    return RendererVisibilityResult(
        frustum_valid=frustum,
        backface_visible=backface,
        occlusion_visible=occlusion,
        backface_and_occlusion_visible=combined,
        neighborhood_radius=int(neighborhood_radius),
        backend=config.backend,
        front_face_winding=config.front_face_winding,
        front_face_counts=front_counts,
        back_face_counts=back_counts,
        two_sided_pixel_counts=two_sided_pixels,
        culled_pixel_counts=culled_pixels,
    )


def frustum_valid(vertices: np.ndarray, camera: Camera, eps: float = 1e-8) -> np.ndarray:
    pixels, depth = camera.project(vertices)
    if camera.image_size is None:
        raise ValueError("Camera image_size is required for renderer visibility.")
    width, height = camera.image_size
    return (
        (depth > eps)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )


def projected_backface_visibility(
    mesh: Mesh,
    camera: Camera,
    front_face_winding: str,
) -> np.ndarray:
    """Return vertices incident to at least one renderer-front-facing face.

    This isolates face culling from depth-test occlusion. It uses the same CV
    projection and the same GL front-face winding conversion as the renderer.
    """

    return projected_backface_visibility_with_counts(
        mesh, camera, front_face_winding
    )[0]


def projected_backface_visibility_with_counts(
    mesh: Mesh,
    camera: Camera,
    front_face_winding: str,
) -> tuple[np.ndarray, int, int]:
    pixels, depth = camera.project(mesh.vertices)
    face_depth = depth[mesh.faces]
    positive = np.all(face_depth > 1e-8, axis=1)
    front = np.zeros(mesh.num_faces, dtype=bool)
    for face_index in np.flatnonzero(positive):
        front[face_index] = _is_front_facing_projected(
            pixels[mesh.faces[face_index]], front_face_winding
        )
    visible = np.zeros(mesh.num_vertices, dtype=bool)
    if np.any(front):
        visible[np.unique(mesh.faces[front])] = True
    valid_face_count = int(positive.sum())
    return visible, int(front.sum()), int(valid_face_count - front.sum())


def vertex_visibility_from_face_id_buffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
    face_id_buffer: np.ndarray,
    *,
    neighborhood_radius: int = 1,
) -> np.ndarray:
    """Check incident-face IDs near each projected vertex pixel.

    A visible pixel elsewhere on an incident triangle is insufficient: only IDs
    in the configurable neighborhood around that vertex's own projection count.
    This avoids marking an occluded vertex visible merely because another part
    of one of its incident faces remains visible.
    """

    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative.")
    if face_id_buffer.ndim != 2:
        raise ValueError("face_id_buffer must have shape [H, W].")
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    pixels, depth = camera.project(vertices)
    height, width = face_id_buffer.shape
    # Raster pixels cover [i, i+1) with their sample at i+0.5. ``floor`` is
    # therefore the correct exact-pixel anchor; rounding would shift vertices
    # at half-pixel triangle edges by one full pixel before applying the 3x3
    # neighborhood.
    safe_pixels = np.where(np.isfinite(pixels), pixels, -1.0)
    center_x = np.floor(safe_pixels[:, 0]).astype(np.int64)
    center_y = np.floor(safe_pixels[:, 1]).astype(np.int64)
    vertex_ids = np.arange(vertices.shape[0], dtype=np.int64)
    visible = np.zeros(vertices.shape[0], dtype=bool)

    for offset_y in range(-neighborhood_radius, neighborhood_radius + 1):
        for offset_x in range(-neighborhood_radius, neighborhood_radius + 1):
            x = center_x + offset_x
            y = center_y + offset_y
            valid = (
                (depth > 1e-8)
                & (x >= 0)
                & (x < width)
                & (y >= 0)
                & (y < height)
                & ~visible
            )
            indices = np.flatnonzero(valid)
            if indices.size == 0:
                continue
            candidate = face_id_buffer[y[indices], x[indices]].astype(np.int64)
            foreground = (candidate >= 0) & (candidate < faces.shape[0])
            if not np.any(foreground):
                continue
            checked_vertices = indices[foreground]
            checked_faces = faces[candidate[foreground]]
            visible[checked_vertices] = np.any(
                checked_faces == vertex_ids[checked_vertices, None], axis=1
            )
    return visible


def union_of_visible_faces(face_id_buffer: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Coarse diagnostic only: vertices of any face visible anywhere in a view."""

    visible_faces = np.unique(face_id_buffer[face_id_buffer >= 0]).astype(np.int64)
    result = np.zeros(int(np.max(faces)) + 1 if np.asarray(faces).size else 0, dtype=bool)
    if visible_faces.size:
        result[np.unique(np.asarray(faces, dtype=np.int64)[visible_faces])] = True
    return result


def visibility_statistics(result: RendererVisibilityResult) -> Mapping[str, float]:
    frustum = result.frustum_valid
    backface_final = frustum & result.backface_visible
    occlusion_final = frustum & result.occlusion_visible
    combined_final = frustum & result.backface_and_occlusion_visible
    denominator = max(int(frustum.sum()), 1)
    visible_counts = combined_final.sum(axis=0)
    pixel_denominator = max(int(result.two_sided_pixel_counts.sum()), 1)
    return {
        "mean_visible_views_per_vertex": float(visible_counts.mean()),
        "median_visible_views_per_vertex": float(np.median(visible_counts)),
        "zero_visible_vertex_ratio": float(np.mean(visible_counts == 0)),
        "frustum_valid_ratio": float(frustum.mean()),
        "backface_rejected_ratio_of_frustum": float(
            np.sum(frustum & ~backface_final) / denominator
        ),
        "occlusion_rejected_ratio_of_frustum": float(
            np.sum(frustum & ~occlusion_final) / denominator
        ),
        "final_visible_ratio": float(combined_final.mean()),
        "front_facing_faces_across_views": int(result.front_face_counts.sum()),
        "back_facing_faces_across_views": int(result.back_face_counts.sum()),
        "backface_culled_pixel_ratio": float(
            1.0 - float(result.culled_pixel_counts.sum()) / pixel_denominator
        ),
    }


def mesh_topology_orientation_diagnostics(faces: np.ndarray) -> Mapping[str, int | float]:
    """Report winding consistency, boundary and non-manifold edge counts."""

    faces = np.asarray(faces, dtype=np.int64)
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(start, end)), int(max(start, end)))
            direction = 1 if int(start) == key[0] else -1
            edge_uses.setdefault(key, []).append((face_index, direction))
    boundary = sum(len(uses) == 1 for uses in edge_uses.values())
    non_manifold = sum(len(uses) > 2 for uses in edge_uses.values())
    manifold_shared = [uses for uses in edge_uses.values() if len(uses) == 2]
    inconsistent = sum(uses[0][1] == uses[1][1] for uses in manifold_shared)

    parent = np.arange(len(faces), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for uses in edge_uses.values():
        for other in uses[1:]:
            union(uses[0][0], other[0])
    components = len({find(index) for index in range(len(faces))}) if len(faces) else 0
    return {
        "connected_face_components": int(components),
        "boundary_edges": int(boundary),
        "non_manifold_edges": int(non_manifold),
        "manifold_shared_edges": int(len(manifold_shared)),
        "inconsistent_winding_edges": int(inconsistent),
        "inconsistent_winding_edge_ratio": float(
            inconsistent / max(len(manifold_shared), 1)
        ),
    }

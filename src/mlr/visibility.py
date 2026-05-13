from __future__ import annotations

import numpy as np

from .data import Array, Camera, Mesh, VisibilityCache, normalize_rows


def update_visibility(
    mesh: Mesh,
    cameras: list[Camera],
    masks: list[Array] | None = None,
    use_normal_facing: bool = True,
    min_view_dot: float = 0.0,
) -> VisibilityCache:
    mesh.ensure_normals()
    visible = np.zeros((mesh.num_vertices, len(cameras)), dtype=bool)
    weights = np.zeros_like(visible, dtype=np.float64)
    for view_idx, camera in enumerate(cameras):
        pixels, depth = camera.project(mesh.vertices)
        in_front = depth > 1e-8
        inside = _inside_image(pixels, camera.image_size, masks[view_idx] if masks is not None else None)
        mask_ok = np.ones(mesh.num_vertices, dtype=bool)
        if masks is not None:
            mask_ok = _sample_binary_mask(masks[view_idx], pixels)

        normal_ok = np.ones(mesh.num_vertices, dtype=bool)
        angle_weight = np.ones(mesh.num_vertices, dtype=np.float64)
        if use_normal_facing and mesh.normals is not None:
            to_camera = normalize_rows(camera.center[None, :] - mesh.vertices)
            dots = np.sum(mesh.normals * to_camera, axis=1)
            normal_ok = dots > min_view_dot
            angle_weight = np.clip(dots, 0.0, 1.0)

        view_visible = in_front & inside & mask_ok & normal_ok
        visible[:, view_idx] = view_visible
        weights[:, view_idx] = view_visible.astype(np.float64) * angle_weight
    return VisibilityCache(visible, weights, metadata={"mode": "projection_normal_mask"})


def _inside_image(pixels: Array, image_size: tuple[int, int] | None, mask: Array | None) -> Array:
    if image_size is None and mask is not None:
        height, width = mask.shape[:2]
    elif image_size is None:
        return np.ones(len(pixels), dtype=bool)
    else:
        width, height = image_size
    u = pixels[:, 0]
    v = pixels[:, 1]
    return (u >= 0) & (v >= 0) & (u < width) & (v < height)


def _sample_binary_mask(mask: Array, pixels: Array) -> Array:
    height, width = mask.shape[:2]
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    ok = (u >= 0) & (v >= 0) & (u < width) & (v < height)
    values = np.zeros(len(pixels), dtype=bool)
    values[ok] = mask[v[ok], u[ok]] > 0
    return values

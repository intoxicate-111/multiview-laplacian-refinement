from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mlr.data import Camera, Mesh
from mlr.datasets import load_masks, load_reconstruction_input
from mlr.gt_laplacian import GTLaplacianTargetConfig, compute_coarse_graph_gt_laplacian_target
from mlr.io import load_mesh
from mlr.laplacian import compute_laplacian_coordinates

from .dataset import save_prepared_sample


def prepare_single_object_sample(
    dataset_path: str | Path,
    coarse_mesh_path: str | Path,
    gt_mesh_path: str | Path | None = None,
    output_path: str | Path | None = None,
    image_size: int | None = None,
    operator_type: str = "uniform",
    distance_confidence_scale: float | None = None,
    coarse_noise_std: float = 0.0,
    seed: int = 7,
) -> dict:
    """Prepare one validated sample while reusing the repository target constructor."""

    reconstruction = load_reconstruction_input(dataset_path)
    if gt_mesh_path is None:
        gt_mesh_path = reconstruction.gt_mesh_path
    if gt_mesh_path is None:
        raise ValueError("gt_mesh_path is required when dataset.json has no mesh_path.")
    coarse_mesh = load_mesh(coarse_mesh_path).ensure_normals()
    gt_mesh = load_mesh(gt_mesh_path).ensure_normals()
    if coarse_noise_std < 0:
        raise ValueError("coarse_noise_std must be non-negative.")
    if coarse_noise_std > 0:
        rng = np.random.default_rng(seed)
        offsets = rng.normal(0.0, coarse_noise_std, size=(coarse_mesh.num_vertices, 1))
        noisy_vertices = coarse_mesh.vertices + offsets * coarse_mesh.normals
        coarse_mesh = Mesh(noisy_vertices, coarse_mesh.faces.copy()).ensure_normals()

    images, scale_xy = _load_images(reconstruction.image_paths, image_size)
    intrinsics = []
    extrinsics = []
    for camera in reconstruction.cameras:
        scaled = camera.intrinsics.copy()
        scaled[0, :] *= scale_xy[0]
        scaled[1, :] *= scale_xy[1]
        intrinsics.append(scaled)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = camera.rotation
        extrinsic[:3, 3] = camera.translation
        extrinsics.append(extrinsic)

    masks = load_masks(reconstruction.mask_paths)
    visibility = None
    if masks is not None:
        resized_masks = [_resize_mask(mask, images.shape[-2:]) for mask in masks]
        visibility = _mask_visibility(coarse_mesh.vertices, reconstruction.cameras, resized_masks, scale_xy)

    target = compute_coarse_graph_gt_laplacian_target(
        coarse_mesh,
        gt_mesh,
        GTLaplacianTargetConfig(
            operator_type=operator_type,
            distance_confidence_scale=distance_confidence_scale,
        ),
    )
    initial_laplacian = compute_laplacian_coordinates(
        coarse_mesh.vertices,
        coarse_mesh.faces,
        operator_type,
    )
    sample = {
        "sample_id": Path(dataset_path).resolve().parent.name,
        "images": images,
        "intrinsics": torch.as_tensor(np.stack(intrinsics), dtype=torch.float32),
        "extrinsics": torch.as_tensor(np.stack(extrinsics), dtype=torch.float32),
        "vertices": torch.as_tensor(coarse_mesh.vertices, dtype=torch.float32),
        "faces": torch.as_tensor(coarse_mesh.faces, dtype=torch.long),
        "vertex_normals": torch.as_tensor(coarse_mesh.normals, dtype=torch.float32),
        "initial_laplacian": torch.as_tensor(initial_laplacian, dtype=torch.float32),
        "laplacian_target": torch.as_tensor(target.delta_target, dtype=torch.float32),
        "target_confidence": torch.as_tensor(target.confidence, dtype=torch.float32),
        "visibility": None if visibility is None else torch.as_tensor(visibility, dtype=torch.bool),
        "target_positions": torch.as_tensor(target.closest_points, dtype=torch.float32),
        "gt_vertices": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_faces": torch.as_tensor(gt_mesh.faces, dtype=torch.long),
        "metadata": {
            "dataset_path": str(Path(dataset_path)),
            "coarse_mesh_path": str(Path(coarse_mesh_path)),
            "gt_mesh_path": str(Path(gt_mesh_path)),
            "operator_type": operator_type,
            "camera_convention": "right-handed CV world-to-camera, +Z forward, +X right, +Y down",
            "coarse_noise_std": float(coarse_noise_std),
            "seed": int(seed),
        },
    }
    if output_path is not None:
        save_prepared_sample(sample, output_path)
    return sample


def _load_images(paths: list[Path], image_size: int | None) -> tuple[torch.Tensor, tuple[float, float]]:
    arrays = []
    original_size = None
    target_size = None
    for path in paths:
        image = Image.open(path).convert("RGB")
        if original_size is None:
            original_size = image.size
        elif image.size != original_size:
            raise ValueError("All sample images must have the same dimensions.")
        if image_size is not None:
            if image_size < 1:
                raise ValueError("image_size must be positive.")
            target_size = (image_size, image_size)
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        else:
            target_size = image.size
        array = np.asarray(image, dtype=np.float32) / 255.0
        arrays.append(array.transpose(2, 0, 1))
    if not arrays or original_size is None or target_size is None:
        raise ValueError("Dataset must contain at least one image.")
    scale_xy = (target_size[0] / original_size[0], target_size[1] / original_size[1])
    return torch.from_numpy(np.stack(arrays)), scale_xy


def _resize_mask(mask: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0


def _mask_visibility(
    vertices: np.ndarray,
    cameras: list[Camera],
    masks: list[np.ndarray],
    scale_xy: tuple[float, float],
) -> np.ndarray:
    result = np.zeros((len(cameras), len(vertices)), dtype=bool)
    for view, (camera, mask) in enumerate(zip(cameras, masks)):
        pixels, depth = camera.project(vertices)
        pixels = pixels * np.asarray(scale_xy)[None, :]
        x = np.rint(pixels[:, 0]).astype(np.int64)
        y = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (
            (depth > 1e-8)
            & (x >= 0)
            & (x < mask.shape[1])
            & (y >= 0)
            & (y < mask.shape[0])
        )
        indices = np.flatnonzero(valid)
        result[view, indices] = mask[y[indices], x[indices]]
    return result

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mlr.data import Camera, Mesh
from mlr.datasets import load_masks, load_reconstruction_input
from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
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


def corrupt_same_topology_mesh(
    gt_mesh: Mesh,
    noise_std: float = 0.015,
    smoothing_iters: int = 2,
    smoothing_strength: float = 0.1,
    seed: int = 7,
) -> Mesh:
    """Create a deterministic recognisable corruption without changing topology."""

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if smoothing_iters < 0:
        raise ValueError("smoothing_iters must be non-negative.")
    if not 0.0 <= smoothing_strength <= 1.0:
        raise ValueError("smoothing_strength must lie in [0, 1].")
    gt_mesh = Mesh(gt_mesh.vertices.copy(), gt_mesh.faces.copy()).ensure_normals()
    vertices = gt_mesh.vertices.copy()
    laplacian_data = build_uniform_laplacian_data(gt_mesh.faces, gt_mesh.num_vertices)
    for _ in range(smoothing_iters):
        laplacian = apply_uniform_laplacian(vertices, laplacian_data)
        vertices -= float(smoothing_strength) * laplacian
    smoothed = Mesh(vertices, gt_mesh.faces.copy()).ensure_normals()
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        normal_offsets = rng.normal(0.0, noise_std, size=(gt_mesh.num_vertices, 1))
        vertices = vertices + normal_offsets * smoothed.normals
    return Mesh(vertices, gt_mesh.faces.copy()).ensure_normals()


def prepare_same_topology_sample(
    dataset_path: str | Path,
    coarse_mesh_path: str | Path,
    gt_mesh_path: str | Path,
    output_path: str | Path | None = None,
    image_size: int | None = None,
    seed: int = 7,
    extra_metadata: dict | None = None,
) -> dict:
    """Prepare a scalable uniform-Laplacian sample with exact GT correspondences.

    This path is intended for controlled experiments where coarse and GT meshes
    have identical vertex/face topology. It avoids dense N-by-N matrices and
    defines ``delta_target = L_prediction_graph @ GT_vertices`` using the
    repository's sparse coarse-oracle implementation.
    """

    reconstruction = load_reconstruction_input(dataset_path)
    coarse_mesh = load_mesh(coarse_mesh_path).ensure_normals()
    gt_mesh = load_mesh(gt_mesh_path).ensure_normals()
    if coarse_mesh.num_vertices != gt_mesh.num_vertices:
        raise ValueError("Same-topology preparation requires equal coarse and GT vertex counts.")
    if coarse_mesh.faces.shape != gt_mesh.faces.shape or not np.array_equal(
        coarse_mesh.faces, gt_mesh.faces
    ):
        raise ValueError("Same-topology preparation requires identical coarse and GT faces.")

    images, scale_xy = _load_images(reconstruction.image_paths, image_size)
    intrinsics, extrinsics = _camera_tensors(reconstruction.cameras, scale_xy)
    masks = load_masks(reconstruction.mask_paths)
    visibility = None
    if masks is not None:
        resized_masks = [_resize_mask(mask, images.shape[-2:]) for mask in masks]
        visibility = _mask_visibility(
            coarse_mesh.vertices, reconstruction.cameras, resized_masks, scale_xy
        )

    laplacian_data = build_uniform_laplacian_data(coarse_mesh.faces, coarse_mesh.num_vertices)
    initial_laplacian = apply_uniform_laplacian(coarse_mesh.vertices, laplacian_data)
    target_laplacian = apply_uniform_laplacian(gt_mesh.vertices, laplacian_data)
    metadata = {
        "dataset_path": str(Path(dataset_path)),
        "coarse_mesh_path": str(Path(coarse_mesh_path)),
        "gt_mesh_path": str(Path(gt_mesh_path)),
        "operator_type": "uniform",
        "target_constructor": "same_topology_correspondence_sparse_uniform",
        "camera_convention": "right-handed CV world-to-camera, +Z forward, +X right, +Y down",
        "seed": int(seed),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    sample = {
        "sample_id": Path(gt_mesh_path).stem,
        "images": images,
        "intrinsics": torch.as_tensor(intrinsics, dtype=torch.float32),
        "extrinsics": torch.as_tensor(extrinsics, dtype=torch.float32),
        "vertices": torch.as_tensor(coarse_mesh.vertices, dtype=torch.float32),
        "faces": torch.as_tensor(coarse_mesh.faces, dtype=torch.long),
        "vertex_normals": torch.as_tensor(coarse_mesh.normals, dtype=torch.float32),
        "initial_laplacian": torch.as_tensor(initial_laplacian, dtype=torch.float32),
        "laplacian_target": torch.as_tensor(target_laplacian, dtype=torch.float32),
        "target_confidence": torch.ones(coarse_mesh.num_vertices, dtype=torch.float32),
        "visibility": None if visibility is None else torch.as_tensor(visibility, dtype=torch.bool),
        "target_positions": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_vertices": torch.as_tensor(gt_mesh.vertices, dtype=torch.float32),
        "gt_faces": torch.as_tensor(gt_mesh.faces, dtype=torch.long),
        "metadata": metadata,
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


def _camera_tensors(
    cameras: list[Camera], scale_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = []
    extrinsics = []
    for camera in cameras:
        scaled = camera.intrinsics.copy()
        scaled[0, :] *= scale_xy[0]
        scaled[1, :] *= scale_xy[1]
        intrinsics.append(scaled)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = camera.rotation
        extrinsic[:3, 3] = camera.translation
        extrinsics.append(extrinsic)
    return np.stack(intrinsics), np.stack(extrinsics)


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

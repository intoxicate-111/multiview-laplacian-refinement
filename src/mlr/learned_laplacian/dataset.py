from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .graph_layers import faces_to_edge_index
from .target_scaling import (
    EDGE_SCALE_DEFINITION,
    EDGE_SCALE_SOURCE,
    RAW_LAPLACIAN,
    edge_scale_statistics,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
)


REQUIRED_TENSOR_FIELDS = (
    "images",
    "intrinsics",
    "extrinsics",
    "vertices",
    "faces",
    "vertex_normals",
    "initial_laplacian",
    "laplacian_target",
    "target_confidence",
)


def validate_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a shallow copy of one batch-size-one training sample."""

    if not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]:
        raise ValueError("sample_id must be a non-empty string.")
    missing = [name for name in REQUIRED_TENSOR_FIELDS if name not in sample]
    if missing:
        raise ValueError(f"Sample is missing required fields: {', '.join(missing)}.")
    for name in REQUIRED_TENSOR_FIELDS:
        if not isinstance(sample[name], torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(sample[name]).__name__}.")

    images = sample["images"]
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape [V, 3, H, W], got {tuple(images.shape)}.")
    views = images.shape[0]
    if views < 1 or images.shape[2] < 1 or images.shape[3] < 1:
        raise ValueError("images must contain at least one non-empty view.")
    _expect_shape(sample, "intrinsics", (views, 3, 3))
    _expect_shape(sample, "extrinsics", (views, 4, 4))

    vertices = sample["vertices"]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N, 3], got {tuple(vertices.shape)}.")
    num_vertices = vertices.shape[0]
    if num_vertices < 1:
        raise ValueError("vertices must contain at least one vertex.")
    faces = sample["faces"]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {tuple(faces.shape)}.")
    _expect_shape(sample, "vertex_normals", (num_vertices, 3))
    _expect_shape(sample, "initial_laplacian", (num_vertices, 3))
    _expect_shape(sample, "laplacian_target", (num_vertices, 3))
    _expect_shape(sample, "target_confidence", (num_vertices,))

    if faces.numel() > 0:
        if faces.dtype not in (torch.int32, torch.int64):
            raise ValueError("faces must use an integer dtype.")
        minimum = int(faces.min().item())
        maximum = int(faces.max().item())
        if minimum < 0 or maximum >= num_vertices:
            raise ValueError(
                f"faces contain vertex indices outside [0, {num_vertices - 1}]: "
                f"min={minimum}, max={maximum}."
            )

    visibility = sample.get("visibility")
    if visibility is not None:
        if not isinstance(visibility, torch.Tensor):
            raise TypeError("visibility must be a torch.Tensor or None.")
        if tuple(visibility.shape) != (views, num_vertices):
            raise ValueError(
                f"visibility must have shape [V, N] = [{views}, {num_vertices}], "
                f"got {tuple(visibility.shape)}."
            )

    for name in REQUIRED_TENSOR_FIELDS:
        tensor = sample[name]
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if torch.any(sample["target_confidence"] < 0):
        raise ValueError("target_confidence must be non-negative.")
    if images.is_floating_point() and (torch.any(images < 0) or torch.any(images > 1)):
        raise ValueError("floating-point images must be scaled to [0, 1].")

    result = dict(sample)
    result["faces"] = faces.to(dtype=torch.long)
    if visibility is not None:
        result["visibility"] = visibility.to(dtype=torch.bool)
    return _ensure_target_scaling_fields(result)


def load_prepared_sample(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {
                name: (str(archive[name].item()) if name == "sample_id" else torch.from_numpy(archive[name]))
                for name in archive.files
            }
            payload.setdefault("visibility", None)
    else:
        raise ValueError("Prepared samples must use .pt, .pth, or .npz.")
    if not isinstance(payload, Mapping):
        raise ValueError("Prepared sample file must contain a mapping.")
    return validate_sample(payload)


def save_prepared_sample(sample: Mapping[str, Any], path: str | Path) -> Path:
    validated = validate_sample(sample)
    path = Path(path)
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("save_prepared_sample currently writes .pt or .pth files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_sample = {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in validated.items()
    }
    torch.save(cpu_sample, path)
    return path


def move_sample_to_device(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in sample.items()
    }


def _expect_shape(sample: Mapping[str, Any], name: str, expected: tuple[int, ...]) -> None:
    actual = tuple(sample[name].shape)
    if actual != expected:
        expected_text = ", ".join(str(value) for value in expected)
        raise ValueError(f"{name} must have shape [{expected_text}], got {actual}.")


def _ensure_target_scaling_fields(sample: dict[str, Any]) -> dict[str, Any]:
    vertices = sample["vertices"]
    num_vertices = vertices.shape[0]
    raw_target = sample.get("raw_laplacian_target", sample["laplacian_target"])
    if not isinstance(raw_target, torch.Tensor) or tuple(raw_target.shape) != (num_vertices, 3):
        raise ValueError("raw_laplacian_target must have shape [N, 3].")
    edge_index = sample.get("edge_index")
    if edge_index is None:
        edge_index = faces_to_edge_index(sample["faces"], num_vertices)
    local_edge_length = sample.get("local_edge_length")
    if local_edge_length is None:
        local_edge_length = mean_incident_edge_length(vertices, edge_index)
    if not isinstance(local_edge_length, torch.Tensor) or tuple(local_edge_length.shape) != (
        num_vertices,
    ):
        raise ValueError("local_edge_length must have shape [N].")
    local_edge_scale = sample.get("local_edge_scale", local_edge_length.square())
    if not isinstance(local_edge_scale, torch.Tensor) or tuple(local_edge_scale.shape) != (
        num_vertices,
    ):
        raise ValueError("local_edge_scale must have shape [N].")
    metadata = dict(sample.get("metadata", {}))
    epsilon = float(metadata.get("edge_scale_epsilon", 1e-12))
    normalized_target = sample.get("normalized_laplacian_target")
    if normalized_target is None:
        normalized_target = normalize_laplacian_by_edge_scale(
            raw_target, local_edge_length, eps=epsilon
        )
    if not isinstance(normalized_target, torch.Tensor) or tuple(normalized_target.shape) != (
        num_vertices,
        3,
    ):
        raise ValueError("normalized_laplacian_target must have shape [N, 3].")
    for name, tensor in (
        ("local_edge_length", local_edge_length),
        ("local_edge_scale", local_edge_scale),
        ("raw_laplacian_target", raw_target),
        ("normalized_laplacian_target", normalized_target),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    metadata.setdefault("laplacian_target_mode", RAW_LAPLACIAN)
    metadata.setdefault("edge_scale_definition", EDGE_SCALE_DEFINITION)
    metadata.setdefault("edge_scale_source", EDGE_SCALE_SOURCE)
    metadata.setdefault("edge_scale_epsilon", epsilon)
    metadata["edge_scale_statistics"] = edge_scale_statistics(local_edge_length)
    sample["metadata"] = metadata
    sample["raw_laplacian_target"] = raw_target
    sample["normalized_laplacian_target"] = normalized_target
    sample["local_edge_length"] = local_edge_length
    sample["local_edge_scale"] = local_edge_scale
    return sample

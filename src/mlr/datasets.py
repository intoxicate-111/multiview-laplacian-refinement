from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Array, Camera, ReconstructionInput


def load_reconstruction_input(dataset_path: str | Path) -> ReconstructionInput:
    dataset_path = Path(dataset_path)
    root = dataset_path.parent
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cameras_path = _resolve(root, payload["cameras_path"])
    cameras = load_cameras_json(cameras_path)
    image_paths = [_resolve(root, path) for path in payload["image_paths"]]
    mask_paths = None
    if payload.get("mask_paths") is not None:
        mask_paths = [_resolve(root, path) for path in payload["mask_paths"]]
    gt_mesh_path = None
    if payload.get("mesh_path") is not None:
        gt_mesh_path = _resolve(root, payload["mesh_path"])
    return ReconstructionInput(
        image_paths=image_paths,
        cameras=cameras,
        mask_paths=mask_paths,
        gt_mesh_path=gt_mesh_path,
        metadata={"dataset_path": str(dataset_path), "root": str(root)},
    )


def load_cameras_json(path: str | Path) -> list[Camera]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = []
    for item in payload:
        image_size = item.get("image_size")
        cameras.append(
            Camera(
                intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                rotation=np.asarray(item["rotation"], dtype=np.float64),
                translation=np.asarray(item["translation"], dtype=np.float64),
                image_size=tuple(image_size) if image_size is not None else None,
                name=item.get("name"),
            )
        )
    return cameras


def load_masks(mask_paths: list[str | Path] | None) -> list[Array] | None:
    if mask_paths is None:
        return None
    masks = []
    for path in mask_paths:
        image = Image.open(path).convert("L")
        masks.append(np.asarray(image) > 0)
    return masks


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path

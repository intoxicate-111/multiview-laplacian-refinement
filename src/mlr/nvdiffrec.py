from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Camera
from .data import ReconstructionInput
from .datasets import load_masks


@dataclass(frozen=True)
class NvdiffrecRunConfig:
    iterations: int = 1000
    save_interval: int = 100
    texture_res: tuple[int, int] = (1024, 1024)
    train_res: tuple[int, int] = (512, 512)
    batch: int = 4
    learning_rate: tuple[float, float] = (0.03, 0.01)
    dmtet_grid: int = 64
    mesh_scale: float = 2.4
    laplace_scale: float = 3000.0
    background: str = "white"
    random_textures: bool = True
    validate: bool = False
    isosurface: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class NvdiffrecPreparedRun:
    nerf_dataset_dir: Path
    config_path: Path
    nvdiffrec_out_dir: Path


def prepare_nvdiffrec_run(
    data: ReconstructionInput,
    run_dir: str | Path,
    out_name: str,
    config: NvdiffrecRunConfig | None = None,
    convert_cv_to_gl: bool = True,
    masks: list[np.ndarray] | None = None,
) -> NvdiffrecPreparedRun:
    config = config or NvdiffrecRunConfig()
    run_dir = Path(run_dir).resolve()
    nerf_dataset_dir = run_dir / "nerf_dataset"
    nvdiffrec_out_dir = run_dir / "nvdiffrec_out" / out_name
    config_path = run_dir / "nvdiffrec_config.json"
    export_nvdiffrec_nerf_dataset(
        data,
        nerf_dataset_dir,
        convert_cv_to_gl=convert_cv_to_gl,
        masks=masks,
    )
    write_nvdiffrec_config(
        config_path=config_path,
        nerf_dataset_dir=nerf_dataset_dir,
        nvdiffrec_out_dir=nvdiffrec_out_dir,
        config=config,
    )
    return NvdiffrecPreparedRun(
        nerf_dataset_dir=nerf_dataset_dir,
        config_path=config_path,
        nvdiffrec_out_dir=nvdiffrec_out_dir,
    )


def export_nvdiffrec_nerf_dataset(
    data: ReconstructionInput,
    out_dir: str | Path,
    convert_cv_to_gl: bool = True,
    masks: list[np.ndarray] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    masks = load_masks(data.mask_paths) if masks is None else masks
    frames = []
    for idx, (image_path, camera) in enumerate(zip(data.image_paths, data.cameras, strict=True)):
        rgba = _load_rgba(image_path, None if masks is None else masks[idx])
        frame_stem = f"images/{idx:04d}"
        Image.fromarray(rgba).save(out_dir / f"{frame_stem}.png")
        frames.append(
            {
                "file_path": frame_stem,
                "transform_matrix": _camera_to_nerf_transform(
                    camera,
                    convert_cv_to_gl=convert_cv_to_gl,
                ).tolist(),
            }
        )

    first = data.cameras[0]
    if first.image_size is None:
        width = int(Image.open(data.image_paths[0]).size[0])
    else:
        width = int(first.image_size[0])
    camera_angle_x = 2.0 * math.atan(width / (2.0 * float(first.intrinsics[0, 0])))
    payload = {"camera_angle_x": camera_angle_x, "frames": frames}
    for name in ("transforms_train.json", "transforms_val.json"):
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return out_dir


def write_nvdiffrec_config(
    config_path: str | Path,
    nerf_dataset_dir: str | Path,
    nvdiffrec_out_dir: str | Path,
    config: NvdiffrecRunConfig | None = None,
) -> Path:
    config = config or NvdiffrecRunConfig()
    payload: dict[str, object] = {
        "ref_mesh": str(Path(nerf_dataset_dir)),
        "random_textures": config.random_textures,
        "iter": config.iterations,
        "save_interval": config.save_interval,
        "texture_res": list(config.texture_res),
        "train_res": list(config.train_res),
        "batch": config.batch,
        "learning_rate": list(config.learning_rate),
        "dmtet_grid": config.dmtet_grid,
        "mesh_scale": config.mesh_scale,
        "laplace_scale": config.laplace_scale,
        "background": config.background,
        "validate": config.validate,
        "out_dir": str(Path(nvdiffrec_out_dir)),
    }
    if config.isosurface is not None:
        payload["isosurface"] = config.isosurface
    payload.update(config.extra)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return config_path


def find_nvdiffrec_mesh(out_dir: str | Path, preferred: str | Path | None = None) -> Path:
    if preferred is not None:
        path = Path(preferred)
        if path.exists():
            return path
        raise FileNotFoundError(f"Requested nvdiffrec mesh was not found: {path}")

    out_dir = Path(out_dir)
    candidates = [
        out_dir / "mesh" / "mesh.obj",
        out_dir / "mesh.obj",
        out_dir / "final.obj",
    ]
    candidates.extend(sorted(out_dir.rglob("*.obj")) if out_dir.exists() else [])
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"No OBJ mesh found under nvdiffrec output directory: {out_dir}")


def copy_nvdiffrec_mesh(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _load_rgba(image_path: str | Path, mask: np.ndarray | None) -> np.ndarray:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    if mask is None:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = (mask.astype(np.uint8) * 255)
    return np.dstack([rgb, alpha])


def _camera_to_nerf_transform(camera: Camera, convert_cv_to_gl: bool = True) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = camera.rotation.T
    c2w[:3, 3] = camera.center
    if convert_cv_to_gl:
        c2w[:3, 1:3] *= -1.0
    return c2w

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from mlr.data import Camera, Mesh
from mlr.coarse import write_colmap_text_model
from mlr.io import save_mesh


# These are the only prepared-sample fields that an external baseline adapter
# may consume.  In particular, supervision/evaluation fields such as
# target_positions, gt_vertices, gt_faces and Laplacian targets are excluded.
ALLOWED_INPUT_FIELDS = frozenset(
    {
        "sample_id",
        "vertices",
        "faces",
        "image_paths",
        "images",
        "intrinsics",
        "extrinsics",
        "_dataset_root",
    }
)


@dataclass(frozen=True)
class ExternalSceneExport:
    sample_id: str
    scene_dir: Path
    initial_obj: Path
    initial_ply: Path
    view_count: int
    metadata_path: Path


def export_openmvs_scene(
    sample: Mapping[str, Any], output_dir: str | Path
) -> ExternalSceneExport:
    """Export cameras/images and the current mesh for OpenMVS RefineMesh."""

    core = _baseline_input(sample)
    output = Path(output_dir).resolve()
    mesh = _current_mesh(core)
    initial_obj, initial_ply = _write_initial_meshes(mesh, output)
    images = _resolved_images(core)
    cameras = _cameras(core, images)
    colmap_dir = output / "colmap"
    write_colmap_text_model(colmap_dir, images, cameras, copy_images=True)
    return _write_metadata(
        core,
        output,
        initial_obj,
        initial_ply,
        images,
        "openmvs_refinemesh",
        {
            "colmap_dir": str(colmap_dir),
            "interface_command_contract": (
                "InterfaceCOLMAP -i <colmap_dir>/sparse "
                "--image-folder <colmap_dir>/images -o scene.mvs; "
                "RefineMesh -i scene.mvs -m initial_current.ply"
            ),
        },
    )


def export_nds_scene(
    sample: Mapping[str, Any], output_dir: str | Path
) -> ExternalSceneExport:
    """Export RGB(A), OpenCV cameras and current mesh for official NDS."""

    core = _baseline_input(sample)
    output = Path(output_dir).resolve()
    views_dir = output / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    mesh = _current_mesh(core)
    initial_obj, initial_ply = _write_initial_meshes(mesh, output)
    images = _resolved_images(core)
    cameras = _cameras(core, images)
    for index, (image_path, camera) in enumerate(
        zip(images, cameras, strict=True), start=1
    ):
        stem = f"{index:04d}"
        _write_rgba_without_gt(image_path, views_dir / f"{stem}.png")
        np.savetxt(views_dir / f"{stem}_k.txt", camera.intrinsics)
        np.savetxt(views_dir / f"{stem}_r.txt", camera.rotation)
        np.savetxt(views_dir / f"{stem}_t.txt", camera.translation.reshape(3, 1))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    span = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-8)
    padding = 0.05 * span
    bbox_path = output / "bbox.txt"
    np.savetxt(bbox_path, np.stack((vertices.min(axis=0) - padding, vertices.max(axis=0) + padding)))
    return _write_metadata(
        core,
        output,
        initial_obj,
        initial_ply,
        images,
        "nds",
        {
            "views_dir": str(views_dir),
            "bbox": str(bbox_path),
            "mask_source": "RGB non-background pixels; no GT geometry",
        },
    )


def export_nerf_scene(
    sample: Mapping[str, Any], output_dir: str | Path, *, method: str
) -> ExternalSceneExport:
    """Export one current-mesh scene for NeRF2Mesh or ExMesh.

    The emitted NeRF/Blender camera convention is accepted by both official
    repositories.  Image masks are derived only from the black RGB background.
    """

    if method not in {"nerf2mesh", "exmesh"}:
        raise ValueError("method must be 'nerf2mesh' or 'exmesh'.")
    core = _baseline_input(sample)
    output = Path(output_dir).resolve()
    images_dir = output / "images"
    masks_dir = output / "mask"
    train_masks_dir = output / "train_mask"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if method == "exmesh":
        train_masks_dir.mkdir(parents=True, exist_ok=True)
    mesh = _current_mesh(core)
    initial_obj, initial_ply = _write_initial_meshes(mesh, output)
    # ExMesh discovers its initialization at the fixed scene-root path.
    if method == "exmesh":
        write_ascii_ply(mesh, output / "mesh.ply")
    images = _resolved_images(core)
    cameras = _cameras(core, images)
    if method == "exmesh":
        # The official ExMesh loader selects COLMAP whenever ``sparse`` is
        # present.  This avoids its Blender loader concatenating duplicated
        # train/test transform files and keeps the input at exactly 28 views.
        write_colmap_text_model(output, images, cameras, copy_images=True)
        sparse = output / "sparse"
        sparse_zero = sparse / "0"
        sparse_zero.mkdir(parents=True, exist_ok=True)
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            (sparse / name).replace(sparse_zero / name)

    frames = []
    for index, (image_path, camera) in enumerate(
        zip(images, cameras, strict=True), start=1
    ):
        stem = f"{index:04d}" if method == "nerf2mesh" else f"{index:08d}"
        destination = images_dir / f"{stem}.png"
        if method == "nerf2mesh":
            _link_or_convert_png(image_path, destination)
        mask = _rgb_foreground_mask(image_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"{stem}.png")
        if method == "exmesh":
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                train_masks_dir / f"{stem}_gtmask.png"
            )
        if method == "nerf2mesh":
            frames.append(
                {
                    "file_path": f"images/{stem}",
                    "transform_matrix": _camera_to_nerf(camera).tolist(),
                }
            )
    first = cameras[0]
    width, height = first.image_size or Image.open(images[0]).size
    transforms_path = None
    if method == "nerf2mesh":
        transforms = {
            "w": int(width),
            "h": int(height),
            "fl_x": float(first.intrinsics[0, 0]),
            "fl_y": float(first.intrinsics[1, 1]),
            "cx": float(first.intrinsics[0, 2]),
            "cy": float(first.intrinsics[1, 2]),
            "frames": frames,
        }
        # The same 28 observations are inputs, not a tuning/evaluation split.
        for split in ("train", "val", "test"):
            (output / f"transforms_{split}.json").write_text(
                json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
            )
        transforms_path = str(output / "transforms_train.json")
    return _write_metadata(
        core,
        output,
        initial_obj,
        initial_ply,
        images,
        method,
        {
            "camera_format": (
                "NeRF/Blender camera-to-world"
                if method == "nerf2mesh"
                else "COLMAP text model in sparse/0"
            ),
            "mask_source": "RGB non-background pixels; no GT geometry",
            "transforms": transforms_path,
        },
    )


def export_nvdiffrec_scene(
    sample: Mapping[str, Any], output_dir: str | Path
) -> ExternalSceneExport:
    """Export exact per-view cameras and the current mesh for nvdiffrec DLMesh.

    The official DatasetNERF assumes centered single-FOV cameras.  The project
    wrapper consumes the additional exact-intrinsics fields written here while
    leaving nvdiffrec's official fixed-topology ``DLMesh`` optimizer unchanged.
    """

    core = _baseline_input(sample)
    output = Path(output_dir).resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    mesh = _current_mesh(core)
    initial_obj, initial_ply = _write_initial_meshes(mesh, output)
    # The official nvdiffrec OBJ loader assumes that every input OBJ names at
    # least one material.  Our generic mesh writer intentionally emits only
    # geometry, so add the smallest legal material contract here.  This changes
    # neither vertex positions nor face ordering and is specific to this adapter.
    material_path = output / "initial_current.mtl"
    material_path.write_text(
        "newmtl defaultMat\n"
        "Kd 0.5 0.5 0.5\n"
        "Ks 0.0 0.0 0.0\n"
        "Ns 1.0\n",
        encoding="utf-8",
    )
    original_obj = initial_obj.read_text(encoding="utf-8")
    initial_obj.write_text(
        f"mtllib {material_path.name}\nusemtl defaultMat\n{original_obj}",
        encoding="utf-8",
    )
    images = _resolved_images(core)
    cameras = _cameras(core, images)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = []
    for index, (image_path, camera) in enumerate(zip(images, cameras, strict=True)):
        stem = f"{index:04d}"
        destination = images_dir / f"{stem}.png"
        _write_rgba_without_gt(image_path, destination)
        world_to_camera_cv = np.eye(4, dtype=np.float64)
        world_to_camera_cv[:3, :3] = camera.rotation
        world_to_camera_cv[:3, 3] = camera.translation
        width, height = camera.image_size or Image.open(image_path).size
        frames.append(
            {
                "file_path": f"images/{stem}",
                "intrinsics": np.asarray(camera.intrinsics).tolist(),
                "world_to_camera_opengl": (cv_to_gl @ world_to_camera_cv).tolist(),
                "resolution_wh": [int(width), int(height)],
            }
        )
    nominal_fov_x = 2.0 * np.arctan(
        frames[0]["resolution_wh"][0] / (2.0 * frames[0]["intrinsics"][0][0])
    )
    transforms = {
        "schema": "synthetic_exact_intrinsics_nvdiffrec_adapter_v1",
        "camera_angle_x": float(nominal_fov_x),
        "frames": frames,
    }
    for name in ("transforms_train.json", "transforms_test.json"):
        (output / name).write_text(json.dumps(transforms, indent=2) + "\n", encoding="utf-8")
    return _write_metadata(
        core,
        output,
        initial_obj,
        initial_ply,
        images,
        "nvdiffrec",
        {
            "camera_format": "exact OpenCV K plus OpenCV-to-OpenGL world-to-camera",
            "mask_source": "RGB non-background pixels; no GT geometry",
            "material_adapter": (
                "constant default OBJ material required by the official loader; "
                "geometry and vertex/face ordering unchanged"
            ),
            "uv_adapter": (
                "official xatlas parameterization is added by the runtime wrapper; "
                "position indices, vertex positions, and face ordering unchanged"
            ),
            "geometry_path": "official DLMesh fixed-topology optimization via --base_mesh",
            "transforms": str(output / "transforms_train.json"),
        },
    )


def write_ascii_ply(mesh: Mesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment generated without GT by multiview-laplacian-refinement\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return path


def _baseline_input(sample: Mapping[str, Any]) -> dict[str, Any]:
    # Explicit selection is a leakage boundary: never copy the full sample.
    core = {name: sample[name] for name in ALLOWED_INPUT_FIELDS if name in sample}
    required = {"sample_id", "vertices", "faces", "intrinsics", "extrinsics"}
    missing = sorted(required - core.keys())
    if missing:
        raise ValueError("External baseline sample is missing: " + ", ".join(missing))
    if "image_paths" not in core and "images" not in core:
        raise ValueError("External baseline requires image_paths or images.")
    return core


def _current_mesh(core: Mapping[str, Any]) -> Mesh:
    return Mesh(
        _numpy(core["vertices"]).astype(np.float64),
        _numpy(core["faces"]).astype(np.int64),
    ).ensure_normals()


def _resolved_images(core: Mapping[str, Any]) -> list[Path]:
    values = core.get("image_paths")
    if not isinstance(values, list) or not values:
        raise ValueError("File-backed image_paths are required for external baselines.")
    root = Path(str(core.get("_dataset_root", "."))).resolve()
    paths = [Path(value) if Path(value).is_absolute() else root / value for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing external-baseline RGB inputs: " + ", ".join(missing))
    return [path.resolve() for path in paths]


def _cameras(core: Mapping[str, Any], images: list[Path]) -> list[Camera]:
    intrinsics = _numpy(core["intrinsics"])
    extrinsics = _numpy(core["extrinsics"])
    if intrinsics.shape != (len(images), 3, 3) or extrinsics.shape != (
        len(images),
        4,
        4,
    ):
        raise ValueError("External baseline camera tensors do not match image count.")
    result = []
    for index, image in enumerate(images):
        with Image.open(image) as opened:
            size = opened.size
        result.append(
            Camera(
                intrinsics=intrinsics[index],
                rotation=extrinsics[index, :3, :3],
                translation=extrinsics[index, :3, 3],
                image_size=size,
                name=f"view_{index:04d}",
            )
        )
    return result


def _write_initial_meshes(mesh: Mesh, output: Path) -> tuple[Path, Path]:
    initial_obj = output / "initial_current.obj"
    initial_ply = output / "initial_current.ply"
    save_mesh(mesh, initial_obj)
    write_ascii_ply(mesh, initial_ply)
    return initial_obj, initial_ply


def _write_rgba_without_gt(source: Path, destination: Path) -> None:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    alpha = _rgb_foreground_mask(source).astype(np.uint8) * 255
    Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA").save(destination)


def _rgb_foreground_mask(source: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    # Future2000's synthetic renderer uses a fixed black background.  This is
    # image-only preprocessing and has no access to target/GT geometry.
    return np.any(rgb != np.asarray((0, 0, 0), dtype=np.uint8), axis=-1)


def _link_or_convert_png(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".png":
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source.resolve(), destination)
    else:
        Image.open(source).convert("RGB").save(destination)


def _camera_to_nerf(camera: Camera) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = camera.rotation.T
    c2w[:3, 3] = camera.center
    c2w[:3, 1:3] *= -1.0
    return c2w


def _write_metadata(
    core: Mapping[str, Any],
    output: Path,
    initial_obj: Path,
    initial_ply: Path,
    images: list[Path],
    method: str,
    extra: Mapping[str, Any],
) -> ExternalSceneExport:
    payload = {
        "sample_id": str(core["sample_id"]),
        "method": method,
        "input_contract": "current mesh + same 28 RGB images + same cameras; no GT",
        "consumed_sample_fields": sorted(core.keys()),
        "forbidden_fields_consumed": [],
        "initial_obj": str(initial_obj),
        "initial_ply": str(initial_ply),
        "initial_obj_sha256": _sha256(initial_obj),
        "view_count": len(images),
        "source_images": [str(path) for path in images],
        "mesh_coordinate_transform_to_method_world": "identity",
        "method_output_coordinate_transform_to_gt": "identity",
        **dict(extra),
    }
    metadata_path = output / "input_contract.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ExternalSceneExport(
        sample_id=str(core["sample_id"]),
        scene_dir=output,
        initial_obj=initial_obj,
        initial_ply=initial_ply,
        view_count=len(images),
        metadata_path=metadata_path,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

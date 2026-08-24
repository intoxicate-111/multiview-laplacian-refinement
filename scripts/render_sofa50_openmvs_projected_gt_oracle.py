#!/usr/bin/env python3
from __future__ import annotations

"""Render matched OpenMVS projected-GT oracle diagnostic panels."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import _point_to_surface_distances
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.synthetic import SyntheticRenderConfig, render_mesh_face_ids, render_mesh_view


ARMS = (
    ("initial", "INITIAL"),
    ("projected_gt_position_oracle", "PROJECTED-GT POSITION ORACLE"),
    ("projected_gt_laplacian_oracle", "PROJECTED-GT LAPLACIAN ORACLE"),
    ("predicted_laplacian_recovery", "LEARNED PREDICTION"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _camera(static: Mapping[str, Any], image_size: int, view: int) -> Camera:
    intrinsic = _numpy(static["intrinsics"])[view].astype(np.float64, copy=True)
    image_path = Path(str(static["image_paths"][view]))
    if not image_path.is_absolute():
        image_path = Path(str(static["_dataset_root"])) / image_path
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    intrinsic[0] *= image_size / source_width
    intrinsic[1] *= image_size / source_height
    intrinsic[2, 2] = 1.0
    extrinsic = _numpy(static["extrinsics"])[view]
    return Camera(
        intrinsics=intrinsic,
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(image_size, image_size),
        name=f"prepared_fixed_view_{view:02d}",
    )


def _grid(images: list[tuple[str, np.ndarray]], path: Path, columns: int = 2) -> None:
    if not images:
        raise ValueError("No images supplied.")
    height, width = images[0][1].shape[:2]
    label_height = 28
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 5, y + 6), label, fill=(245, 245, 245))
        canvas.paste(Image.fromarray(image.astype(np.uint8)), (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _face_normals(mesh: Mesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _normal_map(mesh: Mesh, face_ids: np.ndarray, camera: Camera) -> np.ndarray:
    normals = _face_normals(mesh) @ camera.rotation.T
    colors = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
    image = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    valid = face_ids >= 0
    image[valid] = colors[face_ids[valid]]
    return image


def _turbo_map(face_ids: np.ndarray, face_values: np.ndarray, high: float) -> np.ndarray:
    from matplotlib import colormaps

    normalized = np.clip(face_values / max(high, 1e-12), 0.0, 1.0)
    colors = np.asarray(colormaps["turbo"](normalized)[:, :3] * 255, dtype=np.uint8)
    image = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    valid = face_ids >= 0
    image[valid] = colors[face_ids[valid]]
    return image


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.diagnostic_root.resolve()
    output = root / "visualizations"
    selection = _read_json(root / "visual_selection_request.json")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    config = SyntheticRenderConfig(
        width=args.image_size,
        height=args.image_size,
        render_mode="lit",
        backend=args.backend,
        normalize_mesh=False,
        background_color=(10, 10, 10),
        object_color=(188, 205, 222),
        light_direction=(0.4, -0.6, 0.7),
        backface_culling=False,
    )
    categories = {
        sample_id: category
        for category in ("best", "worst", "difficult")
        for sample_id in selection[category]
    }
    records = []
    for ordinal, sample_id in enumerate(selection["sample_ids"], 1):
        static = dataset.load_static(index_by_id[sample_id])
        camera = _camera(static, args.image_size, args.view)
        sample_dir = root / "samples" / sample_id
        gt = load_mesh(sample_dir / "gt.obj").ensure_normals()
        meshes = {
            key: load_mesh(sample_dir / f"{key}.obj").ensure_normals()
            for key, _ in ARMS
        }
        face_ids = {
            key: render_mesh_face_ids(mesh, camera, config) for key, mesh in meshes.items()
        }
        shaded = [
            (label, render_mesh_view(meshes[key], camera, config)[0])
            for key, label in ARMS
        ]
        prefix = f"{ordinal:02d}_{categories[sample_id]}_{sample_id}"
        shaded_path = output / "shaded" / f"{prefix}.png"
        _grid(shaded, shaded_path)

        normal_images = [
            (label, _normal_map(meshes[key], face_ids[key], camera))
            for key, label in ARMS
        ]
        normal_path = output / "normal_maps" / f"{prefix}.png"
        _grid(normal_images, normal_path)

        distance_values: dict[str, np.ndarray] = {}
        distance_engines = {}
        for key, _ in ARMS:
            values, engine = _point_to_surface_distances(meshes[key].vertices, gt)
            distance_values[key] = values
            distance_engines[key] = engine
        distance_high = float(
            np.quantile(np.concatenate(list(distance_values.values())), 0.99)
        )
        distance_images = []
        for key, label in ARMS:
            face_values = distance_values[key][meshes[key].faces].mean(axis=1)
            distance_images.append(
                (label, _turbo_map(face_ids[key], face_values, distance_high))
            )
        distance_path = output / "gt_surface_distance" / f"{prefix}.png"
        _grid(distance_images, distance_path)

        initial_vertices = meshes["initial"].vertices
        displacement_values = {
            key: np.linalg.norm(mesh.vertices - initial_vertices, axis=1)
            for key, mesh in meshes.items()
        }
        displacement_high = float(
            np.quantile(np.concatenate(list(displacement_values.values())), 0.99)
        )
        displacement_images = []
        for key, label in ARMS:
            face_values = displacement_values[key][meshes[key].faces].mean(axis=1)
            displacement_images.append(
                (label, _turbo_map(face_ids[key], face_values, displacement_high))
            )
        displacement_path = output / "vertex_displacement" / f"{prefix}.png"
        _grid(displacement_images, displacement_path)

        with np.load(sample_dir / "expanded_position_oracle.npz") as archive:
            expanded_initial = Mesh(archive["initial_vertices"], archive["faces"]).ensure_normals()
            expanded_oracle = Mesh(archive["projected_vertices"], archive["faces"]).ensure_normals()
        expansion_entries = [
            ("OPENMVS INITIAL", render_mesh_view(meshes["initial"], camera, config)[0]),
            ("COARSE POSITION ORACLE", render_mesh_view(meshes["projected_gt_position_oracle"], camera, config)[0]),
            ("MIDPOINT-EXPANDED INITIAL", render_mesh_view(expanded_initial, camera, config)[0]),
            ("EXPANDED POSITION ORACLE", render_mesh_view(expanded_oracle, camera, config)[0]),
        ]
        expansion_path = output / "expansion" / f"{prefix}.png"
        _grid(expansion_entries, expansion_path)

        arrays_path = output / "heatmap_values" / f"{prefix}.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            **{f"distance_{key}": value for key, value in distance_values.items()},
            **{f"displacement_{key}": value for key, value in displacement_values.items()},
        )
        records.append(
            {
                "sample_id": sample_id,
                "category": categories[sample_id],
                "fixed_camera_view": args.view,
                "image_size": args.image_size,
                "shaded_panel": str(shaded_path),
                "normal_map_panel": str(normal_path),
                "gt_surface_distance_panel": str(distance_path),
                "vertex_displacement_panel": str(displacement_path),
                "expansion_panel": str(expansion_path),
                "heatmap_values": str(arrays_path),
                "distance_engines": distance_engines,
                "distance_color_range": [0.0, distance_high],
                "displacement_color_range": [0.0, displacement_high],
                "same_camera_lighting_material_size_normalization_crop": True,
            }
        )
        print(f"rendered {sample_id} ({categories[sample_id]})", flush=True)
    result = {
        "contract_audit": True,
        "selection": selection,
        "renderer_backend": args.backend,
        "same_camera_lighting_material_size_normalization_crop_within_sample": True,
        "normal_map_definition": "camera-space oriented face normal mapped from [-1,1] to RGB",
        "gt_surface_distance_definition": "output vertex to exact GT triangle surface",
        "vertex_displacement_definition": "per-vertex magnitude relative to exact OpenMVS initial",
        "records": records,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--backend", choices=("opengl", "cpu"), default="opengl")
    result = run(parser.parse_args())
    print(json.dumps({"contract_audit": result["contract_audit"], "samples": len(result["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

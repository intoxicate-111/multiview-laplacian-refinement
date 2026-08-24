#!/usr/bin/env python3
from __future__ import annotations

"""Render fixed-camera evidence for the matched-domain exact-target oracle diagnostic."""

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


STATE_LABELS = (
    ("initial", "INITIAL"),
    ("clean", "EXACT CLEAN"),
    ("exact_target_oracle", "EXACT-TARGET ORACLE"),
    ("predicted_recovery", "PREDICTED RECOVERY"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def camera(static: Mapping[str, Any], image_size: int, view: int) -> Camera:
    intrinsic = numpy(static["intrinsics"])[view].astype(np.float64, copy=True)
    image_path = Path(str(static["image_paths"][view]))
    if not image_path.is_absolute():
        image_path = Path(str(static["_dataset_root"])) / image_path
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    intrinsic[0] *= image_size / source_width
    intrinsic[1] *= image_size / source_height
    intrinsic[2, 2] = 1.0
    extrinsic = numpy(static["extrinsics"])[view]
    return Camera(
        intrinsics=intrinsic,
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(image_size, image_size),
        name=f"prepared_fixed_view_{view:02d}",
    )


def grid(images: list[tuple[str, np.ndarray]], path: Path, columns: int = 2) -> None:
    height, width = images[0][1].shape[:2]
    label_height = 30
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 5, y + 7), label, fill=(245, 245, 245))
        canvas.paste(Image.fromarray(image.astype(np.uint8)), (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def turbo(face_ids: np.ndarray, face_values: np.ndarray, high: float) -> np.ndarray:
    from matplotlib import colormaps

    normalized = np.clip(face_values / max(high, 1e-12), 0.0, 1.0)
    colors = np.asarray(colormaps["turbo"](normalized)[:, :3] * 255, dtype=np.uint8)
    image = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    valid = face_ids >= 0
    image[valid] = colors[face_ids[valid]]
    return image


def mask_image(face_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    valid = face_ids >= 0
    image[valid] = np.where(mask[face_ids[valid], None], np.array([255, 42, 42]), np.array([42, 42, 42]))
    return image


def crosses(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-manifest", required=True, type=Path)
    parser.add_argument("--v2-manifest", required=True, type=Path)
    parser.add_argument("--v1-diagnostic-root", required=True, type=Path)
    parser.add_argument("--v2-diagnostic-root", required=True, type=Path)
    parser.add_argument("--v1-prediction-source", required=True, type=Path)
    parser.add_argument("--v2-prediction-source", required=True, type=Path)
    parser.add_argument("--v1-prediction-arm", default="new_multitopology_rawlap")
    parser.add_argument("--v2-prediction-arm", default="v2_strong_smoothing")
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--backend", choices=("opengl", "cpu"), default="opengl")
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    selection = read_json(report_root / "visual_selection.json")
    resources = {
        "v1_legacy_smoothing": {
            "dataset": PreparedMeshDataset.from_manifest(args.v1_manifest.resolve(), "test"),
            "diagnostic": args.v1_diagnostic_root.resolve(),
            "source": args.v1_prediction_source.resolve(),
            "prediction_arm": args.v1_prediction_arm,
        },
        "v2_strong_smoothing": {
            "dataset": PreparedMeshDataset.from_manifest(args.v2_manifest.resolve(), "test"),
            "diagnostic": args.v2_diagnostic_root.resolve(),
            "source": args.v2_prediction_source.resolve(),
            "prediction_arm": args.v2_prediction_arm,
        },
    }
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
    output = report_root / "visualizations"
    records = []
    for ordinal, request in enumerate(selection["records"], 1):
        dataset_arm = str(request["dataset_arm"])
        sample_id = str(request["sample_id"])
        resource = resources[dataset_arm]
        dataset = resource["dataset"]
        index_by_id = {value: index for index, value in enumerate(dataset.sample_ids)}
        static = dataset.load_static(index_by_id[sample_id])
        initial = Mesh(numpy(static["vertices"]), numpy(static["faces"]).astype(np.int64)).ensure_normals()
        clean = Mesh(
            numpy(static.get("clean_reference_vertices", static["gt_vertices"])),
            numpy(static.get("clean_reference_faces", static["gt_faces"])).astype(np.int64),
        ).ensure_normals()
        oracle = load_mesh(resource["diagnostic"] / "oracle_recovery" / sample_id / "predicted_refined.obj").ensure_normals()
        pred_dir = resource["source"] / "reconstruction" / resource["prediction_arm"] / sample_id
        predicted = load_mesh(pred_dir / "predicted_refined.obj").ensure_normals()
        prediction = np.load(pred_dir / "delta_pred_raw.npy")
        target = numpy(static.get("raw_laplacian_target", static["laplacian_target"]))
        meshes = {
            "initial": initial,
            "clean": clean,
            "exact_target_oracle": oracle,
            "predicted_recovery": predicted,
        }
        fixed_camera = camera(static, args.image_size, args.view)
        face_ids = {key: render_mesh_face_ids(mesh, fixed_camera, config) for key, mesh in meshes.items()}
        prefix = f"{ordinal:02d}_{dataset_arm}_{request['category']}_{sample_id}"
        shaded_path = output / "shaded" / f"{prefix}.png"
        grid(
            [(label, render_mesh_view(meshes[state], fixed_camera, config)[0]) for state, label in STATE_LABELS],
            shaded_path,
        )

        distances = {}
        distance_engines = {}
        for state, _ in STATE_LABELS:
            values, engine = _point_to_surface_distances(meshes[state].vertices, clean)
            distances[state] = values
            distance_engines[state] = engine
        high = float(np.quantile(np.concatenate(list(distances.values())), 0.99))
        distance_path = output / "gt_distance" / f"{prefix}.png"
        grid(
            [
                (label, turbo(face_ids[state], distances[state][meshes[state].faces].mean(axis=1), high))
                for state, label in STATE_LABELS
            ],
            distance_path,
        )

        oracle_pred = np.linalg.norm(predicted.vertices - oracle.vertices, axis=1)
        lap_error = np.linalg.norm(prediction - target, axis=1)
        initial_cross = crosses(initial)
        oracle_flip = np.einsum("ij,ij->i", initial_cross, crosses(oracle)) < 0
        predicted_flip = np.einsum("ij,ij->i", initial_cross, crosses(predicted)) < 0
        displacement_high = float(np.quantile(oracle_pred, 0.99))
        lap_high = float(np.quantile(lap_error, 0.99))
        attribution_path = output / "attribution" / f"{prefix}.png"
        grid(
            [
                (
                    "ORACLE-vs-PRED DISPLACEMENT",
                    turbo(face_ids["predicted_recovery"], oracle_pred[predicted.faces].mean(axis=1), displacement_high),
                ),
                (
                    "PREDICTED LAPLACIAN ERROR",
                    turbo(face_ids["initial"], lap_error[initial.faces].mean(axis=1), lap_high),
                ),
                ("ORACLE INTRODUCED FLIPS", mask_image(face_ids["exact_target_oracle"], oracle_flip)),
                ("PREDICTED INTRODUCED FLIPS", mask_image(face_ids["predicted_recovery"], predicted_flip)),
            ],
            attribution_path,
        )
        values_path = output / "heatmap_values" / f"{prefix}.npz"
        values_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            values_path,
            oracle_vs_predicted_displacement=oracle_pred,
            predicted_laplacian_error=lap_error,
            oracle_introduced_flip_mask=oracle_flip,
            predicted_introduced_flip_mask=predicted_flip,
            **{f"gt_distance_{key}": value for key, value in distances.items()},
        )
        records.append(
            {
                **request,
                "fixed_camera_view": args.view,
                "image_size": args.image_size,
                "shaded_panel": str(shaded_path),
                "gt_distance_panel": str(distance_path),
                "attribution_panel": str(attribution_path),
                "heatmap_values": str(values_path),
                "distance_engines": distance_engines,
                "same_camera_lighting_material_crop_scale": True,
            }
        )
        print(f"rendered {dataset_arm}/{sample_id} ({request['category']})", flush=True)
    result = {
        "contract_audit": True,
        "renderer_backend": args.backend,
        "same_camera_lighting_material_crop_scale_within_each_case": True,
        "selection_fixed_before_rendering": True,
        "records": records,
    }
    (output / "visualization_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"contract_audit": True, "rendered": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

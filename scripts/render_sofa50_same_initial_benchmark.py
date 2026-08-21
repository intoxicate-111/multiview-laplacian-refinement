#!/usr/bin/env python3
from __future__ import annotations

"""Render fixed-camera Group A panels and per-vertex GT-distance PLYs."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import _point_to_surface_distances
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid


METHODS = ("ours", "exmesh", "nds", "nvdiffrec")


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _camera(sample: dict[str, Any], image_size: int, view: int) -> Camera:
    intrinsic = _numpy(sample["intrinsics"])[view].astype(np.float64, copy=True)
    root = Path(str(sample["_dataset_root"]))
    image = Path(sample["image_paths"][view])
    image = image if image.is_absolute() else root / image
    with Image.open(image) as opened:
        width, height = opened.size
    intrinsic[0] *= image_size / width
    intrinsic[1] *= image_size / height
    intrinsic[2, 2] = 1.0
    extrinsic = _numpy(sample["extrinsics"])[view]
    return Camera(
        intrinsics=intrinsic,
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(image_size, image_size),
        name=f"fixed_prepared_view_{view:02d}",
    )


def _write_colored_ply(path: Path, mesh: Mesh, colors: np.ndarray) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _colors(values: np.ndarray, high: float) -> np.ndarray:
    from matplotlib import colormaps

    normalized = np.clip(np.asarray(values) / max(high, 1e-12), 0.0, 1.0)
    return np.asarray(np.rint(colormaps["turbo"](normalized)[:, :3] * 255), dtype=np.uint8)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select(rows: list[dict[str, str]], count: int) -> list[str]:
    ours = sorted(
        (
            (float(row["chamfer_improvement_percent"]), row["sample_id"])
            for row in rows
            if row["method"] == "ours"
            and row["status"] == "completed"
            and row["chamfer_improvement_percent"] not in ("", None)
        ),
        key=lambda item: item[0],
    )
    if len(ours) < count:
        raise ValueError(f"Need {count} completed ours rows, found {len(ours)}")
    indices = np.linspace(0, len(ours) - 1, count).round().astype(int)
    return [ours[int(index)][1] for index in indices]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.per_sample_csv)
    selected = _select(rows, args.count)
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_by_id = {row["sample_id"]: row for row in manifest["samples"]}
    records = []
    for ordinal, sample_id in enumerate(selected, start=1):
        static = dataset.load_static(index_by_id[sample_id])
        gt = Mesh(_numpy(static["gt_vertices"]), _numpy(static["gt_faces"])).ensure_normals()
        source = source_by_id[sample_id]
        entries: list[tuple[str, Mesh]] = [
            ("GT", gt),
            ("COMMON INITIAL", load_mesh(source["common_initial_mesh"])),
        ]
        missing = []
        for method in METHODS:
            path = args.results_root / method / "samples" / sample_id / "refined.obj"
            if path.is_file():
                entries.append((method.upper(), load_mesh(path)))
            else:
                missing.append(method)
        if missing:
            raise FileNotFoundError(f"Missing full outputs for {sample_id}: {missing}")
        panel = args.output_dir / "fixed_camera" / f"{ordinal:02d}_{sample_id}.png"
        render_mesh_comparison_grid(
            entries,
            _camera(static, args.image_size, args.view),
            panel,
            image_size=args.image_size,
            columns=3,
        )
        distances: dict[str, np.ndarray] = {}
        engines = {}
        for label, mesh in entries[1:]:
            values, engine = _point_to_surface_distances(np.asarray(mesh.vertices), gt)
            distances[label] = values
            engines[label] = engine
        high = float(np.quantile(np.concatenate(list(distances.values())), 0.99))
        error_paths = {}
        for label, mesh in entries[1:]:
            path = args.output_dir / "surface_error_ply" / sample_id / f"{label.lower().replace(' ', '_')}.ply"
            _write_colored_ply(path, mesh, _colors(distances[label], high))
            error_paths[label] = str(path)
        metadata = {
            "sample_id": sample_id,
            "fixed_camera_view": args.view,
            "panel": str(panel),
            "surface_error_definition": "output vertex to GT triangle-surface distance",
            "shared_color_range": [0.0, high],
            "shared_color_clipping": "per-sample global p99 across common initial and all four refined outputs",
            "colormap": "turbo",
            "engines": engines,
            "surface_error_ply": error_paths,
        }
        metadata_path = args.output_dir / "surface_error_ply" / sample_id / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        records.append(metadata)
    payload = {
        "status": "completed",
        "selection": "quantiles of ours per-sample Chamfer improvement",
        "sample_count": len(records),
        "identical_fixed_camera_for_every_method_within_each_sample": True,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visualization_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--per-sample-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", default=5, type=int)
    parser.add_argument("--image-size", default=360, type=int)
    parser.add_argument("--view", default=0, type=int)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

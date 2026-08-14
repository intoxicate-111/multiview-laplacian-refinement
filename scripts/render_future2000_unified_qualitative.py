#!/usr/bin/env python3
from __future__ import annotations

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
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid


EXTERNAL = (
    ("OpenMVS", "openmvs_refinemesh"),
    ("NDS", "nds"),
    ("NeRF2Mesh", "nerf2mesh"),
    ("ExMesh", "exmesh"),
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    learned_rows = _read_csv(args.learned_analysis / "per_sample.csv")
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    availability = []
    for row in learned_rows:
        sample_id = row["sample_id"]
        count = sum(
            (args.external_output / method / "samples" / sample_id / "refined.obj").is_file()
            for _, method in EXTERNAL
        )
        availability.append((count, float(row["laplacian_chamfer_improvement_rate"]), row))
    best_count = max(item[0] for item in availability)
    pool = sorted((item for item in availability if item[0] == best_count), key=lambda item: item[1])
    selected = _quantile_select(pool, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, (_, improvement, row) in enumerate(selected, start=1):
        sample_id = row["sample_id"]
        static = dataset.load_static(index_by_id[sample_id])
        gt = Mesh(_numpy(static["gt_vertices"]), _numpy(static["gt_faces"])).ensure_normals()
        sample_dir = args.learned_analysis / "samples" / sample_id
        entries = [
            ("GT", gt),
            ("CURRENT", load_mesh(sample_dir / "initial.obj")),
            ("LEARNED LAPLACIAN", load_mesh(sample_dir / "laplacian_refined.obj")),
            ("DIRECT DISPLACEMENT", load_mesh(sample_dir / "displacement_refined.obj")),
        ]
        present = []
        missing = []
        for label, method in EXTERNAL:
            path = args.external_output / method / "samples" / sample_id / "refined.obj"
            if path.is_file():
                entries.append((label, load_mesh(path)))
                present.append(method)
            else:
                missing.append(method)
        output = args.output_dir / f"{ordinal:02d}_{sample_id}.png"
        render_mesh_comparison_grid(
            entries,
            _camera(static, args.image_size),
            output,
            image_size=args.image_size,
            columns=4,
        )
        records.append(
            {
                "sample_id": sample_id,
                "laplacian_chamfer_improvement_rate": improvement,
                "available_external_methods": present,
                "missing_external_methods": missing,
                "image": output.name,
            }
        )
    payload = {
        "status": "completed",
        "count": len(records),
        "selection": "four quantiles among samples with maximal external-method availability",
        "max_external_methods_available": best_count,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _quantile_select(values: list[tuple[int, float, dict[str, str]]], count: int):
    if len(values) < count:
        raise ValueError(f"Need {count} qualitative samples, found {len(values)}.")
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(position)] for position in positions]


def _camera(sample: dict[str, Any], image_size: int) -> Camera:
    intrinsic = _numpy(sample["intrinsics"])[0].astype(np.float64, copy=True)
    root = Path(str(sample["_dataset_root"]))
    image = Path(sample["image_paths"][0])
    image = image if image.is_absolute() else root / image
    with Image.open(image) as opened:
        width, height = opened.size
    intrinsic[0] *= image_size / width
    intrinsic[1] *= image_size / height
    intrinsic[2, 2] = 1.0
    extrinsic = _numpy(sample["extrinsics"])[0]
    return Camera(
        intrinsics=intrinsic,
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(image_size, image_size),
        name="held_out_view_0001",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--learned-analysis", type=Path, required=True)
    parser.add_argument("--external-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

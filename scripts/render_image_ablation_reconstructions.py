#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.prediction_visualizer import _resolve_camera
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Render fixed-camera ablation mesh grids.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reconstruction-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image-size", default=256, type=int)
    args = parser.parse_args()

    dataset = PreparedMeshDataset.from_manifest(args.manifest, args.split)
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        sample_id = str(sample["sample_id"])
        sample_dir = args.reconstruction_dir / sample_id
        original = sample_dir / "original_rgb"
        entries = [
            ("Initial query mesh", load_mesh(original / "coarse.obj")),
            ("GT-delta oracle", load_mesh(original / "oracle_refined.obj")),
            ("Original RGB", load_mesh(original / "predicted_refined.obj")),
            ("Zero RGB", load_mesh(sample_dir / "zero_rgb" / "predicted_refined.obj")),
            (
                "Shuffled RGB",
                load_mesh(sample_dir / "shuffled_images" / "predicted_refined.obj"),
            ),
            (
                "Cross-object RGB",
                load_mesh(sample_dir / "cross_object_rgb" / "predicted_refined.obj"),
            ),
        ]
        # Lazy samples do not contain an image tensor.  Give the shared camera
        # resolver the true prepared dimensions so it rescales 960-pixel
        # intrinsics to the requested thumbnail instead of rendering off-frame.
        camera_sample = dict(sample)
        prepared_size = int(sample.get("prepared_image_size", args.image_size))
        camera_sample["images"] = np.empty(
            (1, 1, prepared_size, prepared_size), dtype=np.uint8
        )
        camera, _ = _resolve_camera(
            camera_sample, entries[0][1], 0, args.image_size
        )
        output = sample_dir / "comparison.png"
        print(f"Rendering {sample_id} -> {output}", flush=True)
        render_mesh_comparison_grid(
            entries, camera, output, image_size=args.image_size, columns=3
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

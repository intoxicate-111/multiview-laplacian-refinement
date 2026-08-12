#!/usr/bin/env python3
"""Export GT/coarse/refined meshes and fixed-camera metadata for an H2 arm."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", default="B_direct_raw_laplacian")
    parser.add_argument("--expected-count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = PreparedMeshDataset.from_manifest(args.manifest, args.split)
    if len(dataset) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} {args.split} samples, found {len(dataset)}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        sample_id = str(sample["sample_id"])
        source_dir = args.reconstruction_dir / sample_id
        coarse_source = source_dir / "coarse.obj"
        refined_source = source_dir / "predicted_refined.obj"
        missing = [path for path in (coarse_source, refined_source) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing reconstruction meshes for {sample_id}: "
                + ", ".join(str(path) for path in missing)
            )

        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        gt_vertices = _array(sample["gt_vertices"])
        gt_faces = _array(sample["gt_faces"]).astype(np.int64, copy=False)
        save_mesh(Mesh(gt_vertices, gt_faces), sample_dir / "gt.obj")
        shutil.copy2(coarse_source, sample_dir / "coarse.obj")
        shutil.copy2(refined_source, sample_dir / "refined.obj")

        intrinsics = _array(sample["intrinsics"])[0]
        extrinsics = _array(sample["extrinsics"])[0]
        prepared_image_size = int(
            sample.get("prepared_image_size", sample.get("image_width", 960))
        )
        records.append(
            {
                "index": index,
                "sample_id": sample_id,
                "mesh_paths": {
                    "gt": f"{sample_id}/gt.obj",
                    "coarse": f"{sample_id}/coarse.obj",
                    "refined": f"{sample_id}/refined.obj",
                },
                "camera": {
                    "view_index": 0,
                    "intrinsics": intrinsics.tolist(),
                    "extrinsics": extrinsics.tolist(),
                    "prepared_image_size": prepared_image_size,
                },
                "counts": {
                    "gt_vertices": int(len(gt_vertices)),
                    "gt_faces": int(len(gt_faces)),
                },
            }
        )

    payload = {
        "format": "mlr_mesh_comparison_bundle_v1",
        "arm": args.arm,
        "split": args.split,
        "count": len(records),
        "labels": ["GT", "COARSE", "REFINED RESULT"],
        "source_manifest": str(args.manifest.resolve()),
        "source_reconstruction_dir": str(args.reconstruction_dir.resolve()),
        "samples": records,
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(records)} comparison mesh groups to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

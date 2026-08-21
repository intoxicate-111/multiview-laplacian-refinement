#!/usr/bin/env python3
from __future__ import annotations

"""Build a provenance-rich manifest for the controlled Sofa50 refinement benchmark.

This script never creates or perturbs geometry.  It validates the existing
prepared tensors against the already-exported canonical ``coarse.obj`` files
from the HF1920 evaluation and records those files as the common initialization.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_geometry_sha256(vertices: torch.Tensor, faces: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in (vertices.float(), faces.long()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def resolved_sample(source_manifest: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (source_manifest.parent / path).resolve()


def resolved_images(dataset_root: Path, values: list[str]) -> list[Path]:
    result = []
    for value in values:
        path = Path(value)
        result.append(path.resolve() if path.is_absolute() else (dataset_root / path).resolve())
    return result


def recovery_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("arm") == "HF_1920"]
    return {str(row["sample_id"]): row for row in rows}


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest = args.source_manifest.resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    dataset_root = Path(source.get("dataset_root", source_manifest.parent))
    if not dataset_root.is_absolute():
        dataset_root = source_manifest.parent / dataset_root
    dataset_root = dataset_root.resolve()
    records = [dict(row) for row in source["samples"] if row.get("split") == "test"]
    if len(records) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} test samples, found {len(records)}")
    canonical = recovery_rows(args.recovery_csv.resolve())
    output_rows: list[dict[str, Any]] = []
    initial_chamfers: list[tuple[float, str]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        sample_path = resolved_sample(source_manifest, record)
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        if str(sample["sample_id"]) != sample_id:
            raise ValueError(f"Sample ID mismatch for {sample_path}")
        images = resolved_images(dataset_root, list(sample["image_paths"]))
        if len(images) != 28 or any(not path.is_file() for path in images):
            raise FileNotFoundError(f"Incomplete 28-view RGB tuple for {sample_id}")
        sizes = {Image.open(path).size for path in images}
        if sizes != {(1920, 1920)}:
            raise ValueError(f"Unexpected image sizes for {sample_id}: {sizes}")
        vertices = sample["vertices"].detach().cpu().numpy().astype(np.float64)
        faces = sample["faces"].detach().cpu().numpy().astype(np.int64)
        gt_vertices = sample["gt_vertices"].detach().cpu().numpy()
        gt_faces = sample["gt_faces"].detach().cpu().numpy()
        if tuple(sample["intrinsics"].shape) != (28, 3, 3) or tuple(sample["extrinsics"].shape) != (28, 4, 4):
            raise ValueError(f"Camera tensor mismatch for {sample_id}")
        if len(gt_vertices) == 0 or len(gt_faces) == 0:
            raise ValueError(f"Missing GT surface for {sample_id}")
        initial_path = (args.canonical_reconstruction_root / sample_id / "coarse.obj").resolve()
        if not initial_path.is_file():
            raise FileNotFoundError(f"Missing existing canonical coarse mesh: {initial_path}")
        initial = load_mesh(initial_path)
        if initial.num_vertices != len(vertices) or initial.num_faces != len(faces):
            raise ValueError(f"Canonical coarse counts differ from prepared tensors: {sample_id}")
        if not np.array_equal(np.asarray(initial.faces, dtype=np.int64), faces):
            raise ValueError(f"Canonical coarse connectivity differs from prepared tensors: {sample_id}")
        max_vertex_error = float(np.max(np.abs(np.asarray(initial.vertices) - vertices)))
        if max_vertex_error > 1e-6:
            raise ValueError(f"Canonical coarse vertices differ for {sample_id}: {max_vertex_error}")
        metric = canonical.get(sample_id)
        if metric is None:
            raise ValueError(f"Missing canonical HF1920 recovery row: {sample_id}")
        initial_chamfer = float(metric["initial_chamfer"])
        initial_chamfers.append((initial_chamfer, sample_id))
        output_rows.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "path": str(sample_path),
                "common_initial_mesh": str(initial_path),
                "common_initial_mesh_sha256": sha256(initial_path),
                "prepared_tensor_geometry_sha256": tensor_geometry_sha256(sample["vertices"], sample["faces"]),
                "initial_vertex_count": int(len(vertices)),
                "initial_face_count": int(len(faces)),
                "initial_obj_max_abs_vertex_error_vs_prepared_tensor": max_vertex_error,
                "initial_obj_faces_exactly_match_prepared_tensor": True,
                "image_directory": str(images[0].parent),
                "image_paths": [str(path) for path in images],
                "image_sha256": [sha256(path) for path in images],
                "view_count": 28,
                "image_size": [1920, 1920],
                "camera_and_gt_container": str(sample_path),
                "camera_contract": "intrinsics[28,3,3] + world_to_camera extrinsics[28,4,4]",
                "gt_contract": "gt_vertices + gt_faces in the same prepared world frame; evaluation only",
                "visibility_contract": "visibility_backface_and_occlusion[28,V] from prepared current graph",
                "coordinate_transform_to_gt": "identity",
                "canonical_initial_chamfer": initial_chamfer,
                "canonical_ours_mesh": str((args.canonical_reconstruction_root / sample_id / "predicted_refined.obj").resolve()),
            }
        )
    initial_chamfers.sort()
    representative = initial_chamfers[len(initial_chamfers) // 2][1]
    payload = {
        "format_version": "sofa50_same_initial_refinement_benchmark_v1",
        "dataset_role": "controlled_same_prepared_mesh_same_rgb_same_cameras_test_benchmark",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "dataset_root": str(dataset_root),
        "selected_checkpoint_family": "canonical Sofa50 synthetic-current 28-view native-1920 HF",
        "sample_count": len(output_rows),
        "sample_ids": [row["sample_id"] for row in output_rows],
        "representative_sample_id": representative,
        "representative_selection": "median canonical initial Chamfer among the 25 test samples",
        "common_input_contract": "exact existing prepared current mesh + exact 28 native-1920 RGB images + exact prepared cameras",
        "gt_usage": "common evaluation only",
        "geometry_created_or_perturbed": False,
        "samples": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--canonical-reconstruction-root", required=True, type=Path)
    parser.add_argument("--recovery-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=25)
    payload = build(parser.parse_args())
    print(json.dumps({"sample_count": payload["sample_count"], "representative_sample_id": payload["representative_sample_id"], "output": str(Path(payload["source_manifest"]).parent)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.learned_laplacian.dataset import resolve_lazy_image_paths
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_raw(vertices: torch.Tensor, faces: torch.Tensor) -> np.ndarray:
    xyz = vertices.detach().cpu().numpy().astype(np.float64, copy=False)
    tri = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    return apply_uniform_laplacian(
        xyz, build_uniform_laplacian_data(tri, len(xyz))
    )


def _object_id(sample_id: str, metadata: Mapping[str, Any]) -> str:
    value = metadata.get("object_id", metadata.get("source_sample_id"))
    return str(value) if value is not None else sample_id.split("__v", maxsplit=1)[0]


def _paths(sample: Mapping[str, Any]) -> list[Path]:
    return resolve_lazy_image_paths(
        list(sample["image_paths"]), Path(str(sample["_dataset_root"]))
    )


def audit(gt_manifest: Path, current_manifest: Path) -> dict[str, Any]:
    gt_payload = json.loads(gt_manifest.read_text(encoding="utf-8"))
    current_payload = json.loads(current_manifest.read_text(encoding="utf-8"))
    gt_datasets = {
        split: PreparedMeshDataset.from_manifest(gt_manifest, split) for split in SPLITS
    }
    current_datasets = {
        split: PreparedMeshDataset.from_manifest(current_manifest, split)
        for split in SPLITS
    }

    gt_by_object: dict[str, tuple[str, Mapping[str, Any]]] = {}
    gt_raw_errors: list[float] = []
    gt_initial_nonzero: list[str] = []
    gt_identity_failures: list[str] = []
    gt_image_missing: list[str] = []
    gt_sizes: list[tuple[int, int]] = []
    gt_split_objects: dict[str, set[str]] = defaultdict(set)
    for split, dataset in gt_datasets.items():
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            sample_id = str(sample["sample_id"])
            metadata = dict(sample.get("metadata", {}))
            object_id = _object_id(sample_id, metadata)
            gt_by_object[object_id] = (split, sample)
            gt_split_objects[split].add(object_id)
            raw = torch.as_tensor(sample["raw_laplacian_target"]).cpu().numpy()
            gt_raw_errors.append(float(np.max(np.abs(raw - _uniform_raw(
                torch.as_tensor(sample["vertices"]), torch.as_tensor(sample["faces"])
            )))))
            if torch.count_nonzero(torch.as_tensor(sample["initial_laplacian"])).item():
                gt_initial_nonzero.append(sample_id)
            if not (
                torch.equal(sample["vertices"], sample["gt_vertices"])
                and torch.equal(sample["faces"], sample["gt_faces"])
                and torch.equal(sample["vertices"], sample["target_positions"])
            ):
                gt_identity_failures.append(sample_id)
            gt_image_missing.extend(str(path) for path in _paths(sample) if not path.is_file())
            gt_sizes.append((int(sample["vertices"].shape[0]), int(sample["faces"].shape[0])))

    camera_intrinsic_errors: list[float] = []
    camera_extrinsic_errors: list[float] = []
    current_gt_vertex_errors: list[float] = []
    current_gt_face_mismatches: list[str] = []
    current_image_path_mismatches: list[str] = []
    current_image_missing: list[str] = []
    current_split_mismatches: list[str] = []
    current_variants: dict[str, list[int]] = defaultdict(list)
    current_sizes: list[tuple[int, int]] = []
    current_identical_to_gt = 0
    current_initial_formula_errors: list[float] = []
    current_target_formula_errors: list[float] = []
    current_test_ids: list[str] = []
    for split, dataset in current_datasets.items():
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            sample_id = str(sample["sample_id"])
            metadata = dict(sample.get("metadata", {}))
            object_id = _object_id(sample_id, metadata)
            if object_id not in gt_by_object:
                current_split_mismatches.append(f"{sample_id}:missing_gt_object")
                continue
            gt_split, gt = gt_by_object[object_id]
            if split != gt_split:
                current_split_mismatches.append(f"{sample_id}:{split}!={gt_split}")
            variant = int(metadata.get("variant_index", sample_id.rsplit("__v", 1)[-1]))
            current_variants[object_id].append(variant)
            if split == "test":
                current_test_ids.append(sample_id)

            camera_intrinsic_errors.append(float(torch.max(torch.abs(
                sample["intrinsics"].double() - gt["intrinsics"].double()
            )).item()))
            camera_extrinsic_errors.append(float(torch.max(torch.abs(
                sample["extrinsics"].double() - gt["extrinsics"].double()
            )).item()))
            current_gt_vertex_errors.append(float(torch.max(torch.abs(
                sample["gt_vertices"].double() - gt["vertices"].double()
            )).item()))
            if not torch.equal(sample["gt_faces"], gt["faces"]):
                current_gt_face_mismatches.append(sample_id)
            if [str(path.resolve()) for path in _paths(sample)] != [
                str(path.resolve()) for path in _paths(gt)
            ]:
                current_image_path_mismatches.append(sample_id)
            current_image_missing.extend(
                str(path) for path in _paths(sample) if not path.is_file()
            )
            if torch.equal(sample["vertices"], gt["vertices"]):
                current_identical_to_gt += 1
            current_sizes.append(
                (int(sample["vertices"].shape[0]), int(sample["faces"].shape[0]))
            )
            current_initial_formula_errors.append(float(np.max(np.abs(
                sample["initial_laplacian"].cpu().numpy()
                - _uniform_raw(sample["vertices"], sample["faces"])
            ))))
            current_target_formula_errors.append(float(np.max(np.abs(
                sample["raw_laplacian_target"].cpu().numpy()
                - _uniform_raw(sample["target_positions"], sample["faces"])
            ))))

    gt_counts = {split: len(dataset) for split, dataset in gt_datasets.items()}
    current_counts = {split: len(dataset) for split, dataset in current_datasets.items()}
    expected_variants = list(range(5))
    variant_failures = {
        object_id: sorted(values)
        for object_id, values in current_variants.items()
        if sorted(values) != expected_variants
    }
    camera_ok = max(camera_intrinsic_errors, default=float("inf")) == 0.0 and max(
        camera_extrinsic_errors, default=float("inf")
    ) == 0.0
    passed = bool(
        gt_counts == {"train": 40, "validation": 5, "test": 5}
        and current_counts == {"train": 200, "validation": 25, "test": 25}
        and len(gt_by_object) == 50
        and not gt_identity_failures
        and not gt_initial_nonzero
        and max(gt_raw_errors, default=float("inf")) <= 1e-7
        and camera_ok
        and max(current_gt_vertex_errors, default=float("inf")) == 0.0
        and not current_gt_face_mismatches
        and not current_image_path_mismatches
        and not gt_image_missing
        and not current_image_missing
        and not current_split_mismatches
        and not variant_failures
        and len(current_test_ids) == 25
        and current_identical_to_gt == 0
        and max(current_initial_formula_errors, default=float("inf")) <= 1e-7
        and max(current_target_formula_errors, default=float("inf")) <= 1e-7
    )
    return {
        "contract_audit": passed,
        "gt_manifest": str(gt_manifest),
        "gt_manifest_sha256": _sha256(gt_manifest),
        "current_manifest": str(current_manifest),
        "current_manifest_sha256": _sha256(current_manifest),
        "manifest_roles": {
            "gt": gt_payload.get("dataset_role"),
            "current": current_payload.get("dataset_role"),
        },
        "split_counts": {"gt": gt_counts, "current": current_counts},
        "object_counts": {
            "gt": len(gt_by_object),
            "gt_by_split": {key: len(value) for key, value in gt_split_objects.items()},
        },
        "gt_training_contract": {
            "all_vertices_equal_gt_and_target_positions": not gt_identity_failures,
            "identity_failures": gt_identity_failures,
            "all_faces_equal_gt_faces": not gt_identity_failures,
            "all_initial_laplacian_exactly_zero": not gt_initial_nonzero,
            "initial_nonzero_samples": gt_initial_nonzero,
            "max_abs_delta_gt_raw_minus_uniform_L_gt_V_gt": max(gt_raw_errors),
            "target_formula_tolerance": 1e-7,
            "view_counts": sorted({len(_paths(sample)) for _, sample in gt_by_object.values()}),
            "prepared_image_sizes": sorted({int(sample["prepared_image_size"]) for _, sample in gt_by_object.values()}),
            "vertex_count_range": [min(x[0] for x in gt_sizes), max(x[0] for x in gt_sizes)],
            "face_count_range": [min(x[1] for x in gt_sizes), max(x[1] for x in gt_sizes)],
        },
        "gt_to_current_lineage": {
            "same_object_split": not current_split_mismatches,
            "split_mismatches": current_split_mismatches,
            "five_variants_per_object": not variant_failures,
            "variant_failures": variant_failures,
            "same_intrinsics_exact": max(camera_intrinsic_errors) == 0.0,
            "max_abs_intrinsics_difference": max(camera_intrinsic_errors),
            "same_extrinsics_exact": max(camera_extrinsic_errors) == 0.0,
            "max_abs_extrinsics_difference": max(camera_extrinsic_errors),
            "same_rgb_file_paths": not current_image_path_mismatches,
            "image_path_mismatches": current_image_path_mismatches,
            "missing_image_count": len(set(gt_image_missing + current_image_missing)),
            "embedded_gt_vertices_exact": max(current_gt_vertex_errors) == 0.0,
            "max_abs_embedded_gt_vertex_difference": max(current_gt_vertex_errors),
            "embedded_gt_faces_exact": not current_gt_face_mismatches,
            "current_meshes_identical_to_gt_count": current_identical_to_gt,
            "current_mesh_count": sum(current_counts.values()),
            "max_abs_initial_laplacian_minus_L_current_V_current": max(current_initial_formula_errors),
            "max_abs_stored_current_target_minus_L_current_P_proxy": max(current_target_formula_errors),
            "current_vertex_count_range": [min(x[0] for x in current_sizes), max(x[0] for x in current_sizes)],
            "current_face_count_range": [min(x[1] for x in current_sizes), max(x[1] for x in current_sizes)],
        },
        "zero_shot_test_sample_ids": sorted(current_test_ids),
        "coarse_raw_metric_policy": (
            "Do not use stored current targets for primary zero-shot prediction metrics; "
            "report only surface geometry on the 25 current test samples."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.gt_manifest.resolve(), args.current_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["contract_audit"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

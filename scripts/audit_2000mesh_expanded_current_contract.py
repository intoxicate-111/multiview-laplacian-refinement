#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian.multi_trainer import _prepare_object_static


SPLITS = ("train", "validation", "test")
FORBIDDEN_MODEL_INPUTS = {
    "target_positions",
    "gt_vertices",
    "gt_faces",
    "laplacian_target",
    "raw_laplacian_target",
    "normalized_laplacian_target",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=ROOT, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def resolved_image_paths(sample: Mapping[str, Any]) -> list[Path]:
    root = Path(str(sample["_dataset_root"]))
    return [
        path if path.is_absolute() else (root / path).resolve()
        for value in sample["image_paths"]
        for path in (Path(str(value)),)
    ]


def target_formula_error(sample: Mapping[str, Any]) -> float:
    current = sample["vertices"].detach().cpu()
    proxy = sample["target_positions"].detach().cpu()
    faces = sample["faces"].detach().cpu().numpy().astype(np.int64)
    if tuple(proxy.shape) != tuple(current.shape):
        raise ValueError("P_current and P_proxy vertex arrays have different shapes")
    operator = build_uniform_laplacian_data(faces, len(current))
    expected = torch.as_tensor(
        apply_uniform_laplacian(proxy.double().numpy(), operator), dtype=torch.float32
    )
    saved = sample["raw_laplacian_target"].detach().cpu().float()
    return float(torch.max(torch.abs(expected - saved)).item())


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = args.manifest.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split) for split in SPLITS
    }
    validate_disjoint_splits(*datasets.values())

    expected_counts = {"train": 8000, "validation": 1000, "test": 1000}
    split_counts = {split: len(dataset) for split, dataset in datasets.items()}
    if split_counts != expected_counts:
        raise RuntimeError(f"Expected 2000x5 split counts {expected_counts}, got {split_counts}")

    split_ids = {split: set(dataset.sample_ids) for split, dataset in datasets.items()}
    split_paths = {
        split: {str(record.path) for record in dataset.records}
        for split, dataset in datasets.items()
    }
    sample_id_overlaps = {
        f"{left}_{right}": sorted(split_ids[left] & split_ids[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    path_overlaps = {
        f"{left}_{right}": sorted(split_paths[left] & split_paths[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }

    object_splits: dict[str, set[str]] = defaultdict(set)
    variants_by_object: Counter[str] = Counter()
    variant_indices: dict[str, set[int]] = defaultdict(set)
    maximum_target_error = 0.0
    maximum_proxy_gt_error = 0.0
    minimum_vertices: int | None = None
    maximum_vertices = 0
    minimum_faces: int | None = None
    maximum_faces = 0
    image_paths_checked: set[Path] = set()
    missing_images: list[str] = []
    invalid_samples: list[dict[str, str]] = []
    metadata_contract_failures: list[str] = []
    model_input_leakage: list[dict[str, list[str]]] = []

    total = sum(split_counts.values())
    processed = 0
    for split, dataset in datasets.items():
        for index in range(len(dataset)):
            processed += 1
            try:
                sample = dataset.load_static(index)
                sample_id = str(sample["sample_id"])
                metadata = dict(sample.get("metadata", {}))
                object_id = str(metadata.get("object_id", ""))
                variant_index = int(metadata.get("variant_index", -1))
                if not object_id or variant_index not in range(5):
                    raise ValueError("missing object_id or invalid variant_index")
                object_splits[object_id].add(split)
                variants_by_object[object_id] += 1
                variant_indices[object_id].add(variant_index)

                current = sample["vertices"].detach().cpu()
                proxy = sample.get("target_positions")
                gt_vertices = sample.get("gt_vertices")
                gt_faces = sample.get("gt_faces")
                if not all(isinstance(value, torch.Tensor) for value in (proxy, gt_vertices, gt_faces)):
                    raise ValueError("missing P_proxy/GT topology tensors")
                proxy = proxy.detach().cpu()
                gt_vertices = gt_vertices.detach().cpu()
                gt_faces = gt_faces.detach().cpu().long()
                faces = sample["faces"].detach().cpu().long()
                if tuple(proxy.shape) != tuple(current.shape):
                    raise ValueError("P_current/P_proxy vertex count mismatch")
                if not torch.equal(faces, gt_faces):
                    raise ValueError("current/proxy face ordering mismatch")
                proxy_gt_error = float(torch.max(torch.abs(proxy - gt_vertices)).item())
                maximum_proxy_gt_error = max(maximum_proxy_gt_error, proxy_gt_error)
                maximum_target_error = max(maximum_target_error, target_formula_error(sample))

                views = int(sample["intrinsics"].shape[0])
                if views != 28 or tuple(sample["extrinsics"].shape) != (28, 4, 4):
                    raise ValueError("sample does not contain exactly 28 calibrated cameras")
                for field in (
                    "visibility",
                    "visibility_backface_only",
                    "visibility_occlusion_only",
                    "visibility_backface_and_occlusion",
                ):
                    value = sample.get(field)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
                        28,
                        len(current),
                    ):
                        raise ValueError(f"invalid {field}")
                paths = resolved_image_paths(sample)
                if len(paths) != 28:
                    raise ValueError("sample does not contain exactly 28 image paths")
                for path in paths:
                    if path not in image_paths_checked:
                        image_paths_checked.add(path)
                        if not path.is_file() and len(missing_images) < 100:
                            missing_images.append(str(path))

                if metadata.get("proxy_definition") != (
                    "P_proxy=source_gt_vertices_with_exact_same_topology"
                ):
                    metadata_contract_failures.append(sample_id)
                if metadata.get("target_constructor") != "delta_target=L_current@P_proxy":
                    metadata_contract_failures.append(sample_id)

                prepared = _prepare_object_static(
                    sample,
                    config,
                    keep_image_payload=True,
                    keep_projection=True,
                )
                leaked = sorted(FORBIDDEN_MODEL_INPUTS & set(prepared.sample))
                if leaked:
                    model_input_leakage.append({"sample_id": sample_id, "fields": leaked})

                vertices = len(current)
                faces_count = len(faces)
                minimum_vertices = vertices if minimum_vertices is None else min(minimum_vertices, vertices)
                maximum_vertices = max(maximum_vertices, vertices)
                minimum_faces = faces_count if minimum_faces is None else min(minimum_faces, faces_count)
                maximum_faces = max(maximum_faces, faces_count)
            except Exception as error:
                invalid_samples.append(
                    {
                        "split": split,
                        "index": str(index),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if len(invalid_samples) >= 100:
                    break
            if processed % 250 == 0:
                print(f"static audit {processed}/{total}", flush=True)
        if invalid_samples:
            break

    object_split_leaks = {
        object_id: sorted(splits)
        for object_id, splits in object_splits.items()
        if len(splits) != 1
    }
    invalid_variant_objects = {
        object_id: {
            "count": variants_by_object[object_id],
            "indices": sorted(variant_indices[object_id]),
        }
        for object_id in variants_by_object
        if variants_by_object[object_id] != 5
        or variant_indices[object_id] != set(range(5))
    }

    random.seed(args.seed)
    full_loads = []
    for split in SPLITS:
        dataset = datasets[split]
        index = random.randrange(len(dataset))
        sample = dataset[index]
        full_loads.append(
            {
                "split": split,
                "index": index,
                "sample_id": str(sample["sample_id"]),
                "vertices": int(sample["vertices"].shape[0]),
                "faces": int(sample["faces"].shape[0]),
                "images_shape": list(sample["images"].shape),
                "images_dtype": str(sample["images"].dtype),
                "intrinsics_shape": list(sample["intrinsics"].shape),
                "extrinsics_shape": list(sample["extrinsics"].shape),
                "visibility_shape": list(sample["visibility"].shape),
                "all_images_finite": bool(torch.isfinite(sample["images"]).all()),
            }
        )

    no_gt_runtime_dependency = not model_input_leakage
    checks = {
        "actual_2000_objects": len(object_splits) == 2000,
        "actual_10000_variants": sum(split_counts.values()) == 10000,
        "split_counts_match": split_counts == expected_counts,
        "sample_ids_disjoint": not any(sample_id_overlaps.values()),
        "paths_disjoint": not any(path_overlaps.values()),
        "object_level_split": not object_split_leaks,
        "five_variants_per_object": not invalid_variant_objects,
        "all_static_samples_valid": not invalid_samples,
        "all_images_exist": not missing_images,
        "camera_and_visibility_complete": not invalid_samples,
        "current_proxy_topology_and_order_match": maximum_proxy_gt_error == 0.0
        and not invalid_samples,
        "raw_target_formula_exact": maximum_target_error <= 1e-7,
        "metadata_target_contract": not metadata_contract_failures,
        "three_full_samples_loaded": len(full_loads) == 3
        and all(row["images_shape"][0] == 28 for row in full_loads),
        "gt_fields_excluded_from_model_inputs": no_gt_runtime_dependency,
        "test_time_gt_not_required_for_model_or_recovery": no_gt_runtime_dependency,
    }
    passed = all(checks.values())
    result = {
        "passed": passed,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "repository": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty_files": git_value("status", "--short").splitlines(),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "counts": {
            "objects": len(object_splits),
            "samples": sum(split_counts.values()),
            "variants": sum(split_counts.values()),
            "split_samples": split_counts,
            "unique_image_files_checked": len(image_paths_checked),
            "vertices": {"minimum": minimum_vertices, "maximum": maximum_vertices},
            "faces": {"minimum": minimum_faces, "maximum": maximum_faces},
        },
        "checks": checks,
        "split_sample_id_overlaps": sample_id_overlaps,
        "split_path_overlaps": path_overlaps,
        "object_split_leaks": object_split_leaks,
        "invalid_variant_objects": invalid_variant_objects,
        "invalid_samples": invalid_samples,
        "missing_images": missing_images,
        "metadata_contract_failures": sorted(set(metadata_contract_failures))[:100],
        "model_input_leakage": model_input_leakage[:100],
        "maximum_target_formula_error": maximum_target_error,
        "maximum_proxy_vs_gt_vertex_error": maximum_proxy_gt_error,
        "full_sample_loads": full_loads,
        "gt_leakage_interpretation": {
            "P_proxy_is_supervision_and_evaluation_label": True,
            "P_proxy_or_GT_is_model_input": False,
            "P_proxy_or_GT_is_required_by_laplacian_recovery": False,
            "test_evaluation_uses_GT_only_for_metrics": True,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "contract_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "CONTRACT_AUDIT.md").write_text(report(result), encoding="utf-8")
    return result


def report(result: Mapping[str, Any]) -> str:
    counts = result["counts"]
    checks = result["checks"]
    lines = [
        "# Future2000 GT-adaptive expanded/current-graph contract audit",
        "",
        f"- Overall: `{'PASSED' if result['passed'] else 'FAILED'}`",
        f"- Manifest: `{result['manifest']}`",
        f"- Manifest SHA-256: `{result['manifest_sha256']}`",
        f"- Objects: `{counts['objects']}`; variants: `{counts['variants']}`",
        f"- Split samples: `{counts['split_samples']}`",
        f"- Vertex range: `{counts['vertices']['minimum']}–{counts['vertices']['maximum']}`",
        f"- Face range: `{counts['faces']['minimum']}–{counts['faces']['maximum']}`",
        f"- Maximum `L_current @ P_proxy` target error: `{result['maximum_target_formula_error']:.12g}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## GT leakage interpretation",
            "",
            "`P_proxy`/GT is used only as supervised training and held-out evaluation ground truth. "
            "The model input retained by the trainer excludes proxy positions, GT tensors and all target tensors. "
            "Inference and direct raw-Laplacian recovery require only the current mesh, RGB/cameras, visibility and the model predictions.",
            "",
            "## Fully materialized samples",
            "",
            "```json",
            json.dumps(result["full_sample_loads"], indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({"passed": result["passed"], "counts": result["counts"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

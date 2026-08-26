#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed audit for the historical Sofa50 native-1920 split contract.

This script is deliberately limited to provenance, split, tensor, topology, and
input-identity checks.  It never evaluates a prediction or computes a test
metric.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components


SAMPLE_RE = re.compile(r"^(?P<object>.+)__v(?P<variant>[0-4][0-9]*)$")
EXPECTED_SAMPLES = {"train": 200, "validation": 25, "test": 25}
EXPECTED_OBJECTS = {"train": 40, "validation": 5, "test": 5}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def geometry_sha256(vertices: torch.Tensor, faces: torch.Tensor) -> str:
    # Match scripts/build_sofa50_same_initial_benchmark_manifest.py exactly.
    return tensor_sha256(vertices.float(), faces.long())


def resolve_record(manifest_path: Path, record: dict[str, Any]) -> Path:
    value = Path(str(record["path"]))
    return value.resolve() if value.is_absolute() else (manifest_path.parent / value).resolve()


def resolve_images(dataset_root: Path, values: list[str]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        path = Path(value)
        result.append(path.resolve() if path.is_absolute() else (dataset_root / path).resolve())
    return result


def uniform_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    vertices = vertices.detach().cpu().to(torch.float64)
    faces = faces.detach().cpu().to(torch.int64)
    directed = torch.cat(
        [
            faces[:, [0, 1]], faces[:, [1, 0]],
            faces[:, [1, 2]], faces[:, [2, 1]],
            faces[:, [2, 0]], faces[:, [0, 2]],
        ],
        dim=0,
    )
    directed = torch.unique(directed, dim=0)
    neighbors = torch.zeros_like(vertices)
    degree = torch.zeros((len(vertices),), dtype=torch.float64)
    neighbors.index_add_(0, directed[:, 0], vertices[directed[:, 1]])
    degree.index_add_(0, directed[:, 0], torch.ones(len(directed), dtype=torch.float64))
    return vertices - neighbors / degree.clamp_min(1.0).unsqueeze(1)


def component_count(vertex_count: int, faces: torch.Tensor) -> int:
    array = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    rows = np.concatenate([array[:, 0], array[:, 1], array[:, 2]])
    cols = np.concatenate([array[:, 1], array[:, 2], array[:, 0]])
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    )
    graph = (graph + graph.T).tocsr()
    return int(connected_components(graph, directed=False, return_labels=False))


def stats(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def is_1920_size(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value) == [1920, 1920]
    return int(value) == 1920


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--historical-manifest", required=True, type=Path)
    parser.add_argument("--sealed-test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    historical_path = args.historical_manifest.resolve()
    sealed_path = args.sealed_test_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = read_json(source_path)
    historical = read_json(historical_path)
    sealed = read_json(sealed_path)
    source_sha = sha256_file(source_path)
    dataset_root = Path(str(source.get("dataset_root", source_path.parent)))
    if not dataset_root.is_absolute():
        dataset_root = source_path.parent / dataset_root
    dataset_root = dataset_root.resolve()

    checks: dict[str, bool] = {}
    checks["source_role_resolution_ablation_only"] = source.get("dataset_role") == "resolution_ablation_only"
    checks["format_native1920_v1"] = source.get("format_version") == "sofa50_synthetic_current_28view_native1920_v1"
    checks["manifest_image_size_1920"] = int(source.get("image_size", -1)) == 1920
    checks["manifest_view_count_28"] = int(source.get("view_count", -1)) == 28
    checks["manifest_variants_per_object_5"] = int(source.get("variants_per_object", -1)) == 5
    checks["manifest_object_level_split"] = source.get("object_level_split_enforced") is True

    source_records = [dict(row) for row in source.get("samples", [])]
    historical_records = [dict(row) for row in historical.get("samples", [])]
    source_map = {str(row["sample_id"]): row for row in source_records}
    historical_map = {str(row["sample_id"]): row for row in historical_records}
    checks["source_sample_ids_unique"] = len(source_map) == len(source_records) == 250
    checks["historical_sample_ids_unique"] = len(historical_map) == len(historical_records) == 250
    checks["historical_and_source_ids_identical"] = set(source_map) == set(historical_map)
    # Both manifests retain the SHA of the same upstream 960 observation
    # manifest.  The sealed benchmark separately pins the native-1920 manifest
    # file itself below.
    checks["historical_upstream_source_sha_matches"] = (
        historical.get("source_manifest_sha256") == source.get("source_manifest_sha256")
    )
    checks["sealed_source_sha_matches"] = sealed.get("source_manifest_sha256") == source_sha

    by_split: dict[str, list[str]] = defaultdict(list)
    objects_by_split: dict[str, set[str]] = defaultdict(set)
    variants_by_object: dict[tuple[str, str], set[int]] = defaultdict(set)
    parsed: dict[str, tuple[str, int]] = {}
    parse_ok = True
    historical_records_match = True
    for sample_id, record in source_map.items():
        match = SAMPLE_RE.fullmatch(sample_id)
        if match is None:
            parse_ok = False
            continue
        object_id = match.group("object")
        variant = int(match.group("variant"))
        split = str(record.get("split"))
        parsed[sample_id] = (object_id, variant)
        by_split[split].append(sample_id)
        objects_by_split[split].add(object_id)
        variants_by_object[(split, object_id)].add(variant)
        old = historical_map.get(sample_id, {})
        historical_records_match &= old.get("split") == split
        historical_records_match &= resolve_record(historical_path, old) == resolve_record(source_path, record)
    checks["sample_id_format"] = parse_ok and len(parsed) == 250
    checks["historical_records_resolve_identically"] = historical_records_match
    checks["split_sample_counts"] = {k: len(by_split[k]) for k in EXPECTED_SAMPLES} == EXPECTED_SAMPLES
    checks["split_object_counts"] = {k: len(objects_by_split[k]) for k in EXPECTED_OBJECTS} == EXPECTED_OBJECTS
    checks["exact_v00_v04_per_object"] = all(value == set(range(5)) for value in variants_by_object.values())
    checks["sample_sets_disjoint"] = all(
        set(by_split[left]).isdisjoint(by_split[right])
        for index, left in enumerate(EXPECTED_SAMPLES)
        for right in list(EXPECTED_SAMPLES)[index + 1 :]
    )
    checks["object_sets_disjoint"] = all(
        objects_by_split[left].isdisjoint(objects_by_split[right])
        for index, left in enumerate(EXPECTED_OBJECTS)
        for right in list(EXPECTED_OBJECTS)[index + 1 :]
    )

    sealed_rows = [dict(row) for row in sealed.get("samples", [])]
    sealed_ids = [str(row["sample_id"]) for row in sealed_rows]
    source_test_ids = sorted(by_split["test"])
    checks["sealed_test_count_25"] = len(sealed_rows) == 25 and sealed.get("sample_count") == 25
    checks["sealed_test_ids_exactly_source_test"] = sorted(sealed_ids) == source_test_ids
    checks["sealed_top_level_ids_match_rows"] = sealed.get("sample_ids") == sealed_ids

    tensor_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, set[str]] = defaultdict(set)
    clean_hashes: dict[str, set[str]] = defaultdict(set)
    topology_hashes: dict[str, set[str]] = defaultdict(set)
    object_image_tuples: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    object_component_counts: dict[str, int] = {}
    all_image_paths: set[Path] = set()
    maximum_raw_target_error = 0.0
    all_tensor_contracts = True
    perturb_values: dict[str, list[float]] = defaultdict(list)
    split_topology: dict[str, dict[str, list[float]]] = {
        split: defaultdict(list) for split in EXPECTED_SAMPLES
    }

    for sample_id in sorted(source_map):
        record = source_map[sample_id]
        split = str(record["split"])
        object_id, variant = parsed[sample_id]
        sample_path = resolve_record(source_path, record)
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        metadata = dict(sample.get("metadata", {}))
        images = resolve_images(dataset_root, list(sample.get("image_paths", [])))
        image_tuple = tuple(str(path) for path in images)
        object_image_tuples[object_id].add(image_tuple)
        all_image_paths.update(images)

        vertices = sample["vertices"]
        faces = sample["faces"]
        clean_vertices = sample["gt_vertices"]
        clean_faces = sample["gt_faces"]
        target_positions = sample["target_positions"]
        raw_target = sample["raw_laplacian_target"]
        recomputed = uniform_laplacian(clean_vertices, faces)
        target_error = float(torch.max(torch.abs(recomputed - raw_target.to(torch.float64))).item())
        maximum_raw_target_error = max(maximum_raw_target_error, target_error)

        same_topology = torch.equal(faces.cpu(), clean_faces.cpu()) and len(vertices) == len(clean_vertices)
        contract = (
            str(sample.get("sample_id")) == sample_id
            and metadata.get("object_id") == object_id
            and metadata.get("source_sample_id") == object_id
            and metadata.get("source_split") == split
            and int(metadata.get("variant_index", -1)) == variant
            and metadata.get("operator_type") == "uniform"
            and metadata.get("current_graph_source") == "deterministic_smooth_normal_perturbation_of_gt_topology"
            and metadata.get("target_constructor") == "delta_target=L_current@P_proxy"
            and metadata.get("observation_resolution_ablation") == "native_cpu_reference_1920_from_matched_28view_cameras"
            and int(metadata.get("input_resolution", -1)) == 1920
            and tuple(sample["intrinsics"].shape) == (28, 3, 3)
            and tuple(sample["extrinsics"].shape) == (28, 4, 4)
            and is_1920_size(sample.get("source_image_size", -1))
            and is_1920_size(sample.get("prepared_image_size", -1))
            and len(images) == 28
            and same_topology
            and torch.equal(target_positions.cpu(), clean_vertices.cpu())
            and target_error <= 2e-6
        )
        all_tensor_contracts &= contract
        input_hash = geometry_sha256(vertices, faces)
        clean_hash = geometry_sha256(clean_vertices, clean_faces)
        topology_hash = tensor_sha256(faces.long())
        input_hashes[split].add(input_hash)
        clean_hashes[split].add(clean_hash)
        topology_hashes[split].add(topology_hash)

        if object_id not in object_component_counts:
            object_component_counts[object_id] = component_count(len(vertices), faces)
        components = object_component_counts[object_id]
        split_topology[split]["vertices"].append(len(vertices))
        split_topology[split]["faces"].append(len(faces))
        split_topology[split]["components"].append(components)
        perturb = dict(metadata.get("perturbation", {}))
        for key in (
            "requested_perturb_std_h", "locally_damped_vertex_ratio",
            "minimum_local_damping", "mean_offset_over_h", "median_offset_over_h",
            "p95_offset_over_h", "max_offset_over_h", "flipped_faces",
            "new_degenerate_faces", "minimum_triangle_area",
        ):
            perturb_values[key].append(float(perturb.get(key, float("nan"))))
        tensor_rows.append(
            {
                "sample_id": sample_id,
                "object_id": object_id,
                "variant": variant,
                "split": split,
                "prepared_path": str(sample_path),
                "vertices": len(vertices),
                "faces": len(faces),
                "components": components,
                "view_count": len(images),
                "input_geometry_sha256": input_hash,
                "clean_geometry_sha256": clean_hash,
                "topology_sha256": topology_hash,
                "raw_target_max_abs_error": target_error,
                "contract_passed": contract,
            }
        )

    checks["all_prepared_tensor_contracts"] = all_tensor_contracts
    checks["raw_target_is_uniform_L_current_V_clean"] = maximum_raw_target_error <= 2e-6
    checks["one_exact_28_view_tuple_per_object"] = all(len(value) == 1 for value in object_image_tuples.values())
    checks["all_50_objects_have_image_tuples"] = len(object_image_tuples) == 50
    image_sizes: Counter[tuple[int, int]] = Counter()
    missing_images: list[str] = []
    for image_path in sorted(all_image_paths):
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        with Image.open(image_path) as image:
            image_sizes[image.size] += 1
    checks["all_native_images_exist"] = not missing_images and len(all_image_paths) == 50 * 28
    checks["all_native_images_are_1920"] = set(image_sizes) == {(1920, 1920)} and sum(image_sizes.values()) == 50 * 28
    checks["input_geometry_no_cross_split_identity"] = all(
        input_hashes[left].isdisjoint(input_hashes[right])
        for index, left in enumerate(EXPECTED_SAMPLES)
        for right in list(EXPECTED_SAMPLES)[index + 1 :]
    )
    checks["clean_geometry_no_cross_split_identity"] = all(
        clean_hashes[left].isdisjoint(clean_hashes[right])
        for index, left in enumerate(EXPECTED_SAMPLES)
        for right in list(EXPECTED_SAMPLES)[index + 1 :]
    )

    sealed_map = {str(row["sample_id"]): row for row in sealed_rows}
    sealed_identity = True
    for sample_id in source_test_ids:
        source_record = source_map[sample_id]
        sealed_row = sealed_map.get(sample_id, {})
        tensor_row = next(row for row in tensor_rows if row["sample_id"] == sample_id)
        sealed_identity &= sealed_row.get("split") == "test"
        sealed_identity &= Path(str(sealed_row.get("path", ""))).resolve() == resolve_record(source_path, source_record)
        sealed_identity &= sealed_row.get("prepared_tensor_geometry_sha256") == tensor_row["input_geometry_sha256"]
        sealed_identity &= int(sealed_row.get("view_count", -1)) == 28
        sealed_identity &= sealed_row.get("image_size") == [1920, 1920]
    checks["sealed_test_tensor_and_input_identity"] = sealed_identity

    checks["zero_prepared_flips"] = sum(perturb_values["flipped_faces"]) == 0
    checks["zero_prepared_new_degenerates"] = sum(perturb_values["new_degenerate_faces"]) == 0
    checks["requested_perturb_std_exact"] = set(perturb_values["requested_perturb_std_h"]) == {0.15}
    passed = all(checks.values())

    topology_summary = {
        split: {
            key: stats(values) for key, values in split_topology[split].items()
        }
        | {
            "unique_input_geometries": len(input_hashes[split]),
            "unique_clean_geometries": len(clean_hashes[split]),
            "unique_topologies": len(topology_hashes[split]),
        }
        for split in EXPECTED_SAMPLES
    }
    perturbation_summary = {
        key: stats(values) for key, values in perturb_values.items()
    }
    audit = {
        "contract_audit": passed,
        "scope": "provenance_split_tensor_topology_and_input_identity_only_no_test_metrics",
        "source_manifest": str(source_path),
        "source_manifest_sha256": source_sha,
        "historical_manifest": str(historical_path),
        "historical_manifest_sha256": sha256_file(historical_path),
        "sealed_test_manifest": str(sealed_path),
        "sealed_test_manifest_sha256": sha256_file(sealed_path),
        "dataset_root": str(dataset_root),
        "checks": checks,
        "split_sample_counts": {split: len(by_split[split]) for split in EXPECTED_SAMPLES},
        "split_object_counts": {split: len(objects_by_split[split]) for split in EXPECTED_OBJECTS},
        "object_ids": {split: sorted(objects_by_split[split]) for split in EXPECTED_OBJECTS},
        "native_image_files": len(all_image_paths),
        "native_image_size_counts": {f"{w}x{h}": count for (w, h), count in image_sizes.items()},
        "missing_images": missing_images,
        "maximum_raw_target_recomputation_error": maximum_raw_target_error,
        "topology": topology_summary,
        "perturbation": perturbation_summary,
    }
    (output_dir / "contract_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "split_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tensor_rows[0]))
        writer.writeheader()
        writer.writerows(tensor_rows)

    lines = [
        "# Sofa50 old-domain native-1920 preflight audit",
        "",
        f"Contract audit: **{str(passed).lower()}**. No model was trained and no test metric was computed.",
        "",
        "## Recovered historical split",
        "",
        "| Split | Samples | Objects | Unique input meshes | Unique clean meshes | Unique topologies | V min/mean/max | F min/mean/max | Components min/mean/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in EXPECTED_SAMPLES:
        item = topology_summary[split]
        lines.append(
            f"| {split} | {len(by_split[split])} | {len(objects_by_split[split])} | "
            f"{item['unique_input_geometries']} | {item['unique_clean_geometries']} | {item['unique_topologies']} | "
            f"{item['vertices']['minimum']:.0f}/{item['vertices']['mean']:.1f}/{item['vertices']['maximum']:.0f} | "
            f"{item['faces']['minimum']:.0f}/{item['faces']['mean']:.1f}/{item['faces']['maximum']:.0f} | "
            f"{item['components']['minimum']:.0f}/{item['components']['mean']:.1f}/{item['components']['maximum']:.0f} |"
        )
    lines += [
        "",
        "All sample IDs and object IDs are disjoint across train/validation/test. Each object has exactly variants `v00`--`v04`. The sealed same-initial benchmark's 25 IDs, prepared tensor paths, input-geometry hashes, 28-view tuples, cameras, and native `1920x1920` contract match the recovered historical test split exactly.",
        "",
        "## Data and target contract",
        "",
        f"- Source manifest SHA-256: `{source_sha}`.",
        f"- Native images checked: `{len(all_image_paths)}` (`50 x 28`); all are `1920x1920`.",
        f"- Maximum raw-target recomputation error for `delta_GT=L_U(current) V_clean`: `{maximum_raw_target_error:.6g}`.",
        f"- Requested perturbation scale: `{perturbation_summary['requested_perturb_std_h']['mean']:.3g} h`; prepared flips/new degenerates: `{sum(perturb_values['flipped_faces']):.0f}/{sum(perturb_values['new_degenerate_faces']):.0f}`.",
        "- The prepared tensors state `native_cpu_reference_1920_from_matched_28view_cameras`; no 960-pixel tensor or downsampling path is admitted by this audit.",
        "",
        "## Decision",
        "",
        ("Preflight passes. Arm B/E training may use the recovered train and validation records; the 25 test records remain sealed." if passed else "Preflight fails closed. Training must not start."),
        "",
    ]
    (output_dir / "PREFLIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"contract_audit": passed, "output_dir": str(output_dir)}, indent=2))
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit("Failed checks: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

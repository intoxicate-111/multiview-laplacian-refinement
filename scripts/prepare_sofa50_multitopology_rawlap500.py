#!/usr/bin/env python3
from __future__ import annotations

"""Build and audit versioned Sofa50 multi-topology raw-Laplacian datasets.

Each sample stores a corrupted input mesh and a clean reference with exactly the
same faces and vertex ordering.  The saved target is the native raw uniform
Laplacian of that clean reference.  No proxy, correspondence transfer, or h^2
target normalization participates in target construction.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.data import Mesh
from mlr.learned_laplacian.multitopology_rawlap import (
    DEFAULT_SMOOTHING_PROFILE,
    DEGRADATION_PROFILES,
    LEGACY_SMOOTHING_PROFILE,
    STRONG_SMOOTHING_PROFILE,
    UNSEEN_VARIANT_NAMES,
    VARIANT_NAMES,
    construct_clean_topology,
    corrupt_clean_reference,
    degradation_for_variant,
    raw_uniform_laplacian,
    triangle_areas,
)


LEGACY_FORMAT_VERSION = "Sofa50MultiTopologyRawLap500_v1"
LEGACY_UNSEEN_FORMAT_VERSION = "Sofa50MultiTopologyRawLapUnseen25_v1"
FORMAT_VERSION = "Sofa50MultiTopologyRawLap500_v2"
UNSEEN_FORMAT_VERSION = "Sofa50MultiTopologyRawLapUnseen25_v2"
SEED_NAMESPACE = LEGACY_FORMAT_VERSION
PROFILE_DATASET_FAMILIES = {
    LEGACY_SMOOTHING_PROFILE: {
        "training": LEGACY_FORMAT_VERSION,
        "unseen": LEGACY_UNSEEN_FORMAT_VERSION,
    },
    STRONG_SMOOTHING_PROFILE: {
        "training": FORMAT_VERSION,
        "unseen": UNSEEN_FORMAT_VERSION,
    },
}
MINIMUM_V2_ATTENUATION = {"mild": 0.50, "strong": 0.75, "unseen_intermediate": 0.65}
OBJECT_SPLITS = {"train": 40, "validation": 5, "test": 5}
SAMPLE_SPLITS = {"train": 400, "validation": 50, "test": 50}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_record(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def stable_seed(
    object_id: str,
    variant: str,
    base_seed: int,
    *,
    namespace: str = SEED_NAMESPACE,
) -> int:
    # Keep the v1 seed namespace so strong_smooth_v2 is a true smoothing-only
    # data ablation with identical perturbation fields.
    digest = hashlib.sha256(
        f"{namespace}|{base_seed}|{object_id}|{variant}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def dataset_family_for(smoothing_profile: str, dataset_mode: str) -> str:
    try:
        return PROFILE_DATASET_FAMILIES[smoothing_profile][dataset_mode]
    except KeyError as error:
        raise ValueError(
            f"Unsupported smoothing profile / dataset mode: {smoothing_profile} / {dataset_mode}"
        ) from error


def resolved_output_root(args: argparse.Namespace, dataset_family: str) -> Path:
    value = args.output_root if args.output_root is not None else Path.home() / dataset_family
    return value.expanduser().resolve()


def source_records(manifest_path: Path) -> list[dict[str, Any]]:
    records = read_json(manifest_path).get("samples")
    if not isinstance(records, list) or len(records) != 50:
        raise ValueError(f"Expected 50 Sofa50 source objects, found {len(records or [])}.")
    counts = Counter(str(row.get("split")) for row in records)
    if dict(counts) != OBJECT_SPLITS:
        raise ValueError(f"Unexpected source object split: {dict(counts)}")
    return [dict(row) for row in records]


def make_cuda_visibility(upstream_root: Path) -> Any:
    module = load_script(
        upstream_root / "src" / "sofa50_refinement" / "gpu_visibility.py",
        "sofa50_multitopology_gpu_visibility",
    )
    compute = module.compute_renderer_visibility_cuda
    plugin_loaded = False

    def load_cached_plugin() -> None:
        nonlocal plugin_loaded
        if plugin_loaded or shutil.which("ninja") is not None:
            plugin_loaded = True
            return
        import nvdiffrast.torch.ops as ops

        cache = getattr(ops, "_cached_plugin", None)
        if isinstance(cache, dict) and cache.get(False) is not None:
            plugin_loaded = True
            return
        candidates = sorted(
            Path.home().glob(".cache/torch_extensions/*/nvdiffrast_plugin/nvdiffrast_plugin.so"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("No Ninja executable or cached nvdiffrast plugin is available.")
        spec = importlib.util.spec_from_file_location("nvdiffrast_plugin", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(candidates[0])
        compiled = importlib.util.module_from_spec(spec)
        sys.modules["nvdiffrast_plugin"] = compiled
        spec.loader.exec_module(compiled)
        cache[False] = compiled
        plugin_loaded = True

    def wrapped(mesh: Mesh, cameras: list[Any], config: Any, *, neighborhood_radius: int = 1) -> Any:
        load_cached_plugin()
        return compute(
            mesh,
            cameras,
            image_size=int(config.width),
            neighborhood_radius=neighborhood_radius,
            front_face_winding=str(config.front_face_winding),
        )

    return wrapped


def resolved_image_paths(source: Mapping[str, Any], source_root: Path, output_root: Path) -> list[str]:
    values = source.get("image_paths")
    if not isinstance(values, list) or len(values) != 28:
        raise ValueError("Every source object must provide exactly 28 lazy RGB paths.")
    output = []
    for value in values:
        path = Path(str(value))
        path = path.resolve() if path.is_absolute() else (source_root / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(os.path.relpath(path, output_root))
    return output


def topology_invalidity(clean: Mesh, corrupted: Mesh) -> dict[str, Any]:
    clean_triangles = clean.vertices[clean.faces]
    input_triangles = corrupted.vertices[corrupted.faces]
    clean_cross = np.cross(
        clean_triangles[:, 1] - clean_triangles[:, 0],
        clean_triangles[:, 2] - clean_triangles[:, 0],
    )
    input_cross = np.cross(
        input_triangles[:, 1] - input_triangles[:, 0],
        input_triangles[:, 2] - input_triangles[:, 0],
    )
    clean_norm = np.linalg.norm(clean_cross, axis=1)
    input_norm = np.linalg.norm(input_cross, axis=1)
    flipped = np.einsum("ij,ij->i", clean_cross, input_cross) < 0
    return {
        "clean_degenerate_faces": int((clean_norm <= 1e-14).sum()),
        "input_degenerate_faces": int((input_norm <= 1e-14).sum()),
        "introduced_flipped_faces": int(flipped.sum()),
        "introduced_flipped_fraction": float(flipped.mean()) if len(flipped) else 0.0,
        "minimum_input_double_area": float(input_norm.min(initial=np.inf)),
    }


def build_sample(
    source: Mapping[str, Any],
    *,
    object_id: str,
    split: str,
    variant: str,
    base_seed: int,
    source_root: Path,
    output_root: Path,
    visibility_artifact: Path,
    current_helper: Any,
    deps: Mapping[str, Any],
    max_vertices: int,
    visibility_backend: str,
    dataset_family: str,
    smoothing_profile: str,
    seed_namespace: str = SEED_NAMESPACE,
    source_dataset_label: str = "Sofa",
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_vertices = source["gt_vertices"].detach().cpu().double().numpy()
    source_faces = source["gt_faces"].detach().cpu().numpy().astype(np.int64)
    clean, topology = construct_clean_topology(
        source_vertices, source_faces, variant, max_vertices=max_vertices
    )
    seed = stable_seed(
        object_id, variant, base_seed, namespace=seed_namespace
    )
    degradation = degradation_for_variant(variant, smoothing_profile)
    corrupted, corruption = corrupt_clean_reference(clean, degradation, seed=seed)
    target_np = raw_uniform_laplacian(clean)
    initial_np = raw_uniform_laplacian(corrupted)

    clean_vertices = torch.as_tensor(clean.vertices, dtype=torch.float32)
    faces = torch.as_tensor(clean.faces, dtype=torch.long)
    input_vertices = torch.as_tensor(corrupted.vertices, dtype=torch.float32)
    target = torch.as_tensor(target_np, dtype=torch.float32)
    center = 0.5 * (input_vertices.amin(dim=0) + input_vertices.amax(dim=0))
    position_scale = torch.linalg.vector_norm(input_vertices - center, dim=-1).amax()
    if not torch.isfinite(position_scale) or float(position_scale) <= 1e-12:
        raise RuntimeError("Invalid position-normalization scale.")
    sample_id = f"{object_id}__{variant}"
    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "image_paths": resolved_image_paths(source, source_root, output_root),
        "prepared_storage_format": "lazy_image_paths_v1",
        "source_image_size": list(source.get("source_image_size", [960, 960])),
        "prepared_image_size": int(source.get("prepared_image_size", 960)),
        "intrinsics": source["intrinsics"].detach().cpu().float().clone(),
        "extrinsics": source["extrinsics"].detach().cpu().float().clone(),
        "vertices": input_vertices,
        "faces": faces,
        "vertex_normals": torch.as_tensor(corrupted.normals, dtype=torch.float32),
        "initial_laplacian": torch.as_tensor(initial_np, dtype=torch.float32),
        "laplacian_target": target,
        "raw_laplacian_target": target.clone(),
        "target_confidence": torch.ones(clean.num_vertices, dtype=torch.float32),
        "target_positions": clean_vertices,
        "gt_vertices": clean_vertices,
        "gt_faces": faces.clone(),
        "clean_reference_vertices": clean_vertices.clone(),
        "clean_reference_faces": faces.clone(),
        "position_normalization_center": center,
        "position_normalization_scale": position_scale,
        "metadata": {
            "dataset_family": dataset_family,
            "dataset_role": "clean_topology_raw_laplacian_restoration",
            "training_eligible": True,
            "object_id": object_id,
            "source_sample_id": str(source["sample_id"]),
            "source_split": split,
            "variant": variant,
            "smoothing_profile": smoothing_profile,
            "family": topology["family"],
            "variant_seed": seed,
            "view_count": 28,
            "input_resolution": 960,
            "clean_reference_definition": (
                f"native topology constructed from source {source_dataset_label} GT"
            ),
            "input_constructor": "clean_reference -> perturb -> smooth",
            "target_constructor": "delta_target_raw=L(clean_reference_faces)@clean_reference_vertices",
            "target_mode": "raw_laplacian",
            "target_scaling_applied": False,
            "proxy_used": False,
            "nearest_surface_correspondence_used": False,
            "target_transfer_used": False,
            "operator_type": "uniform",
            "topology": topology,
            "corruption": corruption,
        },
    }
    visibility = current_helper._attach_visibility(
        sample,
        corrupted,
        sample,
        backend=visibility_backend,
        artifact_path=visibility_artifact,
        deps=deps,
    )
    sample["metadata"]["renderer_visibility"] = visibility

    # Validate the runtime schema, but intentionally do not persist the derived
    # normalized target that validate_sample creates for backward compatibility.
    validated = deps["validate_sample"](sample)
    validated.pop("normalized_laplacian_target", None)
    recomputed = torch.as_tensor(raw_uniform_laplacian(clean), dtype=torch.float32)
    target_error = float(torch.max(torch.abs(recomputed - target)))
    input_target_delta = float(torch.linalg.vector_norm(target - sample["initial_laplacian"], dim=1).mean())
    invalidity = topology_invalidity(clean, corrupted)
    audit = {
        "sample_id": sample_id,
        "object_id": object_id,
        "split": split,
        "variant": variant,
        "dataset_family": dataset_family,
        "smoothing_profile": smoothing_profile,
        "seed_namespace": seed_namespace,
        "family": topology["family"],
        "seed": seed,
        "original_vertices": topology["original_vertices"],
        "original_faces": topology["original_faces"],
        "clean_vertices": clean.num_vertices,
        "clean_faces": clean.num_faces,
        "vertex_ratio_vs_gt": topology["vertex_ratio_vs_gt"],
        "face_ratio_vs_gt": topology["face_ratio_vs_gt"],
        "tau_area_normalized": topology["tau_area_normalized"],
        "tau_edge_normalized": topology["tau_edge_normalized"],
        "edge_only_selected_faces_total": topology["edge_only_selected_faces_total"],
        "clean_input_vertex_count_equal": clean.num_vertices == corrupted.num_vertices,
        "clean_input_face_count_equal": clean.num_faces == corrupted.num_faces,
        "clean_input_faces_exact": bool(np.array_equal(clean.faces, corrupted.faces)),
        "saved_gt_faces_exact": bool(torch.equal(faces, validated["gt_faces"])),
        "target_shape_matches": tuple(target.shape) == (clean.num_vertices, 3),
        "target_recompute_max_abs_error": target_error,
        "target_recompute_exact_float32": bool(torch.equal(recomputed, target)),
        "input_target_mean_vector_difference": input_target_delta,
        "target_differs_from_input_laplacian": input_target_delta > 1e-9,
        "all_tensors_finite": bool(
            torch.isfinite(input_vertices).all()
            and torch.isfinite(clean_vertices).all()
            and torch.isfinite(target).all()
            and torch.isfinite(sample["initial_laplacian"]).all()
        ),
        "native_expanded_topology_target": bool(
            topology["family"] in {"original", "uniform_midpoint", "area", "area_or_edge"}
            and target_error == 0.0
        ),
        "clean_to_input_displacement_mean": corruption["clean_to_input_displacement_mean"],
        "clean_to_input_displacement_median": corruption["clean_to_input_displacement_median"],
        "clean_to_input_displacement_max": corruption["clean_to_input_displacement_max"],
        "clean_to_input_displacement_p95": corruption["clean_to_input_displacement_p95"],
        "clean_to_input_displacement_mean_over_bbox_diagonal": corruption[
            "clean_to_input_displacement_mean_over_bbox_diagonal"
        ],
        "clean_to_input_displacement_p95_over_bbox_diagonal": corruption[
            "clean_to_input_displacement_p95_over_bbox_diagonal"
        ],
        "perturb_to_input_smoothing_displacement_mean": corruption[
            "perturb_to_input_smoothing_displacement_mean"
        ],
        "perturb_to_input_smoothing_displacement_mean_over_bbox_diagonal": corruption[
            "perturb_to_input_smoothing_displacement_mean_over_bbox_diagonal"
        ],
        "smoothing_iterations": degradation.mesh_smoothing_iterations,
        "smoothing_strength": degradation.mesh_smoothing_strength,
        "smoothing_high_frequency_attenuation_proxy": corruption[
            "mesh_smoothing_high_frequency_attenuation_proxy"
        ],
        "raw_laplacian_magnitude_mean": float(np.linalg.norm(target_np, axis=1).mean()),
        "raw_laplacian_magnitude_median": float(np.median(np.linalg.norm(target_np, axis=1))),
        "raw_laplacian_magnitude_max": float(np.linalg.norm(target_np, axis=1).max(initial=0.0)),
        **invalidity,
    }
    minimum_attenuation = MINIMUM_V2_ATTENUATION.get(degradation.name, 0.0)
    audit["strong_smoothing_budget_pass"] = bool(
        smoothing_profile != STRONG_SMOOTHING_PROFILE
        or audit["smoothing_high_frequency_attenuation_proxy"] >= minimum_attenuation
    )
    audit["contract_pass"] = bool(
        audit["clean_input_vertex_count_equal"]
        and audit["clean_input_face_count_equal"]
        and audit["clean_input_faces_exact"]
        and audit["saved_gt_faces_exact"]
        and audit["target_shape_matches"]
        and audit["target_recompute_exact_float32"]
        and audit["target_differs_from_input_laplacian"]
        and audit["all_tensors_finite"]
        and audit["native_expanded_topology_target"]
        and audit["clean_degenerate_faces"] == 0
        and audit["input_degenerate_faces"] == 0
        and audit["strong_smoothing_budget_pass"]
    )
    return validated, audit


def generate(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.expanduser().resolve()
    dataset_family = dataset_family_for(args.smoothing_profile, args.dataset_mode)
    output_root = resolved_output_root(args, dataset_family)
    upstream_root = args.upstream_root.expanduser().resolve()
    downstream_root = args.downstream_root.expanduser().resolve()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count).")
    output_root.mkdir(parents=True, exist_ok=True)
    current_helper = load_script(
        upstream_root / "scripts" / "prepare_sofa50_synthetic_current.py",
        f"multitopology_current_helper_{args.shard_index}",
    )
    current_helper.VIEW_COUNT = 28
    deps = current_helper._dependencies(downstream_root)
    if args.visibility_backend == "cuda":
        deps["compute_renderer_visibility"] = make_cuda_visibility(upstream_root)
    records = source_records(source_manifest)
    variants = VARIANT_NAMES
    if args.dataset_mode == "unseen":
        records = [row for row in records if str(row["split"]) == "test"]
        variants = UNSEEN_VARIANT_NAMES
    selected = [row for index, row in enumerate(records) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    output_records: list[dict[str, str]] = []
    audits: list[dict[str, Any]] = []
    for object_index, record in enumerate(selected, start=1):
        object_id = str(record["sample_id"])
        split = str(record["split"])
        source = torch.load(resolve_record(source_manifest, record), map_location="cpu", weights_only=False)
        if int(source["intrinsics"].shape[0]) != 28 or int(source.get("prepared_image_size", 0)) != 960:
            raise ValueError(f"{object_id}: expected the established 28-view 960 source.")
        print(f"[{object_index}/{len(selected)}] {object_id} split={split}", flush=True)
        for variant in variants:
            relative = Path("prepared") / split / object_id / f"{variant}.pt"
            sample_path = output_root / relative
            visibility_path = output_root / "renderer_visibility" / split / object_id / f"{variant}.npz"
            audit_path = output_root / "sample_audits" / split / object_id / f"{variant}.json"
            if args.resume and sample_path.is_file() and visibility_path.is_file() and audit_path.is_file():
                audit = read_json(audit_path)
                existing_profile = str(
                    audit.get("smoothing_profile", LEGACY_SMOOTHING_PROFILE)
                )
                if existing_profile != args.smoothing_profile:
                    raise RuntimeError(
                        f"{object_id}__{variant}: existing smoothing profile {existing_profile!r} "
                        f"does not match requested {args.smoothing_profile!r}."
                    )
                sample = torch.load(sample_path, map_location="cpu", weights_only=False)
                deps["validate_sample"](sample)
            else:
                sample, audit = build_sample(
                    source,
                    object_id=object_id,
                    split=split,
                    variant=variant,
                    base_seed=args.seed,
                    source_root=source_manifest.parent,
                    output_root=output_root,
                    visibility_artifact=visibility_path,
                    current_helper=current_helper,
                    deps=deps,
                    max_vertices=args.max_vertices,
                    visibility_backend=args.visibility_backend,
                    dataset_family=dataset_family,
                    smoothing_profile=args.smoothing_profile,
                )
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_sample = {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()
                    if key not in {"_static_prepared", "_dataset_root", "edge_index", "vertex_degree", "normalized_laplacian_target"}
                }
                torch.save(cpu_sample, sample_path)
                write_json(audit_path, audit)
            if not audit["contract_pass"]:
                raise RuntimeError(f"Contract audit failed for {audit['sample_id']}")
            output_records.append(
                {"sample_id": str(sample["sample_id"]), "split": split, "path": relative.as_posix()}
            )
            audits.append(audit)
            print(
                f"  {variant}: V={audit['clean_vertices']} F={audit['clean_faces']} "
                f"disp={audit['clean_to_input_displacement_mean']:.6g}",
                flush=True,
            )
    shard_root = output_root / "shards"
    write_json(
        shard_root / f"manifest_shard_{args.shard_index:02d}.json",
        {"format_version": dataset_family, "samples": output_records},
    )
    write_json(
        shard_root / f"audit_shard_{args.shard_index:02d}.json",
        {"format_version": dataset_family, "samples": audits},
    )


def aggregate_stats_for_variants(
    rows: list[dict[str, Any]], variants: tuple[str, ...]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in variants:
        values = [row for row in rows if row["variant"] == variant]
        vertices = np.asarray([row["clean_vertices"] for row in values], dtype=np.float64)
        faces = np.asarray([row["clean_faces"] for row in values], dtype=np.float64)
        ratios = np.asarray([row["face_ratio_vs_gt"] for row in values], dtype=np.float64)
        displacement = np.asarray([row["clean_to_input_displacement_mean"] for row in values])
        normalized_displacement = np.asarray(
            [
                row["clean_to_input_displacement_mean_over_bbox_diagonal"]
                for row in values
                if "clean_to_input_displacement_mean_over_bbox_diagonal" in row
            ]
        )
        lap = np.asarray([row["raw_laplacian_magnitude_mean"] for row in values])
        output[variant] = {
            "sample_count": len(values),
            "vertices": {"mean": float(vertices.mean()), "median": float(np.median(vertices)), "min": int(vertices.min()), "max": int(vertices.max())},
            "faces": {"mean": float(faces.mean()), "median": float(np.median(faces)), "min": int(faces.min()), "max": int(faces.max())},
            "face_subdivision_ratio_vs_gt_mean": float(ratios.mean()),
            "clean_to_input_displacement_mean": float(displacement.mean()),
            "clean_to_input_displacement_mean_over_bbox_diagonal": (
                float(normalized_displacement.mean())
                if len(normalized_displacement)
                else None
            ),
            "smoothing_iterations": sorted(set(row["smoothing_iterations"] for row in values)),
            "smoothing_strength": sorted(set(row["smoothing_strength"] for row in values)),
            "smoothing_high_frequency_attenuation_proxy": sorted(
                set(
                    row.get(
                        "smoothing_high_frequency_attenuation_proxy",
                        1.0
                        - (1.0 - float(row["smoothing_strength"]))
                        ** int(row["smoothing_iterations"]),
                    )
                    for row in values
                )
            ),
            "raw_laplacian_magnitude_mean": float(lap.mean()),
            "edge_only_selected_faces_total": int(sum(row["edge_only_selected_faces_total"] for row in values)),
        }
    return output


def aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_stats_for_variants(rows, VARIANT_NAMES)


def merge(args: argparse.Namespace) -> None:
    dataset_family = dataset_family_for(args.smoothing_profile, args.dataset_mode)
    output_root = resolved_output_root(args, dataset_family)
    variants = VARIANT_NAMES if args.dataset_mode == "training" else UNSEEN_VARIANT_NAMES
    expected_samples = 500 if args.dataset_mode == "training" else 25
    records: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for index in range(args.shard_count):
        records.extend(read_json(output_root / "shards" / f"manifest_shard_{index:02d}.json")["samples"])
        rows.extend(read_json(output_root / "shards" / f"audit_shard_{index:02d}.json")["samples"])
    if len(records) != expected_samples or len(rows) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples/audits, got {len(records)}/{len(rows)}."
        )
    if len({row["sample_id"] for row in records}) != expected_samples:
        raise ValueError("Sample IDs are not unique.")
    variant_counts = Counter(row["variant"] for row in rows)
    split_counts = Counter(row["split"] for row in records)
    expected_per_variant = 50 if args.dataset_mode == "training" else 5
    if variant_counts != Counter({variant: expected_per_variant for variant in variants}):
        raise ValueError(f"Unexpected variant counts: {dict(variant_counts)}")
    expected_splits = SAMPLE_SPLITS if args.dataset_mode == "training" else {"test": 25}
    if dict(split_counts) != expected_splits:
        raise ValueError(f"Unexpected split counts: {dict(split_counts)}")
    if not all(bool(row["contract_pass"]) for row in rows):
        raise RuntimeError("One or more full-dataset contract audits failed.")
    if any(
        str(row.get("smoothing_profile", LEGACY_SMOOTHING_PROFILE)) != args.smoothing_profile
        for row in rows
    ):
        raise RuntimeError("Shard audits mix smoothing profiles.")
    by_object: dict[str, set[str]] = defaultdict(set)
    object_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_object[row["object_id"]].add(row["variant"])
        object_splits[row["object_id"]].add(row["split"])
    expected_objects = 50 if args.dataset_mode == "training" else 5
    if len(by_object) != expected_objects or any(values != set(variants) for values in by_object.values()):
        raise RuntimeError("Every object must have exactly the configured variants.")
    if any(len(values) != 1 for values in object_splits.values()):
        raise RuntimeError("Object-level split leakage detected.")
    if args.dataset_mode == "training":
        stats = aggregate_stats(rows)
        c_faces = [stats[name]["faces"]["mean"] for name in ("C1", "C2", "C3", "C4")]
        monotonic = all(left >= right for left, right in zip(c_faces, c_faces[1:]))
        meaningful = all(left / right >= 1.05 for left, right in zip(c_faces, c_faces[1:]))
        if not monotonic or not meaningful:
            raise RuntimeError(f"C1-C4 topology densities are not meaningfully monotonic: {c_faces}")
    else:
        stats = aggregate_stats_for_variants(rows, variants)
        monotonic = True
        meaningful = True
    records.sort(key=lambda row: (row["split"], row["sample_id"]))
    manifest = {
        "format_version": dataset_family,
        "dataset_name": dataset_family,
        "dataset_role": "multi_topology_clean_reference_raw_laplacian_restoration",
        "dataset_root": str(output_root),
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "object_count": expected_objects,
        "variants_per_object": len(variants),
        "sample_count": expected_samples,
        "object_split_counts": OBJECT_SPLITS if args.dataset_mode == "training" else {"test": 5},
        "sample_split_counts": expected_splits,
        "object_level_split_enforced": True,
        "views": 28,
        "resolution": 960,
        "target_mode": "raw_laplacian",
        "target_definition": "L(clean_reference_faces)@clean_reference_vertices",
        "h2_normalization_used": False,
        "proxy_used": False,
        "target_transfer_used": False,
        "smoothing_profile": args.smoothing_profile,
        "smoothing_only_change_vs_legacy_v1": (
            args.smoothing_profile == STRONG_SMOOTHING_PROFILE
        ),
        "perturbation_seed_namespace": SEED_NAMESPACE,
        "samples": records,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(
        output_root / "full_audit.json",
        {
            "format_version": dataset_family,
            "contract_audit": True,
            "sample_count": expected_samples,
            "variant_counts": dict(variant_counts),
            "sample_split_counts": dict(split_counts),
            "all_target_recompute_exact_float32": all(row["target_recompute_exact_float32"] for row in rows),
            "maximum_target_recompute_error": max(row["target_recompute_max_abs_error"] for row in rows),
            "all_clean_input_faces_exact": all(row["clean_input_faces_exact"] for row in rows),
            "all_finite": all(row["all_tensors_finite"] for row in rows),
            "smoothing_profile": args.smoothing_profile,
            "perturbation_seed_namespace": SEED_NAMESPACE,
            "all_strong_smoothing_budget_pass": all(
                row.get("strong_smoothing_budget_pass", True) for row in rows
            ),
            "c_density_monotonic": monotonic,
            "c_density_meaningfully_distinct": meaningful,
            "topology_statistics": stats,
            "samples": rows,
        },
    )
    print(json.dumps({"status": "passed", "samples": expected_samples, "split_counts": dict(split_counts)}, indent=2))


def calibrate(args: argparse.Namespace) -> None:
    manifest_path = args.source_manifest.expanduser().resolve()
    dataset_family = dataset_family_for(args.smoothing_profile, args.dataset_mode)
    rows = []
    for record in source_records(manifest_path):
        source = torch.load(resolve_record(manifest_path, record), map_location="cpu", weights_only=False)
        vertices = source["gt_vertices"].detach().cpu().double().numpy()
        faces = source["gt_faces"].detach().cpu().numpy().astype(np.int64)
        for variant in VARIANT_NAMES:
            clean, metadata = construct_clean_topology(vertices, faces, variant, max_vertices=args.max_vertices)
            rows.append(
                {
                    "object_id": str(record["sample_id"]),
                    "variant": variant,
                    "vertices": clean.num_vertices,
                    "faces": clean.num_faces,
                    "face_ratio_vs_gt": metadata["face_ratio_vs_gt"],
                    "edge_only_selected_faces_total": metadata["edge_only_selected_faces_total"],
                    "tau_area_normalized": metadata["tau_area_normalized"],
                    "tau_edge_normalized": metadata["tau_edge_normalized"],
                }
            )
    summary = {}
    for variant in VARIANT_NAMES:
        values = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "mean_vertices": float(np.mean([row["vertices"] for row in values])),
            "mean_faces": float(np.mean([row["faces"] for row in values])),
            "mean_face_ratio_vs_gt": float(np.mean([row["face_ratio_vs_gt"] for row in values])),
            "edge_only_selected_faces_total": int(sum(row["edge_only_selected_faces_total"] for row in values)),
        }
    value = {
        "format_version": dataset_family,
        "smoothing_profile": args.smoothing_profile,
        "summary": summary,
        "samples": rows,
    }
    if args.calibration_output:
        write_json(args.calibration_output.expanduser().resolve(), value)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    home = Path.home()
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source-manifest",
        type=Path,
        default=home / "sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json",
    )
    value.add_argument("--output-root", type=Path)
    value.add_argument("--downstream-root", type=Path, default=home / "multiview-laplacian-refinement")
    value.add_argument("--upstream-root", type=Path, default=home / "data_prepare")
    value.add_argument("--seed", type=int, default=7)
    value.add_argument("--dataset-mode", choices=("training", "unseen"), default="training")
    value.add_argument(
        "--smoothing-profile",
        choices=tuple(DEGRADATION_PROFILES),
        default=DEFAULT_SMOOTHING_PROFILE,
    )
    value.add_argument("--shard-count", type=int, default=2)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--max-vertices", type=int, default=500_000)
    value.add_argument("--visibility-backend", choices=("cuda", "opengl"), default="opengl")
    value.add_argument("--limit", type=int)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--merge-only", action="store_true")
    value.add_argument("--calibrate-only", action="store_true")
    value.add_argument("--calibration-output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.calibrate_only:
        calibrate(args)
    elif args.merge_only:
        merge(args)
    else:
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Build FUTURE-2000 x 10 using the audited Sofa50 v2 coarse recipes.

The 2,000 original FUTURE GT meshes define geometry and the existing expanded
FUTURE dataset supplies the already-rendered 28-view observations/cameras.  RGB
files are referenced lazily and are never duplicated.  Every object receives
the exact A1--D2 topology/degradation recipes used by strong_smooth_v2.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.learned_laplacian.multitopology_rawlap import (
    STRONG_SMOOTHING_PROFILE,
    VARIANT_NAMES,
)
from prepare_sofa50_multitopology_rawlap500 import (
    aggregate_stats,
    build_sample,
    load_script,
    make_cuda_visibility,
    read_json,
    resolve_record,
    write_json,
)


OBJECT_COUNT = 2000
VARIANT_COUNT = 10
SAMPLE_COUNT = OBJECT_COUNT * VARIANT_COUNT
OBJECT_SPLITS = {"train": 1600, "validation": 200, "test": 200}
SAMPLE_SPLITS = {key: value * VARIANT_COUNT for key, value in OBJECT_SPLITS.items()}
FORMAT_VERSION = "Future2000MultiTopologyRawLap20000_v1"
SEED_NAMESPACE = FORMAT_VERSION
SOURCE_DEGENERATE_DOUBLE_AREA_EPSILON = 1e-14


def _records(path: Path, *, expected: int) -> list[dict[str, Any]]:
    rows = read_json(path).get("samples")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Expected {expected} records in {path}, found {len(rows or [])}")
    return [dict(row) for row in rows]


def _split_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        split: sum(str(row.get("split")) == split for row in rows)
        for split in OBJECT_SPLITS
    }


def _observation_object_id(sample_id: str) -> tuple[str, int]:
    object_id, marker, value = sample_id.rpartition("__v")
    if not marker or not object_id or len(value) != 2 or not value.isdigit():
        raise ValueError(f"Unexpected expanded FUTURE sample id: {sample_id!r}")
    return object_id, int(value)


def observation_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _records(path, expected=OBJECT_COUNT * 5)
    selected: dict[str, dict[str, Any]] = {}
    seen_variants: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        object_id, variant = _observation_object_id(str(row["sample_id"]))
        seen_variants[object_id].add(variant)
        if variant == 0:
            if object_id in selected:
                raise ValueError(f"Duplicate v00 observation source for {object_id}")
            selected[object_id] = row
    if len(selected) != OBJECT_COUNT:
        raise ValueError(f"Expected {OBJECT_COUNT} v00 observation sources, found {len(selected)}")
    if any(values != set(range(5)) for values in seen_variants.values()):
        raise ValueError("Existing expanded dataset does not contain exactly v00--v04 per object")
    return selected


def clean_source_geometry(
    vertices_tensor: torch.Tensor,
    faces_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Remove geometric zero-area faces and newly unreferenced vertices deterministically."""

    vertices = vertices_tensor.detach().cpu().double().numpy()
    faces = faces_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Source GT vertices must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Source GT faces must have shape [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("Source GT vertices contain NaN or Inf")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("Source GT faces contain an out-of-range vertex index")

    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    retained_mask = doubled_area > SOURCE_DEGENERATE_DOUBLE_AREA_EPSILON
    removed_faces = np.flatnonzero(~retained_mask).astype(np.int64)
    retained_faces = faces[retained_mask]
    if len(retained_faces) == 0:
        raise RuntimeError("Source GT cleaning removed every face")

    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[np.unique(retained_faces)] = True
    new_to_old = np.flatnonzero(referenced).astype(np.int64)
    removed_vertices = np.flatnonzero(~referenced).astype(np.int64)
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(new_to_old), dtype=np.int64)
    cleaned_faces = old_to_new[retained_faces]
    cleaned_vertices = vertices[new_to_old]

    cleaned_triangles = cleaned_vertices[cleaned_faces]
    cleaned_doubled_area = np.linalg.norm(
        np.cross(
            cleaned_triangles[:, 1] - cleaned_triangles[:, 0],
            cleaned_triangles[:, 2] - cleaned_triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(cleaned_doubled_area <= SOURCE_DEGENERATE_DOUBLE_AREA_EPSILON):
        raise RuntimeError("Source GT cleaning retained a geometric degenerate face")

    audit = {
        "policy": "remove_faces_with_double_area_le_1e-14_then_remove_unreferenced_vertices",
        "double_area_epsilon": SOURCE_DEGENERATE_DOUBLE_AREA_EPSILON,
        "applied": bool(len(removed_faces) or len(removed_vertices)),
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "cleaned_vertices": int(len(cleaned_vertices)),
        "cleaned_faces": int(len(cleaned_faces)),
        "removed_degenerate_faces": int(len(removed_faces)),
        "removed_unreferenced_vertices": int(len(removed_vertices)),
        "removed_face_indices": removed_faces.tolist(),
        "removed_vertex_indices": removed_vertices.tolist(),
        "minimum_retained_double_area": float(cleaned_doubled_area.min()),
    }
    return (
        torch.as_tensor(cleaned_vertices, dtype=vertices_tensor.dtype),
        torch.as_tensor(cleaned_faces, dtype=torch.long),
        audit,
    )


def combined_source(
    source_manifest: Path,
    source_record: Mapping[str, Any],
    observation_manifest: Path,
    observation_record: Mapping[str, Any],
) -> dict[str, Any]:
    original = torch.load(
        resolve_record(source_manifest, source_record), map_location="cpu", weights_only=False
    )
    observation = torch.load(
        resolve_record(observation_manifest, observation_record),
        map_location="cpu",
        weights_only=False,
    )
    object_id = str(source_record["sample_id"])
    observed_object_id, observed_variant = _observation_object_id(
        str(observation_record["sample_id"])
    )
    if observed_object_id != object_id or observed_variant != 0:
        raise RuntimeError(f"Observation/source lineage mismatch for {object_id}")
    if str(source_record["split"]) != str(observation_record["split"]):
        raise RuntimeError(f"Observation/source split mismatch for {object_id}")
    if len(observation.get("image_paths", [])) != 28:
        raise RuntimeError(f"{object_id}: expected 28 archived image paths")
    if tuple(observation["intrinsics"].shape) != (28, 3, 3):
        raise RuntimeError(f"{object_id}: invalid archived intrinsics")
    if tuple(observation["extrinsics"].shape) != (28, 4, 4):
        raise RuntimeError(f"{object_id}: invalid archived extrinsics")
    cleaned_vertices, cleaned_faces, source_cleaning = clean_source_geometry(
        original["gt_vertices"], original["gt_faces"]
    )
    return {
        "sample_id": object_id,
        "gt_vertices": cleaned_vertices,
        "gt_faces": cleaned_faces,
        "image_paths": list(observation["image_paths"]),
        "intrinsics": observation["intrinsics"],
        "extrinsics": observation["extrinsics"],
        "prepared_image_size": int(observation.get("prepared_image_size", 960)),
        "source_image_size": list(observation.get("source_image_size", [960, 960])),
        "source_cleaning": source_cleaning,
    }


def generate(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.expanduser().resolve()
    observation_manifest = args.observation_manifest.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    downstream = args.downstream_root.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    if output.exists() and (output / "manifest.json").exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite completed dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_rows = _records(source_manifest, expected=OBJECT_COUNT)
    if _split_counts(source_rows) != OBJECT_SPLITS:
        raise ValueError(f"Unexpected source splits: {_split_counts(source_rows)}")
    observations = observation_index(observation_manifest)
    source_ids = {str(row["sample_id"]) for row in source_rows}
    if source_ids != set(observations):
        raise ValueError("Original GT and 28-view observation object IDs differ")

    current_helper = load_script(
        upstream / "scripts" / "prepare_sofa50_synthetic_current.py",
        f"future20k_current_helper_{args.shard_index}",
    )
    current_helper.VIEW_COUNT = 28
    deps = current_helper._dependencies(downstream)
    if args.visibility_backend == "cuda":
        deps["compute_renderer_visibility"] = make_cuda_visibility(upstream)

    selected = [
        row
        for index, row in enumerate(source_rows)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    output_records: list[dict[str, str]] = []
    audits: list[dict[str, Any]] = []
    for object_index, record in enumerate(selected, start=1):
        object_id = str(record["sample_id"])
        split = str(record["split"])
        source = combined_source(
            source_manifest,
            record,
            observation_manifest,
            observations[object_id],
        )
        print(f"[{object_index}/{len(selected)}] {object_id} split={split}", flush=True)
        for variant in VARIANT_NAMES:
            relative = Path("prepared") / split / object_id / f"{variant}.pt"
            sample_path = output / relative
            visibility_path = (
                output / "renderer_visibility" / split / object_id / f"{variant}.npz"
            )
            audit_path = output / "sample_audits" / split / object_id / f"{variant}.json"
            complete = sample_path.is_file() and visibility_path.is_file() and audit_path.is_file()
            if args.resume and complete:
                sample = torch.load(sample_path, map_location="cpu", weights_only=False)
                audit = read_json(audit_path)
                deps["validate_sample"](sample)
                if (
                    audit.get("dataset_family") != FORMAT_VERSION
                    or audit.get("seed_namespace") != SEED_NAMESPACE
                    or audit.get("smoothing_profile") != STRONG_SMOOTHING_PROFILE
                ):
                    raise RuntimeError(f"{object_id}__{variant}: resume contract mismatch")
            else:
                if not args.resume and any(
                    path.exists() for path in (sample_path, visibility_path, audit_path)
                ):
                    raise RuntimeError(
                        f"{object_id}__{variant}: partial output exists; use a clean output or resume"
                    )
                sample, audit = build_sample(
                    source,
                    object_id=object_id,
                    split=split,
                    variant=variant,
                    base_seed=args.seed,
                    source_root=observation_manifest.parent,
                    output_root=output,
                    visibility_artifact=visibility_path,
                    current_helper=current_helper,
                    deps=deps,
                    max_vertices=args.max_vertices,
                    visibility_backend=args.visibility_backend,
                    dataset_family=FORMAT_VERSION,
                    smoothing_profile=STRONG_SMOOTHING_PROFILE,
                    seed_namespace=SEED_NAMESPACE,
                    source_dataset_label="3D-FUTURE",
                )
                source_cleaning = dict(source["source_cleaning"])
                audit["source_cleaning"] = source_cleaning
                audit["source_cleaning_applied"] = bool(source_cleaning["applied"])
                audit["source_degenerate_faces_removed"] = int(
                    source_cleaning["removed_degenerate_faces"]
                )
                audit["source_unreferenced_vertices_removed"] = int(
                    source_cleaning["removed_unreferenced_vertices"]
                )
                sample["metadata"]["source_cleaning"] = source_cleaning
                sample["metadata"]["observation_source_dataset"] = str(
                    observation_manifest.parent
                )
                sample["metadata"]["observation_source_variant"] = "v00"
                sample["metadata"]["rgb_files_duplicated"] = False
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_sample = {
                    key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()
                    if key
                    not in {
                        "_static_prepared",
                        "_dataset_root",
                        "edge_index",
                        "vertex_degree",
                        "normalized_laplacian_target",
                    }
                }
                torch.save(cpu_sample, sample_path)
                write_json(audit_path, audit)
            if not audit["contract_pass"]:
                raise RuntimeError(f"Contract audit failed for {audit['sample_id']}")
            output_records.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "split": split,
                    "path": relative.as_posix(),
                }
            )
            audits.append(audit)
            print(
                f"  {variant}: V={audit['clean_vertices']} F={audit['clean_faces']} "
                f"disp/bbox={audit['clean_to_input_displacement_mean_over_bbox_diagonal']:.6g}",
                flush=True,
            )

    write_json(
        output / "shards" / f"manifest_shard_{args.shard_index:02d}.json",
        {"format_version": FORMAT_VERSION, "samples": output_records},
    )
    write_json(
        output / "shards" / f"audit_shard_{args.shard_index:02d}.json",
        {"format_version": FORMAT_VERSION, "samples": audits},
    )
    print(json.dumps({"shard": args.shard_index, "samples": len(output_records)}, indent=2))


def merge(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    records: list[dict[str, str]] = []
    audits: list[dict[str, Any]] = []
    for index in range(args.shard_count):
        records.extend(
            read_json(output / "shards" / f"manifest_shard_{index:02d}.json")["samples"]
        )
        audits.extend(
            read_json(output / "shards" / f"audit_shard_{index:02d}.json")["samples"]
        )
    if len(records) != SAMPLE_COUNT or len(audits) != SAMPLE_COUNT:
        raise ValueError(
            f"Expected {SAMPLE_COUNT} records/audits, got {len(records)}/{len(audits)}"
        )
    if len({row["sample_id"] for row in records}) != SAMPLE_COUNT:
        raise ValueError("Generated sample IDs are not unique")
    split_counts = Counter(row["split"] for row in records)
    variant_counts = Counter(row["variant"] for row in audits)
    if dict(split_counts) != SAMPLE_SPLITS:
        raise ValueError(f"Unexpected sample splits: {dict(split_counts)}")
    if variant_counts != Counter({variant: OBJECT_COUNT for variant in VARIANT_NAMES}):
        raise ValueError(f"Unexpected variant counts: {dict(variant_counts)}")
    if not all(bool(row["contract_pass"]) for row in audits):
        raise RuntimeError("One or more per-sample contract audits failed")
    if any(
        row.get("dataset_family") != FORMAT_VERSION
        or row.get("seed_namespace") != SEED_NAMESPACE
        or row.get("smoothing_profile") != STRONG_SMOOTHING_PROFILE
        for row in audits
    ):
        raise RuntimeError("Shard audit lineage mismatch")

    by_object: dict[str, set[str]] = defaultdict(set)
    object_splits: dict[str, set[str]] = defaultdict(set)
    for row in audits:
        by_object[str(row["object_id"])].add(str(row["variant"]))
        object_splits[str(row["object_id"])].add(str(row["split"]))
    if len(by_object) != OBJECT_COUNT or any(
        values != set(VARIANT_NAMES) for values in by_object.values()
    ):
        raise RuntimeError("Every FUTURE object must have exactly A1--D2")
    if any(len(values) != 1 for values in object_splits.values()):
        raise RuntimeError("Object-level split leakage detected")

    stats = aggregate_stats(audits)
    c_faces = [stats[name]["faces"]["mean"] for name in ("C1", "C2", "C3", "C4")]
    c_density_monotonic = all(left >= right for left, right in zip(c_faces, c_faces[1:]))
    c_density_distinct = all(left / right >= 1.05 for left, right in zip(c_faces, c_faces[1:]))
    if not c_density_monotonic or not c_density_distinct:
        raise RuntimeError(f"C1--C4 topology densities failed: {c_faces}")

    records.sort(key=lambda row: (row["split"], row["sample_id"]))
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset_name": FORMAT_VERSION,
        "dataset_role": "future2000_multi_topology_clean_reference_raw_laplacian_restoration",
        "dataset_root": str(output),
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "observation_manifest": str(args.observation_manifest.expanduser().resolve()),
        "observation_reuse_variant": "v00",
        "rgb_files_duplicated": False,
        "object_count": OBJECT_COUNT,
        "variants_per_object": VARIANT_COUNT,
        "sample_count": SAMPLE_COUNT,
        "object_split_counts": OBJECT_SPLITS,
        "sample_split_counts": SAMPLE_SPLITS,
        "object_level_split_enforced": True,
        "views": 28,
        "resolution": 960,
        "variants": list(VARIANT_NAMES),
        "recipe_source": "Sofa50 strong_smooth_v2 A1--D2",
        "target_mode": "raw_laplacian",
        "target_definition": "L(clean_reference_faces)@clean_reference_vertices",
        "smoothing_profile": STRONG_SMOOTHING_PROFILE,
        "perturbation_seed_namespace": SEED_NAMESPACE,
        "source_cleaning_policy": (
            "remove_faces_with_double_area_le_1e-14_then_remove_unreferenced_vertices"
        ),
        "samples": records,
    }
    full_audit = {
        "format_version": FORMAT_VERSION,
        "contract_audit": True,
        "sample_count": SAMPLE_COUNT,
        "variant_counts": dict(variant_counts),
        "sample_split_counts": dict(split_counts),
        "all_target_recompute_exact_float32": all(
            row["target_recompute_exact_float32"] for row in audits
        ),
        "maximum_target_recompute_error": max(
            row["target_recompute_max_abs_error"] for row in audits
        ),
        "all_clean_input_faces_exact": all(row["clean_input_faces_exact"] for row in audits),
        "all_finite": all(row["all_tensors_finite"] for row in audits),
        "all_strong_smoothing_budget_pass": all(
            row["strong_smoothing_budget_pass"] for row in audits
        ),
        "smoothing_profile": STRONG_SMOOTHING_PROFILE,
        "perturbation_seed_namespace": SEED_NAMESPACE,
        "source_cleaning_policy": (
            "remove_faces_with_double_area_le_1e-14_then_remove_unreferenced_vertices"
        ),
        "source_samples_with_cleaning": sum(
            bool(row.get("source_cleaning_applied", False)) for row in audits
        ),
        "source_degenerate_faces_removed": sum(
            int(row.get("source_degenerate_faces_removed", 0)) for row in audits
        ),
        "source_unreferenced_vertices_removed": sum(
            int(row.get("source_unreferenced_vertices_removed", 0)) for row in audits
        ),
        "c_density_monotonic": c_density_monotonic,
        "c_density_meaningfully_distinct": c_density_distinct,
        "topology_statistics": stats,
        "samples": audits,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "full_audit.json", full_audit)
    print(
        json.dumps(
            {
                "status": "passed",
                "objects": OBJECT_COUNT,
                "samples": SAMPLE_COUNT,
                "split_counts": dict(split_counts),
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    home = Path.home()
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source-manifest",
        type=Path,
        default=home / "future2000_compact/multiview_960/gt_query_manifest.json",
    )
    value.add_argument(
        "--observation-manifest",
        type=Path,
        default=home / "future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json",
    )
    value.add_argument(
        "--output-root", type=Path, default=home / FORMAT_VERSION
    )
    value.add_argument(
        "--downstream-root", type=Path, default=home / "multiview-laplacian-refinement"
    )
    value.add_argument("--upstream-root", type=Path, default=home / "data_prepare")
    value.add_argument("--seed", type=int, default=7)
    value.add_argument("--shard-count", type=int, default=16)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--max-vertices", type=int, default=500_000)
    value.add_argument("--visibility-backend", choices=("cuda", "opengl"), default="cuda")
    value.add_argument("--limit", type=int)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--merge-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.shard_count < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Invalid shard or limit arguments")
    if args.merge_only:
        merge(args)
    else:
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

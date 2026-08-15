#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from mlr.data import Camera
from mlr.io import load_mesh
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view


SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"train": 200, "validation": 25, "test": 25}
VIEW_COUNT = 28
SOURCE_SIZE = 960
OUTPUT_SIZE = 1920
UNCHANGED_TENSOR_FIELDS = (
    "extrinsics",
    "vertices",
    "faces",
    "vertex_normals",
    "initial_laplacian",
    "laplacian_target",
    "raw_laplacian_target",
    "normalized_laplacian_target",
    "target_confidence",
    "target_positions",
    "gt_vertices",
    "gt_faces",
    "position_normalization_center",
    "position_normalization_scale",
    "local_edge_length",
    "local_edge_scale",
    "valid_scale_mask",
    "visibility_backface_only",
    "visibility_occlusion_only",
    "visibility_backface_and_occlusion",
    "visibility",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_record(manifest: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _resolve_image(source_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def scale_intrinsics_to_1920(intrinsics: torch.Tensor) -> torch.Tensor:
    if tuple(intrinsics.shape) != (VIEW_COUNT, 3, 3):
        raise ValueError(f"Expected 28 intrinsics, got {tuple(intrinsics.shape)}")
    output = intrinsics.detach().clone()
    output[:, 0, :] *= OUTPUT_SIZE / SOURCE_SIZE
    output[:, 1, :] *= OUTPUT_SIZE / SOURCE_SIZE
    return output


def _object_id(sample_id: str) -> str:
    marker = "__v"
    if marker not in sample_id:
        raise ValueError(f"Invalid synthetic-current sample ID: {sample_id}")
    return sample_id.split(marker, maxsplit=1)[0]


def _camera_list(source: Mapping[str, Any]) -> list[Camera]:
    intrinsics = scale_intrinsics_to_1920(source["intrinsics"]).double().numpy()
    extrinsics = source["extrinsics"].detach().cpu().double().numpy()
    if tuple(extrinsics.shape) != (VIEW_COUNT, 4, 4):
        raise ValueError("Expected 28 extrinsics")
    return [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(OUTPUT_SIZE, OUTPUT_SIZE),
            name=f"matched_28view_{index:04d}",
        )
        for index in range(VIEW_COUNT)
    ]


def _render_config() -> SyntheticRenderConfig:
    return SyntheticRenderConfig(
        num_views=VIEW_COUNT,
        width=OUTPUT_SIZE,
        height=OUTPUT_SIZE,
        trajectory="nested_cube_surface",
        radius_scale=2.5,
        elevation_degrees=20.0,
        min_elevation_degrees=-60.0,
        max_elevation_degrees=60.0,
        fov_degrees=90.0,
        render_mode="lit",
        backend="cpu",
        normalize_mesh=False,
        cube_half_extent=1.5,
        antialiasing="msaa4",
        camera_layout_version="cube_surface_nested_fps_antipodal_14_28_56_cpu_master_v3",
        backface_culling=False,
        front_face_winding="ccw",
    )


def _render_object(
    source: Mapping[str, Any],
    source_root: Path,
    output_root: Path,
    object_id: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    source_images = [_resolve_image(source_root, value) for value in source["image_paths"]]
    if len(source_images) != VIEW_COUNT or not all(path.is_file() for path in source_images):
        raise FileNotFoundError(f"Incomplete source observations for {object_id}")
    source_render_dir = source_images[0].parent.parent
    mesh_path = source_render_dir / "mesh.obj"
    source_dataset = source_render_dir / "dataset.json"
    if not mesh_path.is_file() or not source_dataset.is_file():
        raise FileNotFoundError(f"Missing source render provenance for {object_id}")
    source_spec = _read_json(source_dataset).get("config", {})
    required_source = {
        "width": SOURCE_SIZE,
        "height": SOURCE_SIZE,
        "num_views": VIEW_COUNT,
        "backend": "cpu",
        "fov_degrees": 90.0,
        "normalize_mesh": False,
        "backface_culling": False,
        "front_face_winding": "ccw",
    }
    if any(source_spec.get(key) != value for key, value in required_source.items()):
        raise ValueError(f"Unexpected source renderer contract for {object_id}")

    image_dir = output_root / "observations" / object_id / "images"
    output_images = [image_dir / f"{index:04d}.png" for index in range(VIEW_COUNT)]
    if not resume or not all(path.is_file() for path in output_images):
        image_dir.mkdir(parents=True, exist_ok=True)
        mesh = load_mesh(mesh_path).ensure_normals()
        config = _render_config()
        for index, (camera, destination) in enumerate(
            zip(_camera_list(source), output_images, strict=True)
        ):
            rgb, _mask, _depth = render_mesh_view(mesh, camera, config)
            Image.fromarray(rgb).save(destination)
            print(f"rendered {object_id} view={index + 1}/{VIEW_COUNT}", flush=True)

    differences = []
    for source_path, output_path in zip(source_images, output_images, strict=True):
        with Image.open(source_path) as source_image:
            source_rgb = source_image.convert("RGB")
            resized = np.asarray(
                source_rgb.resize(
                    (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.int16,
            )
        with Image.open(output_path) as output_image:
            if output_image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                raise ValueError(f"Wrong output size: {output_path}")
            native = np.asarray(output_image.convert("RGB"), dtype=np.int16)
        differences.append(float(np.abs(native - resized).mean()))
    audit = {
        "object_id": object_id,
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": _file_sha256(mesh_path),
        "source_camera_poses_reused_exactly": True,
        "intrinsics_scaled_exactly_by_two": True,
        "renderer_backend": "cpu",
        "native_render_size": [OUTPUT_SIZE, OUTPUT_SIZE],
        "source_render_size": [SOURCE_SIZE, SOURCE_SIZE],
        "native_vs_bilinear_upsample_mean_abs_difference_minimum": min(differences),
        "native_vs_bilinear_upsample_mean_abs_difference_mean": float(
            np.mean(differences)
        ),
        "all_native_images_differ_from_resized_960": all(value > 0.0 for value in differences),
        "output_image_sha256": [_file_sha256(path) for path in output_images],
    }
    _write_json(output_root / "observations" / object_id / "render_provenance.json", audit)
    return audit


def _prepare_sample(
    source_path: Path,
    destination: Path,
    source_root: Path,
    output_root: Path,
    object_id: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    sample_id = str(source["sample_id"])
    output = copy.copy(dict(source))
    output["intrinsics"] = scale_intrinsics_to_1920(source["intrinsics"])
    output["image_paths"] = [
        (Path("observations") / object_id / "images" / f"{index:04d}.png").as_posix()
        for index in range(VIEW_COUNT)
    ]
    output["prepared_image_size"] = OUTPUT_SIZE
    output["source_image_size"] = [OUTPUT_SIZE, OUTPUT_SIZE]
    output["metadata"] = {
        **dict(source.get("metadata", {})),
        "input_resolution": OUTPUT_SIZE,
        "observation_resolution_ablation": "native_cpu_reference_1920_from_matched_28view_cameras",
        "observation_source_resolution": SOURCE_SIZE,
        "current_graph_and_targets_reused_exactly": True,
        "renderer_visibility_reused_exactly": True,
    }
    unchanged_hashes = {
        field: {
            "source": _tensor_sha256(source[field]),
            "output": _tensor_sha256(output[field]),
        }
        for field in UNCHANGED_TENSOR_FIELDS
    }
    exact_unchanged = all(
        hashes["source"] == hashes["output"] for hashes in unchanged_hashes.values()
    )
    if not exact_unchanged:
        raise RuntimeError(f"Non-image tensor changed for {sample_id}")
    expected_intrinsics = scale_intrinsics_to_1920(source["intrinsics"])
    if not torch.equal(output["intrinsics"], expected_intrinsics):
        raise RuntimeError(f"Intrinsics scaling failed for {sample_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    split = str(source["metadata"]["source_split"])
    record = {
        "sample_id": sample_id,
        "split": split,
        "path": destination.relative_to(output_root).as_posix(),
    }
    audit = {
        "sample_id": sample_id,
        "split": split,
        "source_path": str(source_path),
        "output_path": str(destination),
        "all_graph_target_visibility_tensors_exact": exact_unchanged,
        "unchanged_tensor_hashes": unchanged_hashes,
    }
    return record, audit


def prepare_shard(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.resolve()
    output_root = args.output_root.resolve()
    source = _read_json(source_manifest)
    records = source.get("samples", [])
    if len(records) != 250:
        raise ValueError(f"Expected 250 source samples, got {len(records)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_object_id(str(record["sample_id"])), []).append(dict(record))
    if len(groups) != 50 or any(len(values) != 5 for values in groups.values()):
        raise ValueError("Expected 50 objects with five variants each")

    output_records = []
    sample_audits = []
    render_audits = []
    for object_index, object_id in enumerate(sorted(groups)):
        if object_index % args.shard_count != args.shard_index:
            continue
        group = sorted(groups[object_id], key=lambda value: str(value["sample_id"]))
        first_path = _resolve_record(source_manifest, group[0])
        first = torch.load(first_path, map_location="cpu", weights_only=False)
        render_audits.append(
            _render_object(
                first,
                source_manifest.parent,
                output_root,
                object_id,
                resume=args.resume,
            )
        )
        for record in group:
            source_path = _resolve_record(source_manifest, record)
            split = str(record["split"])
            variant = str(record["sample_id"]).rsplit("__v", maxsplit=1)[1]
            destination = (
                output_root / "prepared" / split / object_id / f"variant_{variant}.pt"
            )
            output_record, audit = _prepare_sample(
                source_path,
                destination,
                source_manifest.parent,
                output_root,
                object_id,
            )
            if output_record["split"] != split:
                raise RuntimeError(f"Split mismatch for {output_record['sample_id']}")
            output_records.append(output_record)
            sample_audits.append(audit)
            print(f"prepared {output_record['sample_id']}", flush=True)
    payload = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _file_sha256(source_manifest),
        "records": output_records,
        "sample_audits": sample_audits,
        "render_audits": render_audits,
    }
    _write_json(output_root / "shards" / f"shard_{args.shard_index}.json", payload)


def merge_shards(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.resolve()
    output_root = args.output_root.resolve()
    source = _read_json(source_manifest)
    source_records = source.get("samples", [])
    shards = [
        _read_json(output_root / "shards" / f"shard_{index}.json")
        for index in range(args.shard_count)
    ]
    expected_hash = _file_sha256(source_manifest)
    if any(shard.get("source_manifest_sha256") != expected_hash for shard in shards):
        raise RuntimeError("Source manifest hash mismatch across shards")
    records_by_id = {
        str(record["sample_id"]): record
        for shard in shards
        for record in shard["records"]
    }
    ordered_records = [records_by_id[str(record["sample_id"])] for record in source_records]
    counts = {
        split: sum(record["split"] == split for record in ordered_records)
        for split in SPLITS
    }
    sample_audits = [row for shard in shards for row in shard["sample_audits"]]
    render_audits = [row for shard in shards for row in shard["render_audits"]]
    sample_ids_match = [record["sample_id"] for record in ordered_records] == [
        record["sample_id"] for record in source_records
    ]
    audit = {
        "passed": bool(
            len(ordered_records) == 250
            and counts == EXPECTED_COUNTS
            and sample_ids_match
            and len(render_audits) == 50
            and all(
                row["all_graph_target_visibility_tensors_exact"]
                for row in sample_audits
            )
            and all(
                row["source_camera_poses_reused_exactly"]
                and row["intrinsics_scaled_exactly_by_two"]
                and row["all_native_images_differ_from_resized_960"]
                for row in render_audits
            )
        ),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": expected_hash,
        "sample_count": len(ordered_records),
        "object_count": len(render_audits),
        "split_counts": counts,
        "sample_ids_and_order_match": sample_ids_match,
        "camera_extrinsics_reused_exactly": True,
        "intrinsics_change": "first two rows multiplied exactly by 2",
        "current_graph_proxy_targets_unchanged": all(
            row["all_graph_target_visibility_tensors_exact"] for row in sample_audits
        ),
        "renderer_visibility_tensors_unchanged": all(
            row["unchanged_tensor_hashes"]["visibility"]["source"]
            == row["unchanged_tensor_hashes"]["visibility"]["output"]
            for row in sample_audits
        ),
        "native_renderer": {
            "backend": "cpu",
            "resolution": [OUTPUT_SIZE, OUTPUT_SIZE],
            "same_renderer_parameters_except_resolution_and_intrinsics": True,
            "minimum_native_vs_resized_960_mean_abs_difference": min(
                row["native_vs_bilinear_upsample_mean_abs_difference_minimum"]
                for row in render_audits
            ),
        },
        "differences_from_source": [
            "native RGB observations rendered at 1920x1920",
            "camera intrinsics scaled exactly by 2",
            "image_paths point to the new native observations",
            "prepared/source image-size metadata changed from 960 to 1920",
            "metadata records resolution provenance",
        ],
        "unchanged_tensor_fields": list(UNCHANGED_TENSOR_FIELDS),
        "render_audits": render_audits,
    }
    _write_json(output_root / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("1920 dataset contract audit failed")
    manifest = {
        "format_version": "sofa50_synthetic_current_28view_native1920_v1",
        "dataset_role": "resolution_ablation_only",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": expected_hash,
        "view_count": VIEW_COUNT,
        "image_size": OUTPUT_SIZE,
        "renderer_backend": "cpu",
        "object_level_split_enforced": True,
        "variants_per_object": 5,
        "samples": ordered_records,
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(
        output_root / "sample_contract_audit.json",
        {"samples": sample_audits},
    )
    print(json.dumps({"status": "passed", "split_counts": counts}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard arguments")
    if args.merge_shards:
        merge_shards(args)
    else:
        prepare_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

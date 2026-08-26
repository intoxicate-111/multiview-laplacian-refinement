#!/usr/bin/env python3
from __future__ import annotations

"""Prepare the 56-view native-1920 input tier for the Sofa50 stress test.

The first 28 observations are linked byte-for-byte from the sealed 28-view
benchmark.  Only views 29--56 are rendered, using the continuation cameras of
the same nested 14/28/56 master.  Geometry is copied from the sealed benchmark;
GT fields are retained solely for the common evaluator and are never exposed by
the external-baseline adapters.
"""

import argparse
import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from mlr.data import Camera
from mlr.io import load_mesh
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view


SOURCE_SIZE = 960
OUTPUT_SIZE = 1920
OLD_VIEWS = 28
NEW_VIEWS = 56
VISIBILITY_FIELDS = (
    "visibility",
    "visibility_backface_only",
    "visibility_occlusion_only",
    "visibility_backface_and_occlusion",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _object_id(sample_id: str) -> str:
    head, marker, variant = sample_id.rpartition("__v")
    if not marker or not head or len(variant) != 2 or not variant.isdigit():
        raise ValueError(f"Unexpected Sofa50 sample ID: {sample_id}")
    return head


def _scaled_intrinsics(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != (NEW_VIEWS, 3, 3):
        raise ValueError(f"Expected [56,3,3] intrinsics, got {tuple(value.shape)}")
    result = value.detach().clone()
    result[:, 0, :] *= OUTPUT_SIZE / SOURCE_SIZE
    result[:, 1, :] *= OUTPUT_SIZE / SOURCE_SIZE
    return result


def _render_config() -> SyntheticRenderConfig:
    return SyntheticRenderConfig(
        num_views=NEW_VIEWS,
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


def _render_object(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    object_id, old_container_value, source56_value, source56_root_value, output_root_value = task
    old_container = Path(old_container_value)
    source56_path = Path(source56_value)
    source56_root = Path(source56_root_value)
    output_root = Path(output_root_value)
    old = torch.load(old_container, map_location="cpu", weights_only=False)
    source56 = torch.load(source56_path, map_location="cpu", weights_only=False)
    intrinsics = _scaled_intrinsics(source56["intrinsics"])
    extrinsics = source56["extrinsics"].detach().clone()
    if tuple(extrinsics.shape) != (NEW_VIEWS, 4, 4):
        raise ValueError(f"{object_id}: invalid 56-view extrinsics")
    if not torch.equal(old["intrinsics"], intrinsics[:OLD_VIEWS]):
        raise RuntimeError(f"{object_id}: first-28 intrinsics are not an exact prefix")
    if not torch.equal(old["extrinsics"], extrinsics[:OLD_VIEWS]):
        raise RuntimeError(f"{object_id}: first-28 extrinsics are not an exact prefix")

    old_images = [_resolve(Path(str(old["_dataset_root"])) if "_dataset_root" in old else old_container.parent, value) for value in old["image_paths"]]
    # Prepared containers normally store paths relative to their manifest's
    # dataset_root.  The sealed benchmark records the resolved paths, so fall
    # back to those in the caller-created sidecar when necessary.
    if len(old_images) != OLD_VIEWS or any(not path.is_file() for path in old_images):
        raise FileNotFoundError(f"{object_id}: sealed 28-view images are incomplete")
    source_images = [_resolve(source56_root, value) for value in source56["image_paths"]]
    if len(source_images) != NEW_VIEWS or any(not path.is_file() for path in source_images):
        raise FileNotFoundError(f"{object_id}: source 56-view images are incomplete")
    mesh_path = source_images[0].parent.parent / "mesh.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"{object_id}: missing source render mesh {mesh_path}")

    image_dir = output_root / "observations" / object_id / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(old_images):
        destination = image_dir / f"{index:04d}.png"
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())

    mesh = load_mesh(mesh_path).ensure_normals()
    config = _render_config()
    for index in range(OLD_VIEWS, NEW_VIEWS):
        destination = image_dir / f"{index:04d}.png"
        if destination.is_file():
            with Image.open(destination) as image:
                if image.size == (OUTPUT_SIZE, OUTPUT_SIZE):
                    continue
        camera = Camera(
            intrinsics=intrinsics[index].double().numpy(),
            rotation=extrinsics[index, :3, :3].double().numpy(),
            translation=extrinsics[index, :3, 3].double().numpy(),
            image_size=(OUTPUT_SIZE, OUTPUT_SIZE),
            name=f"nested_56view_{index:04d}",
        )
        rgb, _mask, _depth = render_mesh_view(mesh, camera, config)
        Image.fromarray(rgb).save(destination)
        print(f"rendered object={object_id} view={index + 1}/{NEW_VIEWS}", flush=True)

    outputs = [image_dir / f"{index:04d}.png" for index in range(NEW_VIEWS)]
    hashes = [_sha256(path) for path in outputs]
    old_hashes = [_sha256(path) for path in old_images]
    if hashes[:OLD_VIEWS] != old_hashes:
        raise RuntimeError(f"{object_id}: first-28 RGB hash identity failed")
    audit = {
        "object_id": object_id,
        "source_56_container": str(source56_path),
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": _sha256(mesh_path),
        "camera_layout": "cube_surface_nested_fps_antipodal_14_28_56_cpu_master_v3",
        "first_28_intrinsics_exact": True,
        "first_28_extrinsics_exact": True,
        "first_28_rgb_sha256_exact": True,
        "additional_native_1920_views": NEW_VIEWS - OLD_VIEWS,
        "image_paths": [str(path.resolve()) for path in outputs],
        "image_sha256": hashes,
    }
    _write_json(output_root / "observations" / object_id / "render_provenance.json", audit)
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_manifest_path = args.old_benchmark_manifest.resolve()
    source56_manifest_path = args.source_56_manifest.resolve()
    output_root = args.output_root.resolve()
    old_manifest = _read_json(old_manifest_path)
    source56_manifest = _read_json(source56_manifest_path)
    old_rows = [dict(row) for row in old_manifest["samples"]]
    if len(old_rows) != 25:
        raise ValueError(f"Expected 25 sealed benchmark samples, got {len(old_rows)}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in old_rows:
        groups.setdefault(_object_id(str(row["sample_id"])), []).append(row)
    if len(groups) != 5 or set(map(len, groups.values())) != {5}:
        raise ValueError("Expected five objects with five variants each")
    source56_by_id = {str(row["sample_id"]): dict(row) for row in source56_manifest["samples"]}
    missing = sorted(set(groups) - set(source56_by_id))
    if missing:
        raise ValueError(f"56-view manifest is missing objects: {missing}")

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for object_id, rows in sorted(groups.items()):
        representative = rows[0]
        old_container = Path(str(representative["camera_and_gt_container"])).resolve()
        # Make old relative image paths resolvable inside the worker without
        # mutating the sealed input container.
        old = torch.load(old_container, map_location="cpu", weights_only=False)
        resolved_old_images = [Path(str(value)).resolve() for value in representative["image_paths"]]
        old_sidecar = copy.copy(dict(old))
        old_sidecar["image_paths"] = [str(path) for path in resolved_old_images]
        sidecar = output_root / "source_sidecars" / f"{object_id}_old28.pt"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        torch.save(old_sidecar, sidecar)
        source_record = source56_by_id[object_id]
        source56_path = _resolve(source56_manifest_path.parent, str(source_record["path"]))
        tasks.append((object_id, str(sidecar), str(source56_path), str(source56_manifest_path.parent), str(output_root)))

    with ProcessPoolExecutor(max_workers=min(args.render_workers, len(tasks))) as pool:
        audits = list(pool.map(_render_object, tasks))
    audits_by_id = {row["object_id"]: row for row in audits}

    prepared_records = []
    benchmark_rows = []
    for row in old_rows:
        sample_id = str(row["sample_id"])
        object_id = _object_id(sample_id)
        old_container = Path(str(row["camera_and_gt_container"])).resolve()
        old = torch.load(old_container, map_location="cpu", weights_only=False)
        source_record = source56_by_id[object_id]
        source56_path = _resolve(source56_manifest_path.parent, str(source_record["path"]))
        source56 = torch.load(source56_path, map_location="cpu", weights_only=False)
        output = copy.copy(dict(old))
        output["intrinsics"] = _scaled_intrinsics(source56["intrinsics"])
        output["extrinsics"] = source56["extrinsics"].detach().clone()
        output["image_paths"] = [
            (Path("observations") / object_id / "images" / f"{index:04d}.png").as_posix()
            for index in range(NEW_VIEWS)
        ]
        output["prepared_storage_format"] = "lazy_image_paths_v1"
        output["prepared_image_size"] = OUTPUT_SIZE
        output["source_image_size"] = [OUTPUT_SIZE, OUTPUT_SIZE]
        for name in VISIBILITY_FIELDS:
            output[name] = None
        output["metadata"] = {
            **dict(old.get("metadata", {})),
            "external_stress_view_count": NEW_VIEWS,
            "first_28_rgb_and_cameras_preserved_exactly": True,
            "additional_views_from_same_nested_master": NEW_VIEWS - OLD_VIEWS,
            "visibility_fields_removed_because_external_adapters_do_not_consume_them": True,
        }
        variant = sample_id.rsplit("__v", 1)[1]
        destination = output_root / "prepared" / "test" / object_id / f"variant_{variant}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, destination)
        prepared_records.append({"sample_id": sample_id, "split": "test", "path": str(destination)})
        images = audits_by_id[object_id]["image_paths"]
        updated = copy.deepcopy(row)
        updated.update(
            {
                "path": str(destination),
                "image_directory": str((output_root / "observations" / object_id / "images").resolve()),
                "image_paths": images,
                "image_sha256": audits_by_id[object_id]["image_sha256"],
                "view_count": NEW_VIEWS,
                "image_size": [OUTPUT_SIZE, OUTPUT_SIZE],
                "camera_and_gt_container": str(destination),
                "camera_contract": "intrinsics[56,3,3] + world_to_camera extrinsics[56,4,4]",
                "visibility_contract": "not consumed by external adapters; removed from 56-view stress containers",
            }
        )
        benchmark_rows.append(updated)

    dataset_manifest = {
        "format_version": "sofa50_same_initial_external_56v_native1920_v1",
        "dataset_role": "external_baseline_view_count_stress_only",
        "dataset_root": str(output_root),
        "view_count": NEW_VIEWS,
        "image_size": OUTPUT_SIZE,
        "samples": prepared_records,
    }
    _write_json(output_root / "dataset_manifest.json", dataset_manifest)
    benchmark_manifest = copy.deepcopy(old_manifest)
    benchmark_manifest.update(
        {
            "format_version": "sofa50_same_initial_external_56v_stress_v1",
            "dataset_role": "unequal_view_stress_external56_vs_ours28",
            "source_28_benchmark_manifest": str(old_manifest_path),
            "source_28_benchmark_manifest_sha256": _sha256(old_manifest_path),
            "source_56_manifest": str(source56_manifest_path),
            "source_56_manifest_sha256": _sha256(source56_manifest_path),
            "source_manifest": str(output_root / "dataset_manifest.json"),
            "dataset_root": str(output_root),
            "common_input_contract": (
                "exact common initial mesh; external methods receive 56 native-1920 views "
                "whose first 28 RGB/cameras exactly match the sealed benchmark and whose "
                "next 28 cameras continue the same nested master; Ours remains 28-view"
            ),
            "samples": benchmark_rows,
        }
    )
    _write_json(output_root / "benchmark_manifest.json", benchmark_manifest)
    contract = {
        "passed": bool(
            len(benchmark_rows) == 25
            and len(audits) == 5
            and all(row["first_28_intrinsics_exact"] for row in audits)
            and all(row["first_28_extrinsics_exact"] for row in audits)
            and all(row["first_28_rgb_sha256_exact"] for row in audits)
        ),
        "sample_count": len(benchmark_rows),
        "object_count": len(audits),
        "external_view_count": NEW_VIEWS,
        "ours_view_count": OLD_VIEWS,
        "render_audits": audits,
    }
    _write_json(output_root / "contract_audit.json", contract)
    if not contract["passed"]:
        raise RuntimeError("56-view stress input contract failed")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-benchmark-manifest", type=Path, required=True)
    parser.add_argument("--source-56-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-workers", type=int, default=5)
    args = parser.parse_args()
    if args.render_workers < 1:
        raise ValueError("--render-workers must be positive")
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

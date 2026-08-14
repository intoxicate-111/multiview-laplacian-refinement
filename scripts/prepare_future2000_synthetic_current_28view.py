#!/usr/bin/env python3
from __future__ import annotations

"""Build a sharded 3D-FUTURE-2000 x5 GT-adaptive current-graph, 28-view dataset.

The first 14 observations are reused from the prepared upstream dataset.  The
additional nested 14 cameras use the same deterministic layout as the Sofa50
28-view experiment and are rendered with the same OpenGL backend.  Masks and
depth images are deliberately not persisted: current-graph visibility is
recomputed per variant with CUDA/nvdiffrast.  Before generating the five
synthetic-current variants, each source GT graph is adaptively subdivided to
match the maximum represented-vertex-area threshold of its GT-sub2 reference.
"""

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


OBJECT_COUNT = 2000
VARIANT_COUNT = 5
VIEW_COUNT = 28
EXPECTED_OBJECT_SPLITS = {"train": 1600, "validation": 200, "test": 200}
EXPECTED_VARIANT_SPLITS = {key: value * VARIANT_COUNT for key, value in EXPECTED_OBJECT_SPLITS.items()}
FORMAT_VERSION = "future2000_gt_adaptive_synthetic_current_28view_v2"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_record(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def base_cameras(source: Mapping[str, Any], camera_type: Any) -> list[Any]:
    intrinsics = source["intrinsics"].detach().cpu().numpy()
    extrinsics = source["extrinsics"].detach().cpu().numpy()
    if intrinsics.shape[0] != 14 or extrinsics.shape[0] != 14:
        raise ValueError("The upstream source must contain exactly 14 base cameras")
    image_size = int(source.get("prepared_image_size", 960))
    return [
        camera_type(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(image_size, image_size),
            name=f"base_{index:04d}",
        )
        for index in range(14)
    ]


def camera_tensors(cameras: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.as_tensor(
        np.stack([camera.intrinsics for camera in cameras]), dtype=torch.float32
    )
    extrinsics = []
    for camera in cameras:
        value = np.eye(4, dtype=np.float32)
        value[:3, :3] = np.asarray(camera.rotation, dtype=np.float32)
        value[:3, 3] = np.asarray(camera.translation, dtype=np.float32)
        extrinsics.append(value)
    return intrinsics, torch.as_tensor(np.stack(extrinsics), dtype=torch.float32)


def source_image_paths(source: Mapping[str, Any], source_root: Path) -> list[Path]:
    values = source.get("image_paths")
    if not isinstance(values, list) or len(values) != 14:
        raise ValueError("The upstream source must contain exactly 14 image paths")
    output = []
    for value in values:
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        output.append(resolved)
    return output


def render_added_views(
    source: Mapping[str, Any],
    cameras: list[Any],
    object_id: str,
    output_root: Path,
    deps: Mapping[str, Any],
    *,
    resume: bool,
) -> list[Path]:
    image_dir = output_root / "observations" / object_id / "images"
    paths = [image_dir / f"{index:04d}.png" for index in range(14, VIEW_COUNT)]
    if resume and all(path.is_file() for path in paths):
        return paths
    image_dir.mkdir(parents=True, exist_ok=True)
    mesh = deps["Mesh"](
        source["gt_vertices"].detach().cpu().double().numpy(),
        source["gt_faces"].detach().cpu().numpy(),
    ).ensure_normals()
    config = deps["SyntheticRenderConfig"](
        num_views=14,
        width=960,
        height=960,
        trajectory="nested_cube_surface",
        fov_degrees=90.0,
        render_mode="lit",
        backend="opengl",
        normalize_mesh=False,
        opengl_context_backend="egl",
        cube_half_extent=1.5,
        antialiasing="msaa4",
        camera_layout_version="cube_surface_nested_fps_antipodal_14_28_56_cpu_master_v3",
        backface_culling=False,
        front_face_winding="ccw",
    )
    rendered = deps["render_mesh_views_opengl"](mesh, cameras[14:VIEW_COUNT], config)
    if len(rendered) != 14:
        raise RuntimeError("OpenGL renderer returned the wrong view count")
    for path, (rgb, _mask, _depth) in zip(paths, rendered, strict=True):
        Image.fromarray(rgb).save(path)
    return paths


def make_source_28(
    source: Mapping[str, Any],
    cameras: list[Any],
    base_images: list[Path],
    added_images: list[Path],
) -> dict[str, Any]:
    output = copy.copy(dict(source))
    intrinsics, extrinsics = camera_tensors(cameras)
    output["intrinsics"] = intrinsics
    output["extrinsics"] = extrinsics
    output["image_paths"] = [str(path) for path in base_images + added_images]
    output["prepared_storage_format"] = "lazy_image_paths_v1"
    output["prepared_image_size"] = 960
    output["source_image_size"] = [960, 960]
    return output


def cuda_visibility_wrapper(upstream_root: Path) -> Any:
    # Import the leaf module directly.  Importing the sofa50_refinement package
    # executes its broad __init__, which unnecessarily requires mesh-preparation
    # dependencies that are irrelevant to CUDA visibility on the training HPC.
    module = load_script(
        upstream_root / "src" / "sofa50_refinement" / "gpu_visibility.py",
        "future2000_gpu_visibility_helper",
    )
    compute_renderer_visibility_cuda = module.compute_renderer_visibility_cuda
    plugin_loaded = False

    def load_cached_plugin_if_needed() -> None:
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
            Path.home().glob(
                ".cache/torch_extensions/*/nvdiffrast_plugin/nvdiffrast_plugin.so"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                "Neither Ninja nor a compiled nvdiffrast_plugin.so cache is available"
            )
        spec = importlib.util.spec_from_file_location("nvdiffrast_plugin", candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load cached nvdiffrast plugin: {candidates[0]}")
        compiled = importlib.util.module_from_spec(spec)
        sys.modules["nvdiffrast_plugin"] = compiled
        spec.loader.exec_module(compiled)
        cache[False] = compiled
        plugin_loaded = True

    def compute(mesh: Any, cameras: list[Any], config: Any, *, neighborhood_radius: int = 1) -> Any:
        load_cached_plugin_if_needed()
        return compute_renderer_visibility_cuda(
            mesh,
            cameras,
            image_size=int(config.width),
            neighborhood_radius=neighborhood_radius,
            front_face_winding=str(config.front_face_winding),
        )

    return compute


def records_from_manifest(path: Path) -> list[dict[str, Any]]:
    records = read_json(path).get("samples")
    if not isinstance(records, list) or len(records) != OBJECT_COUNT:
        raise ValueError(f"Expected {OBJECT_COUNT} source objects, found {len(records or [])}")
    counts = {
        split: sum(str(record.get("split")) == split for record in records)
        for split in EXPECTED_OBJECT_SPLITS
    }
    if counts != EXPECTED_OBJECT_SPLITS:
        raise ValueError(f"Unexpected object split counts: {counts}")
    return [dict(record) for record in records]


def make_gt_adaptive_source(
    source: Mapping[str, Any],
    adaptive: Any,
    *,
    reference: str,
    area_scale: float,
    max_iters: int,
    max_vertices: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace the source query graph with its per-object GT-adaptive graph."""

    gt_vertices = source["gt_vertices"].detach().cpu().double().numpy()
    gt_faces = source["gt_faces"].detach().cpu().numpy().astype(np.int64)
    variants, info = adaptive.build_variants(
        gt_vertices,
        gt_faces,
        adaptive_reference=reference,
        adaptive_area_scale=area_scale,
        adaptive_max_iters=max_iters,
        max_vertices=max_vertices,
    )
    adaptive_vertices, adaptive_faces = variants["gt_adaptive"]
    original_stats = adaptive.mesh_stats(gt_vertices, gt_faces)
    adaptive_stats = adaptive.mesh_stats(adaptive_vertices, adaptive_faces)
    sub2_stats = adaptive.mesh_stats(*variants["gt_sub2"])
    final_history = info["history"][-1]
    if final_history["oversized_vertices"] != 0:
        raise RuntimeError("GT-adaptive graph did not satisfy its represented-area threshold")

    output = copy.copy(dict(source))
    vertices_tensor = torch.as_tensor(adaptive_vertices, dtype=torch.float32)
    faces_tensor = torch.as_tensor(adaptive_faces, dtype=torch.long)
    output["vertices"] = vertices_tensor
    output["faces"] = faces_tensor
    output["gt_vertices"] = vertices_tensor
    output["gt_faces"] = faces_tensor
    metadata = copy.deepcopy(dict(source.get("metadata", {})))
    metadata.update(
        {
            "query_graph_variant": "gt_adaptive",
            "query_graph_surface": "same_piecewise_linear_gt_surface",
            "adaptive_subdivision_criterion": "represented_vertex_area",
            "adaptive_reference": reference,
            "adaptive_max_represented_vertex_area_reference": info[
                "reference_max_represented_vertex_area"
            ],
            "adaptive_max_represented_vertex_area_scale": info["area_scale"],
            "adaptive_max_represented_vertex_area_threshold": info["threshold"],
            "adaptive_iterations": len(info["history"]) - 1,
        }
    )
    output["metadata"] = metadata
    audit = {
        "source_vertices": int(original_stats["vertices"]),
        "source_faces": int(original_stats["faces"]),
        "adaptive_vertices": int(adaptive_stats["vertices"]),
        "adaptive_faces": int(adaptive_stats["faces"]),
        "sub2_vertices": int(sub2_stats["vertices"]),
        "sub2_faces": int(sub2_stats["faces"]),
        "adaptive_vertex_ratio_vs_sub2": float(adaptive_stats["vertices"])
        / float(sub2_stats["vertices"]),
        "adaptive_face_ratio_vs_sub2": float(adaptive_stats["faces"])
        / float(sub2_stats["faces"]),
        "reference": reference,
        "area_scale": float(info["area_scale"]),
        "threshold": float(info["threshold"]),
        "adaptive_max_represented_vertex_area": float(
            adaptive_stats["max_represented_area"]
        ),
        "sub2_max_represented_vertex_area": float(sub2_stats["max_represented_area"]),
        "represented_area_contract_pass": bool(
            float(adaptive_stats["max_represented_area"])
            <= float(info["threshold"]) * (1.0 + 1e-12)
        ),
        "surface_area_abs_difference": abs(
            float(adaptive_stats["total_area"]) - float(original_stats["total_area"])
        ),
        "iterations": len(info["history"]) - 1,
    }
    if not audit["represented_area_contract_pass"]:
        raise RuntimeError("GT-adaptive represented-area audit failed")
    return output, audit


def generate(args: argparse.Namespace) -> None:
    source_manifest = args.source_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    downstream_root = args.downstream_root.expanduser().resolve()
    upstream_root = args.upstream_root.expanduser().resolve()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    output_root.mkdir(parents=True, exist_ok=True)

    current = load_script(
        upstream_root / "scripts" / "prepare_sofa50_synthetic_current.py",
        "future2000_current_helper",
    )
    nested = load_script(
        upstream_root / "scripts" / "prepare_sofa50_nested_views_14_28_56.py",
        "future2000_nested_view_helper",
    )
    adaptive = load_script(
        upstream_root / "scripts" / "prepare_sofa50_query_resolution_ablation.py",
        "future2000_gt_adaptive_helper",
    )
    current.VIEW_COUNT = VIEW_COUNT
    deps = current._dependencies(downstream_root)
    from mlr.synthetic import look_at_world_to_camera, render_mesh_views_opengl

    deps["render_mesh_views_opengl"] = render_mesh_views_opengl
    if args.visibility_backend == "cuda":
        deps["compute_renderer_visibility"] = cuda_visibility_wrapper(upstream_root)

    records = records_from_manifest(source_manifest)
    selected = [
        record for index, record in enumerate(records) if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    output_records: list[dict[str, str]] = []
    oracle_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    reference_centers: np.ndarray | None = None

    for object_index, record in enumerate(selected, start=1):
        object_id = str(record["sample_id"])
        split = str(record["split"])
        print(
            f"[shard {args.shard_index}/{args.shard_count} {object_index}/{len(selected)}] "
            f"{object_id} split={split}",
            flush=True,
        )
        source = torch.load(resolve_record(source_manifest, record), map_location="cpu", weights_only=False)
        source, adaptive_audit = make_gt_adaptive_source(
            source,
            adaptive,
            reference=args.adaptive_reference,
            area_scale=args.adaptive_area_scale,
            max_iters=args.adaptive_max_iters,
            max_vertices=args.max_vertices,
        )
        adaptive_rows.append({"object_id": object_id, "split": split, **adaptive_audit})
        print(
            "  gt-adaptive "
            f"V={adaptive_audit['source_vertices']}->{adaptive_audit['adaptive_vertices']} "
            f"F={adaptive_audit['source_faces']}->{adaptive_audit['adaptive_faces']} "
            f"sub2_ratio={adaptive_audit['adaptive_vertex_ratio_vs_sub2']:.4f}",
            flush=True,
        )
        cameras_56, layout = nested._build_nested_cameras(
            base_cameras(source, deps["Camera"]),
            {"look_at_world_to_camera": look_at_world_to_camera, "Camera": deps["Camera"]},
        )
        cameras = cameras_56[:VIEW_COUNT]
        centers = np.asarray(layout["centers_56"], dtype=np.float64)[:VIEW_COUNT]
        if reference_centers is None:
            reference_centers = centers
        elif not np.allclose(reference_centers, centers, rtol=0.0, atol=1e-9):
            raise ValueError(f"{object_id}: nested camera layout differs across objects")
        base_images = source_image_paths(source, source_manifest.parent)
        added_images = render_added_views(
            source, cameras, object_id, output_root, deps, resume=args.resume
        )
        source28 = make_source_28(source, cameras, base_images, added_images)

        for variant_index in range(VARIANT_COUNT):
            relative = Path("prepared") / split / object_id / f"variant_{variant_index:02d}.pt"
            prepared_path = output_root / relative
            artifact = (
                output_root
                / "renderer_visibility"
                / split
                / object_id
                / f"variant_{variant_index:02d}.npz"
            )
            if args.resume and prepared_path.is_file() and artifact.is_file():
                sample = torch.load(prepared_path, map_location="cpu", weights_only=False)
                deps["validate_sample"](sample)
                oracle = current._oracle_from_saved_sample(sample, deps)
                print(f"  reuse variant={variant_index:02d}", flush=True)
            else:
                sample, oracle = current.build_current_sample(
                    source28,
                    object_id=object_id,
                    split=split,
                    variant_index=variant_index,
                    base_seed=args.seed,
                    perturb_std_h=args.perturb_std_h,
                    smooth_iterations=args.smooth_iterations,
                    neighbor_weight=args.neighbor_weight,
                    output_root=output_root,
                    source_root=source_manifest.parent,
                    visibility_backend=args.visibility_backend,
                    visibility_artifact=artifact,
                    deps=deps,
                )
                metadata = dict(sample["metadata"])
                metadata.update(
                    {
                        "dataset_family": FORMAT_VERSION,
                        "observation_view_count": VIEW_COUNT,
                        "camera_layout_version": layout["layout_version"],
                        "observation_renderer": "opengl_egl_base14_reused_plus_nested14",
                        "masks_persisted": False,
                        "depth_images_persisted": False,
                        "query_graph_variant": "gt_adaptive",
                        "adaptive_reference": args.adaptive_reference,
                        "adaptive_area_scale": args.adaptive_area_scale,
                    }
                )
                sample["metadata"] = metadata
                deps["save_prepared_sample"](sample, prepared_path)
            output_records.append(
                {"sample_id": str(sample["sample_id"]), "split": split, "path": relative.as_posix()}
            )
            oracle_rows.append(oracle)

    shard_dir = output_root / "shards"
    write_json(
        shard_dir / f"manifest_shard_{args.shard_index:02d}.json",
        {
            "format_version": FORMAT_VERSION,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "objects": len(selected),
            "samples": output_records,
        },
    )
    write_json(
        shard_dir / f"oracle_shard_{args.shard_index:02d}.json",
        {
            "format_version": FORMAT_VERSION,
            "all_target_contracts_pass": all(row["target_contract_pass"] for row in oracle_rows),
            "all_normalization_roundtrips_pass": all(
                row["normalization_roundtrip_pass"] for row in oracle_rows
            ),
            "samples": oracle_rows,
        },
    )
    write_json(
        shard_dir / f"adaptive_shard_{args.shard_index:02d}.json",
        {
            "format_version": FORMAT_VERSION,
            "query_graph_variant": "gt_adaptive",
            "objects": adaptive_rows,
        },
    )
    if reference_centers is not None:
        write_json(
            shard_dir / f"camera_layout_shard_{args.shard_index:02d}.json",
            {"view_count": VIEW_COUNT, "centers": reference_centers.tolist()},
        )
    print(json.dumps({"status": "passed", "objects": len(selected), "samples": len(output_records)}))


def merge(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    shard_dir = output_root / "shards"
    records: list[dict[str, str]] = []
    oracle_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    layouts = []
    for index in range(args.shard_count):
        manifest = read_json(shard_dir / f"manifest_shard_{index:02d}.json")
        oracle = read_json(shard_dir / f"oracle_shard_{index:02d}.json")
        adaptive = read_json(shard_dir / f"adaptive_shard_{index:02d}.json")
        records.extend(manifest["samples"])
        oracle_rows.extend(oracle["samples"])
        adaptive_rows.extend(adaptive["objects"])
        layouts.append(read_json(shard_dir / f"camera_layout_shard_{index:02d}.json"))
    if len(records) != OBJECT_COUNT * VARIANT_COUNT:
        raise ValueError(f"Expected {OBJECT_COUNT * VARIANT_COUNT} variants, found {len(records)}")
    if len(adaptive_rows) != OBJECT_COUNT:
        raise ValueError(f"Expected {OBJECT_COUNT} adaptive graph audits, found {len(adaptive_rows)}")
    adaptive_ids = [row["object_id"] for row in adaptive_rows]
    if len(set(adaptive_ids)) != OBJECT_COUNT:
        raise ValueError("Duplicate object IDs across adaptive graph audits")
    if not all(row["represented_area_contract_pass"] for row in adaptive_rows):
        raise ValueError("One or more GT-adaptive represented-area contracts failed")
    sample_ids = [record["sample_id"] for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs across shards")
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in EXPECTED_VARIANT_SPLITS
    }
    if split_counts != EXPECTED_VARIANT_SPLITS:
        raise ValueError(f"Unexpected variant split counts: {split_counts}")
    for record in records:
        if not (output_root / record["path"]).is_file():
            raise FileNotFoundError(output_root / record["path"])
    first_centers = np.asarray(layouts[0]["centers"], dtype=np.float64)
    if any(
        not np.allclose(first_centers, np.asarray(item["centers"]), rtol=0.0, atol=1e-9)
        for item in layouts[1:]
    ):
        raise ValueError("Camera layouts differ across shards")
    records.sort(key=lambda item: (item["split"], item["sample_id"]))
    write_json(
        output_root / "manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "dataset_role": "gt_adaptive_fixed_synthetic_current_28view_training",
            "object_count": OBJECT_COUNT,
            "object_split_counts": EXPECTED_OBJECT_SPLITS,
            "variant_split_counts": split_counts,
            "variants_per_object": VARIANT_COUNT,
            "object_level_split_enforced": True,
            "view_count": VIEW_COUNT,
            "target": "delta_target_raw=L_current@P_proxy",
            "query_graph": {
                "variant": "gt_adaptive",
                "criterion": "represented_vertex_area",
                "reference": adaptive_rows[0]["reference"],
                "area_scale": adaptive_rows[0]["area_scale"],
                "contract_pass": True,
            },
            "samples": records,
        },
    )
    write_json(
        output_root / "oracle_validation.json",
        {
            "format_version": FORMAT_VERSION,
            "sample_count": len(oracle_rows),
            "all_target_contracts_pass": all(row["target_contract_pass"] for row in oracle_rows),
            "all_normalization_roundtrips_pass": all(
                row["normalization_roundtrip_pass"] for row in oracle_rows
            ),
            "maximum_target_error": max(row["max_abs_Lc_Pproxy_target_error"] for row in oracle_rows),
            "maximum_roundtrip_error": max(row["max_abs_h2_roundtrip_error"] for row in oracle_rows),
        },
    )
    write_json(
        output_root / "gt_adaptive_validation.json",
        {
            "format_version": FORMAT_VERSION,
            "object_count": len(adaptive_rows),
            "all_represented_area_contracts_pass": True,
            "maximum_surface_area_abs_difference": max(
                row["surface_area_abs_difference"] for row in adaptive_rows
            ),
            "adaptive_vertex_count": {
                "minimum": min(row["adaptive_vertices"] for row in adaptive_rows),
                "maximum": max(row["adaptive_vertices"] for row in adaptive_rows),
            },
            "adaptive_face_count": {
                "minimum": min(row["adaptive_faces"] for row in adaptive_rows),
                "maximum": max(row["adaptive_faces"] for row in adaptive_rows),
            },
            "maximum_adaptive_vertex_ratio_vs_sub2": max(
                row["adaptive_vertex_ratio_vs_sub2"] for row in adaptive_rows
            ),
            "objects": sorted(adaptive_rows, key=lambda row: row["object_id"]),
        },
    )
    write_json(output_root / "nested_camera_layout_28.json", layouts[0])
    print(json.dumps({"status": "passed", "samples": len(records), "split_counts": split_counts}, indent=2))


def parser() -> argparse.ArgumentParser:
    home = Path.home()
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-manifest", type=Path, default=home / "future2000_compact/multiview_960/gt_query_manifest.json")
    value.add_argument(
        "--output-root",
        type=Path,
        default=home / "future2000_gt_adaptive_synthetic_current_28view_v2",
    )
    value.add_argument("--downstream-root", type=Path, default=home / "multiview-laplacian-refinement")
    value.add_argument("--upstream-root", type=Path, default=home / "data_prepare")
    value.add_argument("--shard-count", type=int, default=3)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--seed", type=int, default=7)
    value.add_argument("--perturb-std-h", type=float, default=0.15)
    value.add_argument("--smooth-iterations", type=int, default=5)
    value.add_argument("--neighbor-weight", type=float, default=0.65)
    value.add_argument("--visibility-backend", choices=("cuda", "opengl"), default="cuda")
    value.add_argument("--adaptive-reference", choices=("sub1", "sub2"), default="sub2")
    value.add_argument("--adaptive-area-scale", type=float, default=1.0)
    value.add_argument("--adaptive-max-iters", type=int, default=12)
    value.add_argument("--max-vertices", type=int, default=1_000_000)
    value.add_argument("--limit", type=int)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--merge-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.shard_count < 1 or args.shard_index < 0 or args.limit is not None and args.limit < 1:
        raise ValueError("Invalid shard or limit arguments")
    if args.adaptive_area_scale <= 0 or args.adaptive_max_iters < 1:
        raise ValueError("Invalid GT-adaptive arguments")
    if args.merge_only:
        merge(args)
    else:
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

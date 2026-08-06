#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh
from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.learned_laplacian.sample_io import (
    prepare_gt_query_sample_from_prepared,
    prepare_same_topology_sample,
)
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset


SPLITS = ("train", "validation", "test")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render sofa50 meshes and prepare lazy direct-GT-query samples without "
            "using any Thingi10K artifact."
        )
    )
    parser.add_argument("--sofa-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--views", type=int, default=14)
    parser.add_argument("--backend", choices=("cpu", "opengl", "cuda"), default="opengl")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.image_size < 1 or args.views < 1:
        raise ValueError("image-size and views must be positive.")
    if args.views != 14:
        raise ValueError("The cube-surface camera layout requires exactly 14 views.")
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    if any(value is not None and value < 0 for value in limits.values()):
        raise ValueError("Split limits must be non-negative.")

    sofa_root = args.sofa_root.expanduser().resolve()
    output_root = args.output_root.resolve()
    selection_path = sofa_root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split_ids = _selected_split_ids(selection, limits)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_root / "prepared_gt_query"
    render_root = output_root / "rendered"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    counts = {split: 0 for split in SPLITS}
    total = sum(len(values) for values in split_ids.values())
    completed = 0
    for split in SPLITS:
        for sample_id in split_ids[split]:
            completed += 1
            mesh_path = sofa_root / sample_id / "mesh.obj"
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing sofa mesh: {mesh_path}")
            destination = prepared_dir / f"{sample_id}.pt"
            if destination.is_file() and not args.overwrite:
                sample = load_prepared_sample(
                    destination,
                    materialize_images=False,
                    dataset_root=output_root,
                )
                _verify_sample(sample, sample_id, args.image_size, args.views, output_root)
                print(
                    f"[{completed}/{total}] reused {sample_id} split={split} "
                    f"vertices={sample['vertices'].shape[0]}",
                    flush=True,
                )
            else:
                sample = _prepare_one(
                    mesh_path,
                    sample_id,
                    split,
                    render_root / sample_id,
                    destination,
                    output_root,
                    image_size=args.image_size,
                    views=args.views,
                    backend=args.backend,
                )
                print(
                    f"[{completed}/{total}] prepared {sample_id} split={split} "
                    f"vertices={sample['vertices'].shape[0]}",
                    flush=True,
                )
            records.append(
                {
                    "sample_id": sample_id,
                    "path": destination.relative_to(output_root).as_posix(),
                    "split": split,
                }
            )
            counts[split] += 1

    manifest = {
        "format_version": "sofa50_gt_query_manifest_v1",
        "source": "3D-FUTURE sofa50",
        "source_root": str(sofa_root),
        "selection_file": str(selection_path),
        "query_training_mode": "gt_vertex_perturbation_v1",
        "rendering": {
            "views": args.views,
            "image_size": args.image_size,
            "trajectory": "cube_surface",
            "fov_degrees": 90.0,
            "cube_half_extent": 1.5,
            "backend": args.backend,
            "normalize_mesh": False,
        },
        "split_counts": counts,
        "samples": records,
    }
    manifest_path = output_root / "prepared_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} with split counts {counts}", flush=True)
    return 0


def _selected_split_ids(
    selection: dict[str, Any], limits: dict[str, int | None]
) -> dict[str, list[str]]:
    raw_splits = selection.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("selection.json must contain a splits object.")
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        values = raw_splits.get(split)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"selection.json split {split!r} must be a list of IDs.")
        limit = limits[split]
        selected = values if limit is None else values[:limit]
        duplicates = seen.intersection(selected)
        if duplicates:
            raise ValueError(f"Sofa split IDs are not disjoint: {sorted(duplicates)}")
        seen.update(selected)
        result[split] = list(selected)
    return result


def _prepare_one(
    mesh_path: Path,
    sample_id: str,
    split: str,
    render_dir: Path,
    destination: Path,
    output_root: Path,
    *,
    image_size: int,
    views: int,
    backend: str,
) -> dict[str, Any]:
    mesh = load_mesh(mesh_path).ensure_normals()
    rendered = generate_synthetic_dataset(
        mesh,
        render_dir,
        config=SyntheticRenderConfig(
            num_views=views,
            width=image_size,
            height=image_size,
            trajectory="cube_surface",
            fov_degrees=90.0,
            render_mode="lit",
            backend=backend,
            normalize_mesh=False,
            cube_half_extent=1.5,
            antialiasing="msaa4",
        ),
    )
    source = prepare_same_topology_sample(
        rendered.dataset_path,
        mesh_path,
        mesh_path,
        image_size=image_size,
        target_mode="edge_scale_normalized_laplacian",
        extra_metadata={
            "dataset_family": "sofa50",
            "source_dataset": "3D-FUTURE",
            "source_sample_id": sample_id,
            "source_split": split,
            "render_source": f"generated_{backend}_cube_surface_14view",
        },
    )
    source["sample_id"] = sample_id
    source.pop("images", None)
    source["image_paths"] = [
        path.resolve().relative_to(output_root).as_posix() for path in rendered.image_paths
    ]
    source["prepared_storage_format"] = "lazy_image_paths_v1"
    source["source_image_size"] = [image_size, image_size]
    source["prepared_image_size"] = image_size
    sample = prepare_gt_query_sample_from_prepared(
        source,
        output_path=destination,
        target_mode="edge_scale_normalized_laplacian",
    )
    _verify_sample(sample, sample_id, image_size, views, output_root)
    return sample


def _verify_sample(
    sample: dict[str, Any],
    sample_id: str,
    image_size: int,
    views: int,
    output_root: Path,
) -> None:
    if sample["sample_id"] != sample_id:
        raise ValueError(f"Prepared sample ID mismatch: {sample['sample_id']} != {sample_id}")
    if sample.get("prepared_storage_format") != "lazy_image_paths_v1":
        raise ValueError("Sofa sample is not using lazy image-path storage.")
    if "images" in sample:
        raise ValueError("Lazy sofa sample unexpectedly embeds decoded images.")
    if sample.get("prepared_image_size") != image_size:
        raise ValueError("Prepared sofa image size does not match the requested size.")
    if len(sample.get("image_paths", [])) != views:
        raise ValueError("Prepared sofa sample has the wrong number of views.")
    missing = [
        value
        for value in sample["image_paths"]
        if not (Path(value) if Path(value).is_absolute() else output_root / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Prepared sofa sample references missing RGB images: {missing}")
    if sample["initial_laplacian"].count_nonzero().item() != 0:
        raise ValueError("GT-query initial_laplacian must be zero to prevent target leakage.")


if __name__ == "__main__":
    raise SystemExit(main())

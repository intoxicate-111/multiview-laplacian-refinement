from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera
from mlr.datasets import load_reconstruction_input
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.projection import project_vertices
from mlr.learned_laplacian.sample_io import (
    corrupt_same_topology_mesh,
    prepare_same_topology_sample,
)
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a same-topology Stanford Bunny experiment.")
    parser.add_argument("--gt-mesh", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reuse-dataset", type=Path, help="Optional clean Bunny dataset to subset/resample.")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--smoothing-iters", type=int, default=2)
    parser.add_argument("--smoothing-strength", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=["cpu", "opengl", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.views < 1 or args.image_size < 1:
        raise ValueError("views and image-size must be positive.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    gt_mesh = load_mesh(args.gt_mesh).ensure_normals()
    coarse_mesh = corrupt_same_topology_mesh(
        gt_mesh,
        noise_std=args.noise_std,
        smoothing_iters=args.smoothing_iters,
        smoothing_strength=args.smoothing_strength,
        seed=args.seed,
    )
    gt_path = output_root / "gt.obj"
    coarse_path = output_root / "coarse.obj"
    save_mesh(gt_mesh, gt_path)
    save_mesh(coarse_mesh, coarse_path)

    inputs_dir = output_root / "inputs"
    if args.reuse_dataset is not None:
        dataset_path = _resample_dataset(
            args.reuse_dataset,
            inputs_dir,
            gt_mesh,
            views=args.views,
            image_size=args.image_size,
            seed=args.seed,
        )
        render_source = "resampled_existing_sphere_dataset"
    else:
        rendered = generate_synthetic_dataset(
            gt_mesh,
            inputs_dir,
            config=SyntheticRenderConfig(
                num_views=args.views,
                width=args.image_size,
                height=args.image_size,
                trajectory="sphere",
                render_mode="lit",
                backend=args.backend,
                normalize_mesh=False,
            ),
        )
        dataset_path = rendered.dataset_path
        render_source = f"generated_{args.backend}"

    corruption = {
        "noise_std": float(args.noise_std),
        "smoothing_iters": int(args.smoothing_iters),
        "smoothing_strength": float(args.smoothing_strength),
        "seed": int(args.seed),
    }
    rendering = {
        "views": int(args.views),
        "width": int(args.image_size),
        "height": int(args.image_size),
        "trajectory": "sphere",
        "render_mode": "lit",
        "source": render_source,
    }
    sample_path = output_root / "prepared_sample.pt"
    sample = prepare_same_topology_sample(
        dataset_path,
        coarse_path,
        gt_path,
        output_path=sample_path,
        seed=args.seed,
        extra_metadata={"corruption": corruption, "rendering": rendering},
    )
    projection_metrics = _write_projection_debug(sample, inputs_dir, output_root)
    preparation = {
        "gt_source": str(args.gt_mesh),
        "gt_vertices": gt_mesh.num_vertices,
        "gt_faces": gt_mesh.num_faces,
        "corruption": corruption,
        "rendering": rendering,
        "projection": projection_metrics,
        "dataset": str(dataset_path),
        "sample": str(sample_path),
    }
    (output_root / "preparation.json").write_text(
        json.dumps(preparation, indent=2), encoding="utf-8"
    )
    print(json.dumps(preparation, indent=2))
    return 0


def _resample_dataset(
    source_dataset: Path,
    output_dir: Path,
    gt_mesh,
    views: int,
    image_size: int,
    seed: int,
) -> Path:
    source = load_reconstruction_input(source_dataset)
    if source.gt_mesh_path is None:
        raise ValueError("The reused dataset must identify its clean GT mesh.")
    source_mesh = load_mesh(source.gt_mesh_path)
    if source_mesh.vertices.shape != gt_mesh.vertices.shape or not np.allclose(
        source_mesh.vertices, gt_mesh.vertices, atol=1e-7
    ) or not np.array_equal(source_mesh.faces, gt_mesh.faces):
        raise ValueError("Reused dataset mesh does not match --gt-mesh exactly.")
    if views > len(source.image_paths):
        raise ValueError("Requested more views than the reused dataset contains.")
    indices = np.floor(np.arange(views) * len(source.image_paths) / views).astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    depth_dir = output_dir / "depth"
    for directory in (image_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_payload = json.loads(source_dataset.read_text(encoding="utf-8"))
    source_root = source_dataset.parent
    camera_payload = []
    image_paths = []
    mask_paths = []
    depth_paths = []
    for output_index, source_index in enumerate(indices):
        image = Image.open(source.image_paths[source_index]).convert("RGB")
        old_width, old_height = image.size
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image_path = image_dir / f"{output_index:04d}.png"
        image.save(image_path)
        image_paths.append(str(image_path.relative_to(output_dir)).replace("\\", "/"))

        if source.mask_paths is not None:
            mask = Image.open(source.mask_paths[source_index]).convert("L")
            mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
            mask_path = mask_dir / f"{output_index:04d}.png"
            mask.save(mask_path)
            mask_paths.append(str(mask_path.relative_to(output_dir)).replace("\\", "/"))

        source_depths = source_payload.get("depth_paths")
        if source_depths:
            depth = np.load(source_root / source_depths[source_index])
            y = np.minimum((np.arange(image_size) * depth.shape[0] / image_size).astype(int), depth.shape[0] - 1)
            x = np.minimum((np.arange(image_size) * depth.shape[1] / image_size).astype(int), depth.shape[1] - 1)
            depth_path = depth_dir / f"{output_index:04d}.npy"
            np.save(depth_path, depth[np.ix_(y, x)])
            depth_paths.append(str(depth_path.relative_to(output_dir)).replace("\\", "/"))

        camera = source.cameras[source_index]
        intrinsics = camera.intrinsics.copy()
        intrinsics[0, :] *= image_size / old_width
        intrinsics[1, :] *= image_size / old_height
        camera_payload.append(
            {
                "intrinsics": intrinsics.tolist(),
                "rotation": camera.rotation.tolist(),
                "translation": camera.translation.tolist(),
                "image_size": [image_size, image_size],
                "name": f"view_{output_index:04d}",
                "source_view_index": int(source_index),
            }
        )
    save_mesh(gt_mesh, output_dir / "mesh.obj")
    (output_dir / "cameras.json").write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")
    payload = {
        "mesh_path": "mesh.obj",
        "source_mesh_path": str(source.gt_mesh_path),
        "cameras_path": "cameras.json",
        "image_paths": image_paths,
        "mask_paths": mask_paths or None,
        "depth_paths": depth_paths or None,
        "config": {
            "num_views": views,
            "width": image_size,
            "height": image_size,
            "trajectory": "sphere",
            "render_mode": "lit",
            "derived_from": str(source_dataset),
            "selected_view_indices": indices.tolist(),
            "seed": seed,
        },
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dataset_path


def _write_projection_debug(sample: dict, inputs_dir: Path, output_root: Path) -> dict:
    projection = project_vertices(
        sample["vertices"],
        sample["intrinsics"],
        sample["extrinsics"],
        image_size=tuple(sample["images"].shape[-2:]),
    )
    visibility = sample.get("visibility")
    valid = projection.valid if visibility is None else projection.valid & visibility
    per_vertex_views = valid.sum(dim=0)
    metrics = {
        "in_frame_fraction": float(projection.valid.float().mean().item()),
        "mask_valid_fraction": float(valid.float().mean().item()),
        "zero_valid_view_vertices": int((per_vertex_views == 0).sum().item()),
        "zero_valid_view_fraction": float((per_vertex_views == 0).float().mean().item()),
        "mean_valid_views_per_vertex": float(per_vertex_views.float().mean().item()),
    }
    image = Image.open(inputs_dir / "images" / "0000.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    pixels = projection.pixels[0].detach().cpu().numpy()
    in_frame = projection.valid[0].detach().cpu().numpy()
    visible = valid[0].detach().cpu().numpy()
    stride = max(len(pixels) // 6000, 1)
    for index in range(0, len(pixels), stride):
        if not in_frame[index]:
            continue
        x, y = pixels[index]
        colour = (20, 255, 80) if visible[index] else (255, 70, 50)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=colour)
    image.save(output_root / "projection_debug.png")
    (output_root / "projection_debug.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())

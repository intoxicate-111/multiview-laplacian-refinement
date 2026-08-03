from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera, Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.learned_laplacian.projection import project_vertices
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view


MODES = (
    ("coarse_only", "coarse_only", False),
    ("coarse_plus_multiview", "coarse_plus_multiview", False),
    ("zero_images", "coarse_plus_multiview", True),
)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run matched Stanford Bunny overfit ablations.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparison = {"sample": str(args.sample), "config": str(args.config), "modes": {}}
    for directory_name, input_mode, zero_images in MODES:
        output_dir = args.output_root / directory_name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "overfit_single_object.py"),
            "--sample",
            str(args.sample),
            "--config",
            str(args.config),
            "--output-dir",
            str(output_dir),
            "--input-mode",
            input_mode,
            "--device",
            args.device,
        ]
        if zero_images:
            command.append("--zero-images")
        if args.steps is not None:
            command.extend(("--steps", str(args.steps)))
        print(f"Running {directory_name}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics["high_error_region"] = _high_error_region(
            output_dir / "laplacian_error.npy", args.sample
        )
        _write_error_projection(
            output_dir / "laplacian_error.npy",
            args.sample,
            output_dir / "laplacian_error_projection.png",
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison["modes"][directory_name] = metrics

    first = args.output_root / MODES[0][0]
    shutil.copyfile(first / "coarse.obj", args.output_root / "coarse.obj")
    shutil.copyfile(first / "oracle_refined.obj", args.output_root / "oracle_refined.obj")
    sample = load_prepared_sample(args.sample)
    gt_mesh = Mesh(sample["gt_vertices"].numpy(), sample["gt_faces"].numpy()).ensure_normals()
    save_mesh(gt_mesh, args.output_root / "gt.obj")
    comparison["common_geometry"] = comparison["modes"]["coarse_only"]["geometry"]
    _write_comparison_render(args.output_root, sample)
    comparison["training_runtime_seconds_total"] = sum(
        mode["training"]["runtime_seconds"] for mode in comparison["modes"].values()
    )
    comparison["wall_runtime_seconds"] = time.perf_counter() - started
    comparison["comparison_render"] = str(args.output_root / "comparison_render.png")
    (args.output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"Saved comparison to {args.output_root / 'comparison.json'}")
    return 0


def _high_error_region(error_path: Path, sample_path: Path) -> dict:
    errors = np.load(error_path)
    sample = load_prepared_sample(sample_path)
    vertices = sample["vertices"].numpy()
    threshold = float(np.quantile(errors, 0.95))
    mask = errors >= threshold
    selected = vertices[mask]
    top = np.argsort(errors)[-20:][::-1]
    return {
        "p95_error": threshold,
        "count_at_or_above_p95": int(mask.sum()),
        "region_min": selected.min(axis=0).tolist(),
        "region_max": selected.max(axis=0).tolist(),
        "region_mean": selected.mean(axis=0).tolist(),
        "top_vertex_indices": top.tolist(),
        "top_vertex_errors": errors[top].tolist(),
    }


def _write_error_projection(error_path: Path, sample_path: Path, output_path: Path) -> None:
    errors = np.load(error_path)
    sample = load_prepared_sample(sample_path)
    projection = project_vertices(
        sample["vertices"],
        sample["intrinsics"],
        sample["extrinsics"],
        image_size=tuple(sample["images"].shape[-2:]),
        visibility=sample.get("visibility"),
    )
    image_array = (
        sample["images"][0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
    )
    image = Image.fromarray(image_array, mode="RGB")
    draw = ImageDraw.Draw(image)
    pixels = projection.pixels[0].numpy()
    valid = projection.valid[0].numpy()
    low, high = np.quantile(errors, [0.05, 0.95])
    scale = max(float(high - low), 1e-12)
    stride = max(len(errors) // 10000, 1)
    for index in range(0, len(errors), stride):
        if not valid[index]:
            continue
        value = float(np.clip((errors[index] - low) / scale, 0.0, 1.0))
        colour = (int(255 * value), int(255 * (1.0 - abs(2.0 * value - 1.0))), int(255 * (1.0 - value)))
        x, y = pixels[index]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=colour)
    image.save(output_path)


def _write_comparison_render(output_root: Path, sample: dict) -> None:
    extrinsic = sample["extrinsics"][0].numpy()
    height, width = sample["images"].shape[-2:]
    camera = Camera(
        intrinsics=sample["intrinsics"][0].numpy(),
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(width, height),
    )
    paths = (
        ("GT", output_root / "gt.obj"),
        ("Coarse", output_root / "coarse.obj"),
        ("Oracle", output_root / "oracle_refined.obj"),
        ("Geometry", output_root / "coarse_only" / "predicted_refined.obj"),
        ("Geometry + RGB", output_root / "coarse_plus_multiview" / "predicted_refined.obj"),
        ("Zero images", output_root / "zero_images" / "predicted_refined.obj"),
    )
    config = SyntheticRenderConfig(
        width=width,
        height=height,
        render_mode="lit",
        normalize_mesh=False,
    )
    canvas = Image.new("RGB", (3 * width, 2 * (height + 18)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(paths):
        rgb, _, _ = render_mesh_view(load_mesh(path), camera, config)
        x = (index % 3) * width
        y = (index // 3) * (height + 18)
        canvas.paste(Image.fromarray(rgb), (x, y + 18))
        draw.text((x + 4, y + 2), label, fill=(240, 240, 240))
    canvas.save(output_root / "comparison_render.png")


if __name__ == "__main__":
    raise SystemExit(main())

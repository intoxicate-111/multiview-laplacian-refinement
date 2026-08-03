from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.dataset import load_prepared_sample


MODES = (
    ("coarse_only", "coarse_only", False),
    ("coarse_plus_multiview", "coarse_plus_multiview", False),
    ("zero_images", "coarse_plus_multiview", True),
)


def main() -> int:
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
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison["modes"][directory_name] = metrics

    first = args.output_root / MODES[0][0]
    shutil.copyfile(first / "coarse.obj", args.output_root / "coarse.obj")
    shutil.copyfile(first / "oracle_refined.obj", args.output_root / "oracle_refined.obj")
    sample = load_prepared_sample(args.sample)
    gt_mesh = Mesh(sample["gt_vertices"].numpy(), sample["gt_faces"].numpy()).ensure_normals()
    save_mesh(gt_mesh, args.output_root / "gt.obj")
    comparison["common_geometry"] = comparison["modes"]["coarse_only"]["geometry"]
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


if __name__ == "__main__":
    raise SystemExit(main())

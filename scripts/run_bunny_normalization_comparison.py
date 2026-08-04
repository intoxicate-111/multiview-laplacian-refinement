#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODES = (
    ("raw_geometry", "coarse_only", "raw"),
    ("normalized_geometry", "coarse_only", "normalized"),
    ("raw_multiview", "coarse_plus_multiview", "raw"),
    ("normalized_multiview", "coarse_plus_multiview", "normalized"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run four matched cleaned Bunny target modes.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--raw-config", required=True, type=Path)
    parser.add_argument("--normalized-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse a mode only when its metrics.json exists; partial modes restart from step zero.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for label, config_path in (("raw", args.raw_config), ("normalized", args.normalized_config)):
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "overfit_single_object.py"),
                "--sample", str(args.sample),
                "--config", str(config_path),
                "--output-dir", str(args.output_root / f"diagnostics_{label}"),
                "--diagnostics-output", str(args.output_root / f"pre_training_diagnostics_{label}.json"),
                "--diagnostics-only",
            ]
        )

    metrics_by_mode = {}
    for directory, input_mode, target_kind in MODES:
        config_path = args.raw_config if target_kind == "raw" else args.normalized_config
        output_dir = args.output_root / directory
        metrics_path = output_dir / "metrics.json"
        if args.resume_completed and metrics_path.exists():
            print(f"Reusing completed mode: {directory}", flush=True)
        else:
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "overfit_single_object.py"),
                    "--sample", str(args.sample),
                    "--config", str(config_path),
                    "--output-dir", str(output_dir),
                    "--input-mode", input_mode,
                    "--device", args.device,
                    "--steps", str(args.steps),
                ]
            )
        metrics_by_mode[directory] = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

    _copy_outputs(args.output_root)
    preparation_path = args.output_root / "preparation.json"
    preparation = (
        json.loads(preparation_path.read_text(encoding="utf-8"))
        if preparation_path.exists()
        else {}
    )
    diagnostics = {
        label: json.loads(
            (args.output_root / f"pre_training_diagnostics_{label}.json").read_text(encoding="utf-8")
        )
        for label in ("raw", "normalized")
    }
    isolated_path = args.output_root / "isolated_vertex_diagnostics.json"
    isolated = (
        json.loads(isolated_path.read_text(encoding="utf-8"))
        if isolated_path.exists()
        else {}
    )
    cleaning = dict(preparation.get("cleaning", {}))
    raw_diagnostics = diagnostics["raw"]
    scale_statistics = raw_diagnostics["local_edge_scale"]
    cleaning.update(
        {
            "edge_count": raw_diagnostics["graph"]["unique_undirected_edge_count"],
            "isolated_vertices_before_cleaning": isolated.get("summary", {}).get(
                "unreferenced_vertices"
            ),
            "isolated_vertices_after_cleaning": raw_diagnostics["graph"]["isolated_vertices"],
        }
    )
    comparison = {
        "sample": str(args.sample),
        "raw_config": str(args.raw_config),
        "normalized_config": str(args.normalized_config),
        "device": args.device,
        "steps": args.steps,
        "cleaning": cleaning,
        "target_statistics": {
            "edge_length": {
                key: value for key, value in scale_statistics.items() if key.endswith("_h")
            },
            "edge_scale": {
                key: value for key, value in scale_statistics.items() if key.endswith("_h2")
            },
            "raw_magnitude": raw_diagnostics["raw_target_magnitude"],
            "normalized_magnitude": raw_diagnostics["normalized_target_magnitude"],
            "finite_checks": {
                "raw": raw_diagnostics["finite_values"],
                "normalized": diagnostics["normalized"]["finite_values"],
            },
        },
        "diagnostics": diagnostics,
        "modes": metrics_by_mode,
    }
    (args.output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_root / "comparison.csv", metrics_by_mode)
    if not args.skip_visualization:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "visualize_bunny_normalization.py"),
                "--sample", str(args.sample),
                "--output-root", str(args.output_root),
            ]
        )
    return 0


def _run(command: list[str]) -> None:
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _copy_outputs(root: Path) -> None:
    copies = {
        root / "raw_geometry" / "predicted_refined.obj": root / "raw_geometry_refined.obj",
        root / "normalized_geometry" / "predicted_refined.obj": root / "normalized_geometry_refined.obj",
        root / "raw_multiview" / "predicted_refined.obj": root / "raw_multiview_refined.obj",
        root / "normalized_multiview" / "predicted_refined.obj": root / "normalized_multiview_refined.obj",
        root / "raw_geometry" / "oracle_refined.obj": root / "oracle_refined.obj",
        root / "raw_geometry" / "delta_target.npy": root / "raw_delta_target.npy",
        root / "raw_geometry" / "delta_pred.npy": root / "raw_delta_pred.npy",
        root / "normalized_geometry" / "delta_hat_target.npy": root / "normalized_delta_target.npy",
        root / "normalized_geometry" / "delta_hat_pred.npy": root / "normalized_delta_pred.npy",
        root / "normalized_geometry" / "delta_pred.npy": root / "normalized_recovered_raw_delta.npy",
        root / "normalized_geometry" / "local_edge_length.npy": root / "local_edge_length.npy",
        root / "normalized_geometry" / "local_edge_scale.npy": root / "local_edge_scale.npy",
    }
    for source, destination in copies.items():
        shutil.copyfile(source, destination)


def _write_csv(path: Path, metrics_by_mode: dict) -> None:
    fields = [
        "mode", "target_mode", "initial_loss", "final_loss", "best_loss", "best_step",
        "runtime_seconds", "raw_mse", "normalized_mse", "p2s_mean", "p2s_median",
        "p2s_max", "chamfer", "target_position_rmse", "normal_consistency", "bbox_ratio",
        "all_finite", "collapsed_or_exploded",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, metrics in metrics_by_mode.items():
            training = metrics["training"]
            geometry = metrics["geometry"]["predicted"]
            writer.writerow(
                {
                    "mode": mode,
                    "target_mode": metrics["target_mode"],
                    "initial_loss": training["initial_loss"],
                    "final_loss": training["final_loss"],
                    "best_loss": training["best_loss"],
                    "best_step": training["best_step"],
                    "runtime_seconds": training["runtime_seconds"],
                    "raw_mse": metrics["laplacian_prediction_raw"]["mse"],
                    "normalized_mse": metrics["laplacian_prediction_normalized"]["mse"],
                    "p2s_mean": geometry["point_to_surface_mean"],
                    "p2s_median": geometry["point_to_surface_median"],
                    "p2s_max": geometry["point_to_surface_max"],
                    "chamfer": geometry["chamfer"],
                    "target_position_rmse": geometry["target_position_rmse"],
                    "normal_consistency": geometry["normal_consistency"],
                    "bbox_ratio": geometry["bbox_diagonal_ratio_to_coarse"],
                    "all_finite": geometry["all_finite"],
                    "collapsed_or_exploded": geometry["collapsed_or_exploded"],
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())

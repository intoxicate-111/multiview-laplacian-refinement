#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.prediction_visualizer import (
    VisualizationOptions,
    discover_predictions,
    discover_run_metadata,
    prediction_listing,
    visualize_prediction_sample,
    visualize_prediction_split,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct and visualize saved learned-Laplacian predictions."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Exact dataset manifest for a legacy run that did not save one.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--operator-type")
    parser.add_argument("--lambda-lap", type=float)
    parser.add_argument("--lambda-anchor", type=float)
    parser.add_argument("--lambda-edge", type=float)
    parser.add_argument("--num-iters", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-refinement", action="store_true")
    args = parser.parse_args()

    metadata = discover_run_metadata(args.run_dir, manifest_override=args.manifest)
    records = discover_predictions(metadata.run_dir, args.split)
    output_dir = args.output_dir or metadata.run_dir / "visualizations" / args.split
    listing = prediction_listing(metadata, args.split, records, output_dir)
    if args.list or (args.sample_id is None and not args.all):
        print(json.dumps({"available_predictions": listing}, indent=2))
        if not args.list:
            print("Specify --sample-id <id> or --all to generate visualizations.")
        return 0
    options = VisualizationOptions(
        output_dir=output_dir,
        camera_index=args.camera_index,
        image_size=args.image_size,
        columns=args.columns,
        operator_type=args.operator_type,
        lambda_lap=args.lambda_lap,
        lambda_anchor=args.lambda_anchor,
        lambda_edge=args.lambda_edge,
        num_iters=args.num_iters,
        learning_rate=args.learning_rate,
        device=args.device,
        overwrite=args.overwrite,
        skip_render=args.skip_render,
        skip_refinement=args.skip_refinement,
    )
    if args.all:
        batch = visualize_prediction_split(metadata, args.split, list(records.values()), options)
        print(json.dumps(batch, indent=2))
        return 1 if batch["failed"] else 0
    if args.sample_id not in records:
        parser.error(
            f"No prediction for sample {args.sample_id!r} in split {args.split!r}. "
            f"Available IDs: {', '.join(records) or '(none)'}"
        )
    summary = visualize_prediction_sample(
        metadata, args.split, records[args.sample_id], options
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

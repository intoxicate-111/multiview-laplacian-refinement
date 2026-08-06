#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.single_mesh_overfit import run_single_mesh_overfit


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled loss ablation on one mesh.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-id")
    parser.add_argument("--steps", default=1000, type=int)
    parser.add_argument("--log-every", default=25, type=int)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--magnitude-weight-lambda", default=4.0, type=float)
    parser.add_argument("--direction-lambda", default=1.0, type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    args = parser.parse_args()
    summary = run_single_mesh_overfit(
        args.manifest,
        args.config,
        args.output_dir,
        split=args.split,
        sample_id=args.sample_id,
        steps=args.steps,
        log_every=args.log_every,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        magnitude_weight_lambda=args.magnitude_weight_lambda,
        direction_lambda=args.direction_lambda,
        overwrite=args.overwrite,
        skip_reconstruction=args.skip_reconstruction,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "sample_id": summary["sample_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

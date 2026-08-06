#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.visibility_ablation import run_renderer_visibility_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint renderer visibility ablation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--visualizations", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_renderer_visibility_ablation(
        args.run_dir,
        args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        split=args.split,
        overwrite=args.overwrite,
        visualizations=args.visualizations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.controlled_screening import run_screening_arm
from mlr.learned_laplacian.geometry_aware_sampling import analyze_geometry_aware_sampling


ARM_MAP = {
    "G2": "strong_importance_0001",
    "G3": "smooth_importance_0001",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sofa50 geometry-aware sampling controls.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", choices=tuple(ARM_MAP))
    parser.add_argument("--max-optimizer-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze_geometry_aware_sampling(
            args.baseline_root, args.output_root, args.manifest
        )
        return 0
    if args.arm is None or args.base_config is None:
        parser.error("--arm and --base-config are required unless --analyze is used")
    run_screening_arm(
        args.manifest,
        args.base_config,
        args.output_root,
        ARM_MAP[args.arm],
        max_optimizer_steps=args.max_optimizer_steps,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

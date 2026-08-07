#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.controlled_screening import run_screening_arm
from mlr.learned_laplacian.image_resolution_ablation import analyze_image_resolution_ablation


ARM_MAP = {
    "F0": "image_resolution_f0",
    "F1": "image_resolution_f1",
    "F2": "image_resolution_f2",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sofa50 feature-resolution ablation.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", choices=tuple(ARM_MAP))
    parser.add_argument("--max-optimizer-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        if args.manifest is None:
            parser.error("--manifest is required with --analyze")
        analyze_image_resolution_ablation(
            args.output_root,
            manifest_path=args.manifest,
            device=args.device,
        )
        return 0
    if args.arm is None or args.manifest is None or args.base_config is None:
        parser.error("--arm, --manifest and --base-config are required unless --analyze is used")
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

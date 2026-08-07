#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.controlled_screening import (
    ARM_CHOICES,
    analyze_screening,
    run_screening_arm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Sofa50 controlled screening arm.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", choices=ARM_CHOICES)
    parser.add_argument("--max-optimizer-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze_screening(args.output_root)
        return 0
    if args.arm is None or args.manifest is None or args.base_config is None:
        parser.error("--arm, --manifest and --base-config are required unless --analyze is used")
    run_screening_arm(
        args.manifest,
        args.base_config,
        args.output_root,
        args.arm,
        max_optimizer_steps=args.max_optimizer_steps,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

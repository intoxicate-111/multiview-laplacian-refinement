#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.synthetic_current_stage2_adaptation import (
    evaluate_stage2_arm,
    merge_stage2_arm_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--stage2-manifest")
    parser.add_argument("--baseline-csv")
    parser.add_argument("--arm-run-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--arm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-arms", action="store_true")
    parser.add_argument("--continuation-steps", type=int, default=20_000)
    args = parser.parse_args()
    if args.merge_arms:
        merge_stage2_arm_results(
            args.experiment_dir, continuation_steps=args.continuation_steps
        )
        return
    required = {
        "stage2_manifest": args.stage2_manifest,
        "baseline_csv": args.baseline_csv,
        "arm_run_dir": args.arm_run_dir,
        "output_dir": args.output_dir,
        "arm": args.arm,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("arm evaluation requires: " + ", ".join(missing))
    evaluate_stage2_arm(
        args.stage2_manifest,
        args.baseline_csv,
        args.arm_run_dir,
        args.output_dir,
        arm=args.arm,
        device=args.device,
    )


if __name__ == "__main__":
    main()

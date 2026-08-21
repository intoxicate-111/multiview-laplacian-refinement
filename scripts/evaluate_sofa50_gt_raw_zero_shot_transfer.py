#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlr.learned_laplacian.gt_raw_zero_shot_transfer import (
    run_gt_raw_zero_shot_transfer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--normalized-gt-run", type=Path, required=True)
    parser.add_argument("--current-hf-run", type=Path, required=True)
    parser.add_argument("--current-hf-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_gt_raw_zero_shot_transfer(
        args.new_run,
        args.gt_manifest,
        args.current_manifest,
        args.normalized_gt_run,
        args.current_hf_run,
        args.current_hf_analysis,
        args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

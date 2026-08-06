#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.image_ablation import run_image_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-checkpoint RGB image ablation.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--coarse-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    args = parser.parse_args()
    summary = run_image_ablation(
        args.run_dir,
        args.coarse_manifest,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_reconstruction=args.skip_reconstruction,
    )
    print(json.dumps({"output_dir": str(args.output_dir or args.run_dir / "image_ablation"), "queries": list(summary["queries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

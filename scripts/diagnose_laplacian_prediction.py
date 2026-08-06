#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.diagnostics import run_laplacian_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose baselines, target magnitudes, and reconstruction for a run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    args = parser.parse_args()

    summary = run_laplacian_diagnostics(
        args.run_dir,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_reconstruction=args.skip_reconstruction,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

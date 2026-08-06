#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.scaling_diagnostics import run_single_checkpoint_image_ablation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_single_checkpoint_image_ablation(args.checkpoint, args.manifest, args.output_dir, split=args.split, sample_id=args.sample_id, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

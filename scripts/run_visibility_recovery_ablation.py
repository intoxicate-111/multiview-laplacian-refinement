#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.visibility_recovery_ablation import (
    run_visibility_recovery_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablate hard any-view visibility weights in Laplacian recovery."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expanded-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_visibility_recovery_ablation(
        args.run_dir,
        args.expanded_manifest,
        args.config,
        args.output_dir,
        split=args.split,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

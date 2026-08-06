#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.expanded_visibility_evaluation import (
    run_expanded_renderer_visibility_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate four renderer visibility masks on expanded queries."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expanded-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reconstruction-iters", type=int, default=200)
    args = parser.parse_args()
    run_expanded_renderer_visibility_evaluation(
        args.run_dir,
        args.expanded_manifest,
        args.output_dir,
        split=args.split,
        device=args.device,
        seed=args.seed,
        reconstruction_iters=args.reconstruction_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

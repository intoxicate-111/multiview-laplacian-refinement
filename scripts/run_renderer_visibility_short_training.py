#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.renderer_visibility_training import (
    run_renderer_visibility_short_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired short training for renderer-native visibility conditions."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mesh-counts", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["frustum_only", "backface_and_occlusion"],
    )
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_renderer_visibility_short_training(
        args.manifest,
        args.config,
        args.output_dir,
        mesh_counts=args.mesh_counts,
        conditions=args.conditions,
        optimizer_steps=args.optimizer_steps,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

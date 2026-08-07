#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.visibility_convergence import (
    run_visibility_convergence_study,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controlled Sofa50 renderer-visibility convergence study."
    )
    parser.add_argument("--gt-query-manifest", required=True)
    parser.add_argument("--expanded-manifest", required=True)
    parser.add_argument(
        "--config",
        default="configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json",
    )
    parser.add_argument(
        "--recovery-config",
        default="configs/learned_laplacian/visibility_recovery_sofa50_expanded.json",
    )
    parser.add_argument("--mesh-count", type=int, default=16)
    parser.add_argument("--optimizer-steps", type=int, default=2000)
    parser.add_argument(
        "--checkpoint-steps",
        nargs="+",
        type=int,
        default=[0, 100, 250, 500, 1000, 2000],
    )
    parser.add_argument("--expanded-split", default="validation")
    parser.add_argument(
        "--visibility-key",
        default="visibility_backface_and_occlusion",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_visibility_convergence_study(
        args.gt_query_manifest,
        args.expanded_manifest,
        args.config,
        args.recovery_config,
        args.output_dir,
        mesh_count=args.mesh_count,
        optimizer_steps=args.optimizer_steps,
        checkpoint_steps=args.checkpoint_steps,
        visibility_key=args.visibility_key,
        expanded_split=args.expanded_split,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

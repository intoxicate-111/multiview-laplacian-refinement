#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlr.learned_laplacian.perturbed_scale_sweep import (
    DEFAULT_SCALES,
    run_perturbed_scale_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Sofa50 step-2000 control/perturbed expanded-query "
            "predicted-delta scale sweep and 960px visualization study."
        )
    )
    parser.add_argument("--expanded-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--sofa-models-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--perturbation-config", type=Path)
    parser.add_argument("--visibility-backend", choices=("cpu", "opengl"), default="opengl")
    parser.add_argument("--render-backend", choices=("cpu", "opengl"), default="opengl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    perturbation = None
    if args.perturbation_config is not None:
        perturbation = json.loads(args.perturbation_config.read_text(encoding="utf-8"))
        perturbation = perturbation.get("coarse_perturbation", perturbation)
    run_perturbed_scale_sweep(
        args.expanded_manifest,
        args.checkpoint,
        args.model_config,
        args.recovery_config,
        args.sofa_models_root,
        args.output_dir,
        split=args.split,
        scales=args.scales,
        perturbation=perturbation,
        visibility_backend=args.visibility_backend,
        render_backend=args.render_backend,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

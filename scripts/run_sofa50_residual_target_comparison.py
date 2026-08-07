from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlr.learned_laplacian.residual_target_comparison import run_residual_target_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the short Sofa50 residual-target comparison.")
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/learned_laplacian/sofa50_residual_target_comparison"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", default="opengl")
    args = parser.parse_args()
    summary = run_residual_target_comparison(
        args.source_run,
        args.output_dir,
        steps=args.steps,
        device=args.device,
        render_backend=args.render_backend,
    )
    print(json.dumps(summary["aggregates"], indent=2))


if __name__ == "__main__":
    main()

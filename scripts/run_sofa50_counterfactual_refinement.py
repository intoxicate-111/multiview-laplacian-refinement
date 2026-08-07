from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlr.learned_laplacian.counterfactual_refinement import run_counterfactual_refinement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled Sofa50 current-geometry/RGB counterfactual."
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep"),
    )
    parser.add_argument(
        "--comparison-run",
        type=Path,
        default=Path("runs/learned_laplacian/sofa50_residual_target_comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/learned_laplacian/sofa50_counterfactual_refinement"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", default="opengl")
    args = parser.parse_args()
    summary = run_counterfactual_refinement(
        args.source_run,
        args.comparison_run,
        args.output_dir,
        device=args.device,
        render_backend=args.render_backend,
    )
    print(json.dumps(summary["answers"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from mlr.learned_laplacian.h2_normalization_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Sofa50 h^2 Laplacian representations.")
    parser.add_argument(
        "--source-run",
        default="runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/learned_laplacian/sofa50_h2_normalization_audit",
    )
    parser.add_argument("--render-backend", choices=("cpu", "opengl"), default="opengl")
    args = parser.parse_args()
    run_audit(args.source_run, args.output_dir, render_backend=args.render_backend)


if __name__ == "__main__":
    main()

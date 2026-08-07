from __future__ import annotations

import argparse

from mlr.learned_laplacian.recovery_identity_oracle import run_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Sofa50 expanded-mesh recovery identity and exact oracle."
    )
    parser.add_argument(
        "--source-run",
        default="runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/learned_laplacian/sofa50_recovery_identity_oracle_diagnostic",
    )
    parser.add_argument("--render-backend", choices=("cpu", "opengl"), default="opengl")
    args = parser.parse_args()
    run_diagnostic(args.source_run, args.output_dir, render_backend=args.render_backend)


if __name__ == "__main__":
    main()

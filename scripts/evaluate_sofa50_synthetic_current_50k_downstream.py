from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_50k_downstream import (
    print_terminal_summary,
    run_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GT-query 50k and synthetic current-query 20k/50k checkpoints."
    )
    parser.add_argument("--old-comparison", type=Path, required=True)
    parser.add_argument("--current50-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_evaluation(
        args.old_comparison,
        args.current50_run,
        args.output_dir,
        device=args.device,
    )
    print_terminal_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()

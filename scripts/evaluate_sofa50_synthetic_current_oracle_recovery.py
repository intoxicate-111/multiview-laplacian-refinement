from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_oracle_recovery import (
    print_terminal_summary,
    run_oracle_recovery_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current-query 20k/50k learned recovery with current-graph "
            "exact-target recovery."
        )
    )
    parser.add_argument("--downstream-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_oracle_recovery_comparison(
        args.downstream_summary,
        args.output_dir,
        device=args.device,
    )
    print_terminal_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()

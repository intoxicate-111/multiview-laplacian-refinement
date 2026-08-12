from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_topk_recovery import (
    print_terminal_summary,
    run_topk_recovery_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Top-k raw-residual exact-target replacement recovery."
    )
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_topk_recovery_comparison(
        args.oracle_summary, args.output_dir, device=args.device
    )
    print_terminal_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()

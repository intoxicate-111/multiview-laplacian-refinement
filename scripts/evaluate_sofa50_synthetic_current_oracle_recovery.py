from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_oracle_recovery import (
    print_terminal_summary,
    refresh_existing_oracle_report,
    run_oracle_recovery_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current-query 20k/50k learned recovery with current-graph "
            "exact-target recovery."
        )
    )
    parser.add_argument("--downstream-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.refresh_existing:
        summary = refresh_existing_oracle_report(args.output_dir)
    else:
        if args.downstream_summary is None:
            parser.error("--downstream-summary is required unless --refresh-existing is used")
        summary = run_oracle_recovery_comparison(
            args.downstream_summary,
            args.output_dir,
            device=args.device,
        )
    print_terminal_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()

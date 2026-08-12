#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    run_h2_normalization_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the C2F2 28-view current-graph h2 normalization ablation."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-a-run", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-c-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    summary = run_h2_normalization_ablation(
        args.manifest,
        args.arm_a_run,
        args.arm_b_run,
        args.arm_c_run,
        args.output_dir,
        device=args.device,
    )
    print(f"audit_passed\t{summary['contract_audit']['passed']}")
    for row in summary["recovery_aggregate"]:
        if int(row["replacement_percent"]) == 0:
            print(
                f"{row['arm']}\tchamfer={row['reconstruction_chamfer']:.9g}\t"
                f"improved={row['improved_over_initial']}/25"
            )
    print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

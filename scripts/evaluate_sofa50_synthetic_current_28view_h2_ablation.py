#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    merge_h2_normalization_ablation_shards,
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
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_h2_normalization_ablation_shards(
            args.manifest,
            args.output_dir,
            shard_count=args.shard_count,
        )
    else:
        summary = run_h2_normalization_ablation(
            args.manifest,
            args.arm_a_run,
            args.arm_b_run,
            args.arm_c_run,
            args.output_dir,
            device=args.device,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        if args.shard_count > 1:
            print(
                f"shard_complete\t{args.shard_index}/{args.shard_count}\t"
                f"predictions={len(summary['prediction_rows'])}\t"
                f"recoveries={len(summary['recovery_rows'])}"
            )
            return 0
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

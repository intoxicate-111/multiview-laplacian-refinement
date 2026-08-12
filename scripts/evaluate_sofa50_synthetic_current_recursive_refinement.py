#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_recursive_refinement import (
    merge_recursive_refinement_shards,
    run_recursive_refinement_shard,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run three recursive B direct-raw current-mesh refinement rounds."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--visibility-size", type=int, default=960)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_recursive_refinement_shards(
            args.manifest,
            args.output_dir,
            rounds=args.rounds,
            shard_count=args.shard_count,
        )
        decision = summary["decision"]
        print(
            f"max_improved={decision['maximum_improved_over_original']}/25 "
            f"policy={decision['maximum_improved_policy']} "
            f"round={decision['maximum_improved_round']}"
        )
        print(f"report={args.output_dir.resolve() / 'REPORT.md'}")
    else:
        payload = run_recursive_refinement_shard(
            args.manifest,
            args.run_dir,
            args.baseline_analysis_dir,
            args.output_dir,
            rounds=args.rounds,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            visibility_size=args.visibility_size,
        )
        print(
            f"shard_complete={args.shard_index}/{args.shard_count} "
            f"rows={len(payload['rows'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

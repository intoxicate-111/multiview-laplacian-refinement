#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.dynamic_residual_expert_evaluation import (
    merge_dynamic_expert_shards,
    run_dynamic_expert_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the Sofa50 learned dynamic residual expert."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expert-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_dynamic_expert_shards(
            args.manifest, args.output_dir, shard_count=args.shard_count
        )
        print(f"audit_passed\t{summary['contract_audit']['passed']}")
        print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
        return 0
    if args.expert_run is None:
        parser.error("--expert-run is required unless --merge-shards is used")
    payload = run_dynamic_expert_evaluation(
        args.manifest,
        args.expert_run,
        args.output_dir,
        device=args.device,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(
        f"shard_complete\t{payload['shard_index']}/{payload['shard_count']}\t"
        f"predictions={len(payload['prediction_rows'])}\t"
        f"recoveries={len(payload['recovery_rows'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

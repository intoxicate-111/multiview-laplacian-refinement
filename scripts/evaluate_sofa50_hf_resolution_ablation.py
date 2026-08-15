#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_hf_resolution_ablation import (
    merge_hf_resolution_ablation_shards,
    run_hf_resolution_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Sofa50 HF at 960 vs 1920.")
    parser.add_argument("--manifest-960", required=True, type=Path)
    parser.add_argument("--manifest-1920", required=True, type=Path)
    parser.add_argument("--run-960", type=Path)
    parser.add_argument("--run-1920", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_hf_resolution_ablation_shards(
            args.manifest_960,
            args.manifest_1920,
            args.output_dir,
            shard_count=args.shard_count,
        )
        print(f"audit_passed\t{summary['contract_audit']['passed']}")
        print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
        return 0
    if args.run_960 is None or args.run_1920 is None:
        parser.error("--run-960 and --run-1920 are required unless merging")
    payload = run_hf_resolution_ablation(
        args.manifest_960,
        args.manifest_1920,
        args.run_960,
        args.run_1920,
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

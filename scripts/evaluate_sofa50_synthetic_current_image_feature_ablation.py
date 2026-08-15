#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_image_feature_ablation import (
    merge_image_feature_ablation_shards,
    run_image_feature_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Sofa50 image-feature ablation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--gaussian-run", type=Path)
    parser.add_argument("--high-frequency-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_image_feature_ablation_shards(
            args.manifest, args.output_dir, shard_count=args.shard_count
        )
        print(f"audit_passed\t{summary['contract_audit']['passed']}")
        print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
        return 0
    if any(
        value is None
        for value in (args.baseline_run, args.gaussian_run, args.high_frequency_run)
    ):
        parser.error(
            "--baseline-run, --gaussian-run, and --high-frequency-run are required"
        )
    payload = run_image_feature_ablation(
        args.manifest,
        args.baseline_run,
        args.gaussian_run,
        args.high_frequency_run,
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

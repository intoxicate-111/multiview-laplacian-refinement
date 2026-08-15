#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_loss_ablation import (
    merge_loss_ablation_shards,
    run_loss_ablation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Sofa50 Arm B Huber vs raw MSE.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--huber-run", type=Path)
    parser.add_argument("--mse-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        summary = merge_loss_ablation_shards(
            args.manifest, args.output_dir, shard_count=args.shard_count
        )
        print(f"audit_passed\t{summary['contract_audit']['passed']}")
        print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
        return 0
    if args.huber_run is None or args.mse_run is None:
        parser.error("--huber-run and --mse-run are required unless --merge-shards is used")
    payload = run_loss_ablation(
        args.manifest,
        args.huber_run,
        args.mse_run,
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

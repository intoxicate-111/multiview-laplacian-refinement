#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlr.learned_laplacian.synthetic_current_stage2_adaptation import (
    generate_stage2_dataset_shard,
    merge_stage2_dataset_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--b-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visibility-size", type=int, default=960)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        merge_stage2_dataset_shards(
            args.manifest, args.output_dir, shard_count=args.shard_count
        )
    else:
        generate_stage2_dataset_shard(
            args.manifest,
            args.b_run_dir,
            args.output_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            visibility_size=args.visibility_size,
        )


if __name__ == "__main__":
    main()

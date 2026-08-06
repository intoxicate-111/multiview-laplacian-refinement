#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.learned_laplacian.sample_io import prepare_gt_query_sample_from_prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert prepared coarse/expanded samples into direct GT-vertex query samples."
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("samples"), list):
        raise ValueError("Source manifest must contain a samples list.")
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    if any(value is not None and value < 0 for value in limits.values()):
        raise ValueError("Split limits must be non-negative.")
    if output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_manifest}; pass --overwrite.")

    prepared_dir = output_manifest.parent / "prepared_gt_query"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in limits}
    converted_records = []
    for record in source["samples"]:
        if not isinstance(record, dict):
            raise ValueError("Manifest sample records must be objects.")
        split = record.get("split")
        if split not in limits:
            continue
        limit = limits[split]
        if limit is not None and counts[split] >= limit:
            continue
        source_path = Path(str(record["path"]))
        if not source_path.is_absolute():
            source_path = source_manifest.parent / source_path
        source_sample = load_prepared_sample(
            source_path,
            materialize_images=False,
            dataset_root=source_manifest.parent,
        )
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_sample["sample_id"])
        destination = prepared_dir / f"{safe_id}.pt"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite.")
        prepared = prepare_gt_query_sample_from_prepared(
            source_sample,
            output_path=destination,
            image_size=args.image_size,
        )
        converted_records.append(
            {
                "sample_id": prepared["sample_id"],
                "path": str(destination.relative_to(output_manifest.parent)),
                "split": split,
            }
        )
        counts[split] += 1
        print(
            f"converted {prepared['sample_id']} split={split} "
            f"vertices={prepared['vertices'].shape[0]}",
            flush=True,
        )

    if counts["train"] < 1 or counts["validation"] < 1:
        raise ValueError("Converted manifest requires at least one train and validation sample.")
    result = {
        "format_version": "gt_query_manifest_v1",
        "source_manifest": str(source_manifest),
        "query_training_mode": "gt_vertex_perturbation_v1",
        "samples": converted_records,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_manifest} with split counts {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

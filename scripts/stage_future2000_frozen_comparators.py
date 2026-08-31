#!/usr/bin/env python3
from __future__ import annotations

"""Stage frozen Future2000 comparator tables without rerunning their methods."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SOURCES = {
    "old_structure": "ours",
    "nds": "nds",
    "nvdiffrec": "nvdiffrec",
    "exmesh": "exmesh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(str(row["sample_id"]) for row in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-results", required=True, type=Path)
    parser.add_argument("--destination-results", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = sorted(
        str(row["sample_id"])
        for row in manifest["samples"]
        if row.get("split") == "test"
    )
    if len(expected) != 1000 or len(set(expected)) != 1000:
        raise ValueError("Expected exactly 1000 unique Future2000 test samples")

    records = []
    for destination, source in SOURCES.items():
        source_dir = args.archive_results.resolve() / source
        destination_dir = args.destination_results.resolve() / destination
        per_sample = source_dir / "per_sample.csv"
        aggregate = source_dir / "aggregate.json"
        if sample_ids(per_sample) != expected:
            raise ValueError(f"Frozen {source} sample IDs do not match the test split")
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(per_sample, destination_dir / "per_sample.csv")
        shutil.copy2(aggregate, destination_dir / "aggregate.json")
        records.append(
            {
                "destination_method": destination,
                "source_method": source,
                "source_directory": str(source_dir),
                "per_sample_sha256": sha256(per_sample),
                "aggregate_sha256": sha256(aggregate),
                "sample_count": len(expected),
                "rerun": False,
            }
        )
    payload = {
        "contract": "frozen_same-input_archive_reuse_no_method_rerun",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "comparators": records,
    }
    output = args.destination_results.resolve().parent / "FROZEN_COMPARATOR_PROVENANCE.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

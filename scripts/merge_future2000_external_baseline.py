#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "initial_chamfer",
    "refined_chamfer",
    "chamfer_improvement_rate",
    "initial_p2s_mean",
    "refined_p2s_mean",
    "initial_p2s_p95",
    "refined_p2s_p95",
    "initial_fscore",
    "refined_fscore",
    "initial_normal_consistency",
    "refined_normal_consistency",
)


def merge(
    manifest: Path,
    output_dir: Path,
    method: str,
    shard_count: int,
    selection: Path | None = None,
) -> dict[str, Any]:
    method_dir = output_dir / method
    shard_dir = method_dir / "shards"
    rows: list[dict[str, str]] = []
    shard_metadata = []
    for index in range(shard_count):
        with (shard_dir / f"per_sample_shard_{index:03d}.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        shard_metadata.append(
            json.loads(
                (shard_dir / f"metadata_shard_{index:03d}.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    rows.sort(key=lambda row: row["sample_id"])
    if selection is None:
        expected = sorted(
            str(item["sample_id"])
            for item in json.loads(manifest.read_text(encoding="utf-8"))["samples"]
            if item["split"] == "test"
        )
    else:
        expected = sorted(
            str(value)
            for value in json.loads(selection.read_text(encoding="utf-8"))["sample_ids"]
        )
    ids = [row["sample_id"] for row in rows]
    if len(rows) != len(expected) or len(set(ids)) != len(expected) or ids != expected:
        raise ValueError(f"{method} shards do not exactly match the selected test set.")
    per_sample = method_dir / "per_sample.csv"
    with per_sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    if len(completed) == len(rows):
        status = "completed"
    elif completed:
        status = "partial"
    else:
        status = "failed"
    aggregate: dict[str, Any] = {
        "method": method,
        "status": status,
        "total_samples": len(rows),
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "improved_meshes": sum(_boolean(row["improved"]) for row in completed),
        "metrics": {},
        "runtime_seconds_per_mesh": _optional_statistics(completed, "runtime_seconds"),
        "peak_gpu_memory_mb": _optional_statistics(completed, "peak_gpu_memory_mb"),
        "failure_reasons": _failure_counts(failed),
        "pinned_commit": shard_metadata[0]["pinned_commit"],
        "repository": shard_metadata[0]["repository"],
        "shards": shard_metadata,
    }
    for metric in METRICS:
        aggregate["metrics"][metric] = _optional_statistics(completed, metric)
    (method_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    return aggregate


def _optional_statistics(
    rows: list[dict[str, str]], field: str
) -> dict[str, float | list[float]] | None:
    values = np.asarray(
        [float(row[field]) for row in rows if row.get(field) not in (None, "", "None")],
        dtype=np.float64,
    )
    invalid_count = int(np.count_nonzero(~np.isfinite(values)))
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    rng = np.random.default_rng(7)
    bootstrap = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
    )
    return {
        "count": int(len(values)),
        "invalid_count": invalid_count,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _failure_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = f"{row['failure_stage']}: {row['failure_reason']}"
        result[key] = result.get(key, 0) + 1
    return result


def _boolean(value: str) -> bool:
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean {value!r}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.manifest,
                args.output_dir,
                args.method,
                args.shard_count,
                args.selection,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

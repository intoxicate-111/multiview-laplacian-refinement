#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("initial", "laplacian", "displacement")
METRICS = ("chamfer", "p2s_mean", "p2s_p95", "fscore", "normal_consistency")


def merge(manifest: Path, output_dir: Path, shard_count: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = output_dir / "shards"
    rows = []
    metadata = []
    for index in range(shard_count):
        csv_path = shards / f"per_sample_shard_{index:03d}.csv"
        metadata_path = shards / f"metadata_shard_{index:03d}.json"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in rows]
    expected = sorted(
        str(item["sample_id"])
        for item in json.loads(manifest.read_text(encoding="utf-8"))["samples"]
        if item["split"] == "test"
    )
    if len(rows) != 1000 or len(set(ids)) != 1000 or ids != expected:
        raise ValueError("Merged shards do not exactly match the 1000-sample test split.")

    per_sample = output_dir / "per_sample.csv"
    with per_sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    peak_memory = [
        float(item["peak_gpu_memory_mb"])
        for item in metadata
        if item.get("peak_gpu_memory_mb") is not None
    ]
    aggregate = {
        "status": "completed",
        "test_samples": len(rows),
        "surface_samples": int(metadata[0]["surface_samples"]),
        "surface_sampling_seed": int(metadata[0]["metric_seed"]),
        "fscore_threshold": float(metadata[0]["fscore_threshold"]),
        "methods": {},
        "learned_diagnostics": {},
        "runtime": {
            "shard_runtime_seconds": [float(item["runtime_seconds"]) for item in metadata],
            "peak_gpu_memory_mb": max(peak_memory) if peak_memory else None,
            "laplacian_forward_seconds_per_mesh": _statistics(
                _required_values(rows, "laplacian_forward_seconds")
            ),
            "laplacian_recovery_seconds_per_mesh": _statistics(
                _required_values(rows, "laplacian_recovery_seconds")
            ),
            "displacement_forward_seconds_per_mesh": _statistics(
                _required_values(rows, "displacement_forward_seconds")
            ),
        },
        "shards": metadata,
    }
    for method in METHODS:
        method_result = {}
        for metric in METRICS:
            values = np.asarray(
                [float(row[f"{method}_{metric}"]) for row in rows], dtype=np.float64
            )
            method_result[metric] = _statistics(values)
        if method != "initial":
            improved = np.asarray(
                [_boolean(row[f"{method}_improved"]) for row in rows]
            )
            rates = np.asarray(
                [float(row[f"{method}_chamfer_improvement_rate"]) for row in rows],
                dtype=np.float64,
            )
            method_result["improved_meshes"] = int(improved.sum())
            method_result["total_meshes"] = len(rows)
            method_result["chamfer_improvement_rate"] = _statistics(rates)
        aggregate["methods"][method] = method_result

    for prefix in ("laplacian", "displacement"):
        diagnostics = {}
        for field in (
            "confidence_mean",
            "confidence_std",
            "visible_confidence_mean",
            "unseen_confidence_mean",
        ):
            values = _optional_values(rows, f"{prefix}_{field}")
            diagnostics[field] = None if not len(values) else _statistics(values)
        aggregate["learned_diagnostics"][prefix] = diagnostics
    for field in (
        "laplacian_raw_target_epe",
        "laplacian_raw_cosine",
        "laplacian_visible_raw_epe",
        "laplacian_unseen_raw_epe",
    ):
        values = _optional_values(rows, field)
        aggregate["learned_diagnostics"]["laplacian"][field] = (
            None if not len(values) else _statistics(values)
        )
    for field in (
        "displacement_target_epe",
        "displacement_visible_epe",
        "displacement_unseen_epe",
    ):
        values = _optional_values(rows, field)
        aggregate["learned_diagnostics"]["displacement"][field] = (
            None if not len(values) else _statistics(values)
        )

    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    baseline_status = {
        "Initial/current mesh": {"status": "completed", "samples": 1000},
        "Direct displacement": {"status": "completed", "samples": 1000},
        "Learned current-graph raw Laplacian": {
            "status": "completed",
            "samples": 1000,
        },
        "OpenMVS RefineMesh": {"status": "not_run"},
        "NDS": {"status": "not_run"},
        "NeRF2Mesh": {"status": "not_run"},
        "ExMesh": {"status": "not_run"},
    }
    (output_dir / "baseline_status.json").write_text(
        json.dumps(baseline_status, indent=2) + "\n", encoding="utf-8"
    )
    return aggregate


def _statistics(values: np.ndarray) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(7)
    bootstrap = np.empty(2000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _optional_values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row.get(field) not in (None, "", "None")],
        dtype=np.float64,
    )


def _required_values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    values = _optional_values(rows, field)
    if len(values) != len(rows):
        raise ValueError(f"Missing required per-sample values for {field!r}.")
    return values


def _boolean(value: str) -> bool:
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-count", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(merge(args.manifest.resolve(), args.output_dir.resolve(), args.shard_count), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Merge Future2000 validation shards and lock B+E lambda by mean CD."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=1000)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    shard_dir = args.output_dir / "validation" / "shards"
    for index in range(args.shard_count):
        path = shard_dir / f"validation_shard_{index:03d}.csv"
        meta_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"Missing validation shard or metadata: {path}")
        rows.extend(_read(path))
        metadata.append(json.loads(meta_path.read_text(encoding="utf-8")))
    if not all(item.get("contract_audit") for item in metadata):
        raise RuntimeError("At least one validation shard failed its contract audit")
    stable_keys = ("manifest_sha256", "arm_b_checkpoint_sha256", "arm_e_checkpoint_sha256")
    for key in stable_keys:
        values = {item[key] for item in metadata}
        if len(values) != 1:
            raise RuntimeError(f"Validation shard {key} mismatch: {values}")
    if sha256(args.manifest.resolve()) != metadata[0]["manifest_sha256"]:
        raise RuntimeError("Manifest changed between evaluation and lambda selection")

    hybrid = [row for row in rows if row["method"] == "hybrid"]
    arm_e = [row for row in rows if row["method"] == "arm_e"]
    ids = {row["sample_id"] for row in arm_e}
    if len(ids) != args.expected_samples or len(arm_e) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} unique Arm-E validation rows")
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in hybrid:
        if row["pcg_converged"].lower() != "true":
            raise RuntimeError(f"Non-converged validation row: {row['sample_id']}")
        grouped[float(row["lambda"])].append(row)
    if not grouped:
        raise RuntimeError("No hybrid validation rows found")
    summary: list[dict[str, Any]] = []
    for regularization in sorted(grouped):
        items = grouped[regularization]
        item_ids = [row["sample_id"] for row in items]
        if len(items) != args.expected_samples or set(item_ids) != ids or len(set(item_ids)) != len(item_ids):
            raise RuntimeError(f"Incomplete or duplicate lambda={regularization} rows")
        summary.append({
            "lambda": regularization,
            "sample_count": len(items),
            "mean_cd": float(np.mean([float(row["refined_chamfer"]) for row in items])),
            "mean_p2s_p95": float(np.mean([float(row["refined_p2s_p95"]) for row in items])),
            "mean_fscore": float(np.mean([float(row["refined_fscore"]) for row in items])),
            "mean_normal": float(np.mean([float(row["refined_normal_consistency"]) for row in items])),
            "mean_vrms": float(np.mean([float(row["same_index_recovered_vertex_rms"]) for row in items])),
            "improved": sum(row["improved"].lower() == "true" for row in items),
            "worsened": sum(row["improved"].lower() != "true" for row in items),
            "maximum_pcg_relative_residual": max(float(row["pcg_relative_residual"]) for row in items),
        })
    selected = min(summary, key=lambda row: (row["mean_cd"], row["lambda"]))
    validation_dir = args.output_dir / "validation"
    _write_csv(validation_dir / "validation_per_sample.csv", rows)
    _write_csv(validation_dir / "lambda_sweep.csv", summary)
    lock = {
        "contract_audit": True,
        "selection_split": "validation",
        "selection_metric": "mean_refined_chamfer",
        "test_data_used_for_selection": False,
        "selected_lambda": selected["lambda"],
        "selected_validation_mean_cd": selected["mean_cd"],
        "lambda_grid": [row["lambda"] for row in summary],
        "validation_sample_count": args.expected_samples,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": metadata[0]["manifest_sha256"],
        "arm_b_checkpoint_sha256": metadata[0]["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint_sha256": metadata[0]["arm_e_checkpoint_sha256"],
        "pcg_tolerance": metadata[0]["pcg_tolerance"],
        "pcg_maximum_iterations": metadata[0]["pcg_maximum_iterations"],
    }
    (validation_dir / "lambda_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

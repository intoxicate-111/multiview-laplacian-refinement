from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/merge_future2000_external_baseline.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_preserves_actual_partial_and_failure_status(tmp_path: Path) -> None:
    merger = _module()
    ids = [f"sample_{index:04d}" for index in range(1000)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"samples": [{"sample_id": item, "split": "test"} for item in ids]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    shards = output / "nds/shards"
    shards.mkdir(parents=True)
    rows = []
    for index, sample_id in enumerate(ids):
        completed = index < 999
        row = {
            "sample_id": sample_id,
            "method": "nds",
            "status": "completed" if completed else "failed",
            "failure_stage": "" if completed else "execution",
            "failure_reason": "" if completed else "missing output",
            "runtime_seconds": "1.5" if completed else "",
            "peak_gpu_memory_mb": "1024" if completed else "",
            "vertex_count": "3" if completed else "",
            "face_count": "1" if completed else "",
            "improved": "true" if completed else "",
        }
        for metric in merger.METRICS:
            row[metric] = "0.1" if completed else ""
        rows.append(row)
    with (shards / "per_sample_shard_000.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (shards / "metadata_shard_000.json").write_text(
        json.dumps(
            {
                "pinned_commit": "a" * 40,
                "repository": "https://example.test/nds",
            }
        ),
        encoding="utf-8",
    )

    result = merger.merge(manifest, output, "nds", 1)

    assert result["status"] == "partial"
    assert result["completed_samples"] == 999
    assert result["failed_samples"] == 1
    assert result["improved_meshes"] == 999
    assert result["failure_reasons"] == {"execution: missing output": 1}

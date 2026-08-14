from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_laplacian_and_displacement_configs_are_a_controlled_pair() -> None:
    evaluation = _load_script("evaluate_future2000_laplacian_vs_displacement.py")
    laplacian = json.loads(
        (
            ROOT
            / "configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json"
        ).read_text(encoding="utf-8")
    )
    displacement = json.loads(
        (
            ROOT
            / "configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_displacement_20k.json"
        ).read_text(encoding="utf-8")
    )
    evaluation._assert_fair_pair(laplacian, displacement)

    displacement["model"]["hidden_dim"] += 1
    with pytest.raises(ValueError, match="controlled pair"):
        evaluation._assert_fair_pair(laplacian, displacement)


def test_merge_reports_per_mesh_runtime_and_displacement_epe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merger = _load_script("merge_future2000_learned_evaluation.py")
    monkeypatch.setattr(merger, "METHODS", ("initial", "laplacian", "displacement"))
    sample_ids = [f"sample_{index:04d}" for index in range(1000)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": sample_id, "split": "test"}
                    for sample_id in sample_ids
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    shards = output / "shards"
    shards.mkdir(parents=True)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            "laplacian_forward_seconds": "0.01",
            "laplacian_recovery_seconds": "0.02",
            "displacement_forward_seconds": "0.03",
            "laplacian_confidence_mean": "0.5",
            "laplacian_confidence_std": "0.1",
            "laplacian_visible_confidence_mean": "0.6",
            "laplacian_unseen_confidence_mean": "0.4",
            "displacement_confidence_mean": "0.5",
            "displacement_confidence_std": "0.1",
            "displacement_visible_confidence_mean": "0.6",
            "displacement_unseen_confidence_mean": "0.4",
            "laplacian_raw_target_epe": "0.1",
            "laplacian_raw_cosine": "0.9",
            "laplacian_visible_raw_epe": "0.09",
            "laplacian_unseen_raw_epe": "0.11",
            "displacement_target_epe": "0.2",
            "displacement_visible_epe": "0.18",
            "displacement_unseen_epe": "0.22",
        }
        for method in merger.METHODS:
            for metric in merger.METRICS:
                row[f"{method}_{metric}"] = str(0.1 + index * 1e-8)
            if method != "initial":
                row[f"{method}_improved"] = "true"
                row[f"{method}_chamfer_improvement_rate"] = "0.25"
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
                "surface_samples": 3000,
                "metric_seed": 7,
                "fscore_threshold": 0.01,
                "runtime_seconds": 12.0,
                "peak_gpu_memory_mb": None,
            }
        ),
        encoding="utf-8",
    )

    result = merger.merge(manifest, output, shard_count=1)

    assert result["runtime"]["peak_gpu_memory_mb"] is None
    assert result["runtime"]["laplacian_forward_seconds_per_mesh"]["mean"] == pytest.approx(0.01)
    assert result["learned_diagnostics"]["displacement"]["displacement_target_epe"][
        "mean"
    ] == pytest.approx(0.2)

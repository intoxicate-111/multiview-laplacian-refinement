from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlr.learned_laplacian.dataset import save_prepared_sample
from mlr.learned_laplacian.prediction_visualizer import (
    PredictionRecord,
    VisualizationOptions,
    discover_predictions,
    discover_run_metadata,
    load_prediction_sample,
    prediction_listing,
    resolve_refinement_config,
    visualize_prediction_sample,
    visualize_prediction_split,
)
from mlr.learned_laplacian.target_scaling import EDGE_SCALE_DEFINITION

from .helpers import tiny_sample


def _make_run(tmp_path: Path, samples: list[tuple[str, dict]], config=None):
    if config is None:
        config = {
            "target_mode": "edge_scale_normalized_laplacian",
            "target_scaling": {
                "method": EDGE_SCALE_DEFINITION,
                "epsilon": 1e-12,
            },
            "reconstruction": {
                "operator_type": "uniform",
                "lambda_anchor": 0.01,
                "num_iters": 2,
                "learning_rate": 0.001,
            },
        }
    run_dir = tmp_path / "arbitrary_run"
    prepared_dir = tmp_path / "prepared"
    prediction_dir = run_dir / "predictions" / "validation"
    prepared_dir.mkdir()
    prediction_dir.mkdir(parents=True)
    manifest_samples = []
    for sample_id, sample in samples:
        sample["sample_id"] = sample_id
        sample.setdefault("metadata", {})["operator_type"] = "uniform"
        if config.get("target_mode") is not None:
            sample["metadata"]["laplacian_target_mode"] = config["target_mode"]
        path = prepared_dir / f"{sample_id}.pt"
        save_prepared_sample(sample, path)
        manifest_samples.append(
            {"sample_id": sample_id, "split": "validation", "path": str(path)}
        )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps({"samples": manifest_samples}), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_dir, prediction_dir


def _options(run_dir: Path, **overrides):
    values = {
        "output_dir": run_dir / "visualizations" / "validation",
        "num_iters": 2,
        "skip_render": True,
        "progress": False,
    }
    values.update(overrides)
    return VisualizationOptions(**values)


def test_prediction_shape_matches_mesh_and_summary_schema(tmp_path):
    sample = tiny_sample()
    run_dir, prediction_dir = _make_run(tmp_path, [("matching", sample)])
    raw_path = prediction_dir / "matching_raw_delta.npy"
    np.save(raw_path, sample["laplacian_target"].numpy())
    metadata = discover_run_metadata(run_dir)
    record = discover_predictions(run_dir, "validation")["matching"]

    summary = visualize_prediction_sample(
        metadata, "validation", record, _options(run_dir)
    )

    required = {
        "sample_id",
        "split",
        "run_dir",
        "prediction_path",
        "prediction_format",
        "prediction_space",
        "input_mesh_path",
        "gt_mesh_path",
        "camera_source",
        "camera_index",
        "vertex_count",
        "face_count",
        "target_mode",
        "operator_type",
        "target_scaling",
        "raw_delta_mean_norm",
        "raw_delta_median_norm",
        "raw_delta_max_norm",
        "mean_vertex_displacement",
        "median_vertex_displacement",
        "max_vertex_displacement",
        "final_refinement_loss",
        "refinement_config",
        "output_files",
        "warnings",
    }
    assert required <= summary.keys()
    output = run_dir / "visualizations" / "validation" / "matching"
    assert (output / "coarse.obj").is_file()
    assert (output / "predicted_refined.obj").is_file()
    assert (output / "gt.obj").is_file()
    assert (output / "refinement_history.json").is_file()
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.parametrize(
    ("prediction", "message"),
    [
        (np.zeros((3, 3)), "does not match input mesh"),
        (np.full((4, 3), np.nan), "NaN or infinite"),
    ],
)
def test_invalid_prediction_is_rejected(tmp_path, prediction, message):
    run_dir, prediction_dir = _make_run(tmp_path, [("invalid", tiny_sample())])
    np.save(prediction_dir / "invalid_raw_delta.npy", prediction)
    metadata = discover_run_metadata(run_dir)
    record = discover_predictions(run_dir, "validation")["invalid"]

    with pytest.raises(ValueError, match=message):
        visualize_prediction_sample(metadata, "validation", record, _options(run_dir))


def test_missing_gt_still_reconstructs(tmp_path):
    sample = tiny_sample()
    sample.pop("gt_vertices")
    sample.pop("gt_faces")
    run_dir, prediction_dir = _make_run(tmp_path, [("no_gt", sample)])
    np.save(prediction_dir / "no_gt_raw_delta.npy", sample["laplacian_target"].numpy())
    metadata = discover_run_metadata(run_dir)

    summary = visualize_prediction_sample(
        metadata,
        "validation",
        discover_predictions(run_dir, "validation")["no_gt"],
        _options(run_dir, skip_render=False, image_size=32),
    )

    assert summary["gt_available"] is False
    output = run_dir / "visualizations" / "validation" / "no_gt"
    assert (output / "predicted_refined.obj").is_file()
    assert (output / "comparison.png").is_file()
    assert not (output / "gt.obj").exists()


def test_batch_continues_after_one_sample_fails(tmp_path):
    good = tiny_sample()
    bad = tiny_sample()
    run_dir, prediction_dir = _make_run(
        tmp_path, [("good", good), ("bad", bad)]
    )
    np.save(prediction_dir / "good_raw_delta.npy", good["laplacian_target"].numpy())
    np.save(prediction_dir / "bad_raw_delta.npy", np.zeros((2, 3)))
    metadata = discover_run_metadata(run_dir)

    batch = visualize_prediction_split(
        metadata,
        "validation",
        list(discover_predictions(run_dir, "validation").values()),
        _options(run_dir),
    )

    assert batch["processed"] == 2
    assert batch["succeeded"] == 1
    assert batch["failed"] == 1
    assert batch["skipped"] == 0
    failed = next(item for item in batch["sample_results"] if item["status"] == "failed")
    assert failed["sample_id"] == "bad"
    assert failed["error_type"] == "ValueError"
    assert (run_dir / "visualizations" / "validation" / "batch_summary.json").is_file()


def test_prediction_and_sample_id_discovery(tmp_path):
    run_dir, prediction_dir = _make_run(tmp_path, [("mesh_a", tiny_sample())])
    np.save(prediction_dir / "mesh_a_raw_delta.npy", np.zeros((4, 3)))
    np.save(prediction_dir / "mesh_a_target_space_delta.npy", np.ones((4, 3)))
    records = discover_predictions(run_dir, "validation")
    metadata = discover_run_metadata(run_dir)

    assert list(records) == ["mesh_a"]
    sample, path = load_prediction_sample(metadata, "validation", "mesh_a")
    assert sample["sample_id"] == "mesh_a"
    assert path.name == "mesh_a.pt"
    listing = prediction_listing(
        metadata, "validation", records, run_dir / "visualizations" / "validation"
    )
    assert listing == [
        {
            "sample_id": "mesh_a",
            "raw_prediction": True,
            "target_space_prediction": True,
            "sample_metadata": True,
            "visualization_exists": False,
        }
    ]


def test_cli_values_override_run_refinement_config(tmp_path):
    run_dir, _ = _make_run(tmp_path, [("override", tiny_sample())])
    metadata = discover_run_metadata(run_dir)
    sample, _ = load_prediction_sample(metadata, "validation", "override")

    config, warnings = resolve_refinement_config(
        metadata.config,
        sample,
        {"lambda_anchor": 0.25, "num_iters": 7, "learning_rate": 0.02},
    )

    assert config.lambda_anchor == 0.25
    assert config.num_iters == 7
    assert config.learning_rate == 0.02
    assert warnings == []


def test_raw_prediction_is_preferred_over_target_space(tmp_path):
    sample = tiny_sample()
    run_dir, prediction_dir = _make_run(tmp_path, [("priority", sample)])
    raw_path = prediction_dir / "priority_raw_delta.npy"
    np.save(raw_path, sample["laplacian_target"].numpy())
    np.save(prediction_dir / "priority_target_space_delta.npy", np.full((4, 3), np.nan))
    metadata = discover_run_metadata(run_dir)

    summary = visualize_prediction_sample(
        metadata,
        "validation",
        discover_predictions(run_dir, "validation")["priority"],
        _options(run_dir),
    )

    assert summary["prediction_path"] == str(raw_path)
    assert summary["prediction_space"] == "raw_laplacian"


def test_target_space_without_run_scaling_metadata_is_rejected(tmp_path):
    run_dir, prediction_dir = _make_run(
        tmp_path,
        [("target_only", tiny_sample())],
        config={"reconstruction": {"operator_type": "uniform", "num_iters": 2}},
    )
    np.save(prediction_dir / "target_only_target_space_delta.npy", np.zeros((4, 3)))
    metadata = discover_run_metadata(run_dir)

    with pytest.raises(ValueError, match="target scaling metadata is insufficient"):
        visualize_prediction_sample(
            metadata,
            "validation",
            discover_predictions(run_dir, "validation")["target_only"],
            _options(run_dir),
        )


def test_npz_prediction_record_is_discovered(tmp_path):
    run_dir, prediction_dir = _make_run(tmp_path, [("packed", tiny_sample())])
    np.savez(
        prediction_dir / "anything.npz",
        sample_id=np.asarray("packed"),
        raw_delta=np.zeros((4, 3), dtype=np.float32),
    )

    record = discover_predictions(run_dir, "validation")["packed"]

    assert isinstance(record, PredictionRecord)
    assert record.record_path.name == "anything.npz"

import pytest

from mlr.learned_laplacian.synthetic_current_50k_downstream import (
    _comparison,
    _mapping_differences,
    _regression_audit,
    _sample_analysis,
    _wide_per_sample,
)


def _aggregate(offset: float = 0.0) -> dict:
    return {
        "normalized_mse": 1.0 + offset,
        "vector_l2": 2.0 + offset,
        "global_cosine": 0.8 + offset,
        "high_10_percent_cosine": 0.9 + offset,
        "prediction_target_norm_ratio": 0.95 + offset,
        "loss": 0.01 + offset,
        "zero_rgb_loss": 0.02 + offset,
        "correct_zero_loss_gap": 0.01,
        "correct_zero_cosine_gap": 0.1,
        "initial_chamfer": 0.004,
        "reconstruction_chamfer": 0.003 + offset,
        "reconstruction_point_to_surface": 0.0031 + offset,
        "reconstruction_normal_consistency": 0.94 + offset,
        "introduced_flipped_faces": 10,
        "new_degenerate_faces": 0,
        "improved_over_initial": 1,
        "sample_count": 1,
    }


def _row(experiment: str, chamfer: float) -> dict:
    return {
        "experiment": experiment,
        "sample_id": "object__v00",
        "object_id": "object",
        "variant_index": 0,
        "initial_chamfer": 0.004,
        "reconstruction_chamfer": chamfer,
        "initial_point_to_surface": 0.0041,
        "reconstruction_point_to_surface": chamfer + 0.0001,
        "initial_normal_consistency": 0.9,
        "reconstruction_normal_consistency": 0.95,
        "introduced_flipped_faces": 2,
        "correct_rgb_loss": 0.01,
        "zero_rgb_loss": 0.02,
        "correct_zero_loss_gap": 0.01,
        "vector_l2": 2.0,
        "global_cosine": 0.9,
        "high_10_percent_cosine": 0.95,
        "prediction_target_norm_ratio": 1.0,
    }


def test_current_configs_may_differ_only_in_step_budget():
    left = {"model": {"hidden": 256}, "multi_object_training": {"max_optimizer_steps": 20000}}
    right = {"model": {"hidden": 256}, "multi_object_training": {"max_optimizer_steps": 50000}}
    assert _mapping_differences(left, right) == [
        {
            "path": "multi_object_training.max_optimizer_steps",
            "left": 20000,
            "right": 50000,
        }
    ]


def test_saved_ab_regression_checks_metrics_and_sample_identity():
    payload = {
        "manifest": "/tmp/manifest.json",
        "test_samples": 25,
        "test_objects": 5,
        "aggregate": {"A": _aggregate(), "B": _aggregate()},
        "per_variant": [_row("A", 0.003), _row("B", 0.003)],
    }
    audit = _regression_audit(payload, payload)
    assert audit["passed"] is True
    changed = {
        **payload,
        "aggregate": {"A": _aggregate(), "B": _aggregate(0.1)},
    }
    audit = _regression_audit(payload, changed)
    assert audit["passed"] is False
    assert audit["aggregate_B_match"]["passed"] is False


def test_three_model_rows_are_joined_by_sample_and_classified():
    regression = [_row("A", 0.005), _row("B", 0.0035)]
    current50 = [_row("A", 0.005), _row("B", 0.0030)]
    rows = _wide_per_sample(regression, current50)
    assert len(rows) == 1
    row = rows[0]
    assert row["variant_id"] == "v00"
    assert row["current20_better_than_initial"] is True
    assert row["current50_better_than_initial"] is True
    assert row["current50_minus_current20_chamfer"] < 0.0
    outcomes = _sample_analysis(rows)
    assert outcomes["retained_from_20k"] == ["object__v00"]
    assert outcomes["new_at_50k"] == []
    assert outcomes["lost_at_50k"] == []


def test_comparison_records_absolute_percent_and_directional_change():
    source = {
        "loss": 2.0,
        "normalized_mse": 2.0,
        "vector_l2": 2.0,
        "global_cosine": 0.8,
        "high_10_percent_cosine": 0.9,
        "prediction_target_norm_ratio": 1.0,
        "zero_rgb_loss": 4.0,
        "correct_zero_loss_gap": 2.0,
        "relative_correct_vs_zero_improvement": 0.5,
        "reconstruction_chamfer": 2.0,
        "reconstruction_point_to_surface": 2.0,
        "reconstruction_normal_consistency": 0.8,
        "introduced_flipped_faces": 10,
        "improved_over_initial": 5,
    }
    target = {**source, "loss": 1.0, "global_cosine": 0.9}
    result = _comparison(source, target)
    assert result["loss"]["absolute_change"] == -1.0
    assert result["loss"]["percent_change"] == -50.0
    assert result["loss"]["directional_improvement_percent"] == 50.0
    assert result["global_cosine"]["absolute_change"] == pytest.approx(0.1)

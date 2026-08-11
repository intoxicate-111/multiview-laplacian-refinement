from __future__ import annotations

import json
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_comparison import (
    _aggregate_per_object,
    _markdown_report,
)


def test_experiment_b_config_is_fixed_c2f2_current_query() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            root
            / "configs"
            / "learned_laplacian"
            / "train_sofa50_synthetic_current_c2f2_14view_20k.json"
        ).read_text(encoding="utf-8")
    )
    assert config["image_encoder"]["feature_dim"] == 64
    assert config["image_encoder"]["first_stride"] == 1
    assert config["image_encoder"]["second_stride"] == 1
    assert config["model"]["hidden_dim"] == 256
    assert config["query_training"]["enabled"] is False
    assert config["query_training"]["zero_initial_laplacian"] is False
    assert config["experiment_metadata"]["views"] == 14
    assert config["multi_object_training"]["max_optimizer_steps"] == 20_000


def test_per_object_report_retains_five_variants() -> None:
    rows = []
    for experiment in ("A", "B"):
        for variant in range(5):
            rows.append(
                {
                    "experiment": experiment,
                    "object_id": "held-out-object",
                    "normalized_mse": 1.0 + variant,
                    "vector_l2": 2.0,
                    "global_cosine": 0.5,
                    "high_10_percent_cosine": 0.4,
                    "prediction_target_norm_ratio": 0.9,
                    "correct_rgb_loss": 0.1,
                    "zero_rgb_loss": 0.2,
                    "correct_zero_loss_gap": 0.1,
                    "initial_chamfer": 0.01,
                    "reconstruction_chamfer": 0.009,
                    "reconstruction_point_to_surface": 0.008,
                    "reconstruction_normal_consistency": 0.95,
                    "introduced_flipped_faces": variant,
                    "improved_over_initial": True,
                }
            )
    report = _aggregate_per_object(rows)
    assert len(report) == 2
    assert all(row["variant_count"] == 5 for row in report)
    assert all(row["improved_over_initial"] == 5 for row in report)
    assert all(row["introduced_flipped_faces"] == 10 for row in report)


def test_markdown_labels_comparison_as_non_strict_training_ablation() -> None:
    metrics = {
        "normalized_mse": 1.0,
        "vector_l2": 1.0,
        "global_cosine": 1.0,
        "high_10_percent_cosine": 1.0,
        "prediction_target_norm_ratio": 1.0,
        "loss": 1.0,
        "zero_rgb_loss": 1.0,
        "correct_zero_loss_gap": 0.0,
        "reconstruction_chamfer": 1.0,
        "reconstruction_point_to_surface": 1.0,
        "reconstruction_normal_consistency": 1.0,
        "introduced_flipped_faces": 0,
        "improved_over_initial": 0,
    }
    text = _markdown_report(
        {
            "aggregate": {"A": metrics, "B": metrics},
            "test_samples": 25,
            "test_objects": 5,
        }
    )
    assert "not a strict paired-training ablation" in text
    assert "| Training rerun | NO | YES |" in text
    assert "same synthetic C" in text

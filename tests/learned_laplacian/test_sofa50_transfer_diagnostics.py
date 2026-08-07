import json
from pathlib import Path

import numpy as np

from mlr.learned_laplacian.sofa50_transfer_diagnostics import (
    _outside_augmentation_summary,
    _ratio_summary,
)


def test_ratio_summary_reports_requested_quantiles():
    values = np.arange(1.0, 11.0)
    summary = _ratio_summary(values)

    assert summary["mean"] == np.mean(values)
    assert summary["median"] == np.median(values)
    assert summary["p90"] == np.quantile(values, 0.90)
    assert summary["p95"] == np.quantile(values, 0.95)
    assert summary["p99"] == np.quantile(values, 0.99)
    assert summary["max"] == 10.0


def test_outside_augmentation_summary_uses_h_normalized_training_cap():
    ratio = np.array([0.0, 0.0005, 0.001, 0.002])
    summary = _outside_augmentation_summary(ratio, 0.001)

    assert summary["fraction_above_training_max_offset_h"] == 0.25
    assert summary["median_multiple_of_training_max"] == 0.75
    assert summary["p95_multiple_of_training_max"] == np.quantile(ratio, 0.95) / 0.001


def test_canonical_sofa50_recovery_has_no_extra_unseen_anchor():
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            root
            / "configs/learned_laplacian/"
            "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
        ).read_text(encoding="utf-8")
    )

    assert config["recovery"]["lambda_anchor"] == 0.01
    assert config["recovery"]["unseen_anchor_weight"] == 0.0
    assert (
        config["confidence"]["recovery_weight"]
        == "renderer_visible_any_times_confidence_prediction"
    )

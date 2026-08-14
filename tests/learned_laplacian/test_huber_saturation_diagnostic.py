from __future__ import annotations

import numpy as np
import pytest

from mlr.learned_laplacian.huber_saturation_diagnostic import (
    summarize_huber_saturation,
)


def test_huber_saturation_uses_componentwise_threshold_and_exact_groups() -> None:
    target = np.zeros((100, 3), dtype=np.float64)
    target[:, 0] = np.arange(100, dtype=np.float64)
    prediction = target.copy()
    prediction[:90, 0] += 0.001
    prediction[90:, 0] += 0.02
    prediction[99, 1] += 0.03
    weight = np.ones(100, dtype=np.float64)
    sample_index = np.zeros(100, dtype=np.int64)

    result = summarize_huber_saturation(
        prediction,
        target,
        weight,
        sample_index,
        huber_delta=0.01,
    )
    rows = {row["group"]: row for row in result["groups"]}

    assert rows["bottom_90_percent"]["vertex_count"] == 90
    assert rows["top_10_percent"]["vertex_count"] == 10
    assert rows["top_1_percent"]["vertex_count"] == 1
    assert rows["bottom_90_percent"]["component_saturation_probability"] == 0.0
    assert rows["top_10_percent"]["component_saturation_probability"] == pytest.approx(
        11 / 30
    )
    assert rows["top_10_percent"][
        "vertex_any_component_saturated_probability"
    ] == 1.0
    assert rows["top_1_percent"]["component_saturation_probability"] == pytest.approx(
        2 / 3
    )
    assert rows["top_10_percent"][
        "huber_gradient_retention_vs_unclipped_l1"
    ] < 1.0
    assert sum(
        rows[name]["weighted_output_gradient_l1_share"]
        for name in ("bottom_90_percent", "top_10_percent")
    ) == pytest.approx(1.0)


def test_huber_saturation_preserves_per_sample_weight_normalization() -> None:
    target = np.zeros((4, 3), dtype=np.float64)
    target[:, 0] = (0.0, 1.0, 2.0, 3.0)
    prediction = target.copy()
    prediction[:, 0] += 0.005
    weight = np.array((1.0, 1.0, 10.0, 10.0))
    sample_index = np.array((0, 0, 1, 1))

    result = summarize_huber_saturation(
        prediction,
        target,
        weight,
        sample_index,
        huber_delta=0.01,
    )

    # Each sample contributes the same total after its own weight normalization.
    assert result["overall"]["weighted_output_gradient_l1_total"] == pytest.approx(
        0.005 / 3.0
    )

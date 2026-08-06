import json

import numpy as np
import torch

from mlr.learned_laplacian.diagnostics import (
    _bin_masks,
    _distribution,
    _magnitude_scope,
    _write_colored_ply,
)


def test_magnitude_scope_reports_vector_error_and_stable_ratios():
    target = np.array([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    prediction = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])

    result = _magnitude_scope(target, prediction, ratio_threshold=1e-8)

    assert result["target_magnitude"]["mean"] == 3.5
    assert result["prediction_magnitude"]["mean"] == 3.0
    np.testing.assert_allclose(
        result["error_magnitude"]["mean"],
        np.mean([np.sqrt(10.0), 1.0]),
    )
    np.testing.assert_allclose(result["magnitude_ratio_global"], 3.0 / 3.5)


def test_global_percentile_bins_use_supplied_validation_thresholds():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    masks = _bin_masks(values, {"p50": 2.0, "p90": 3.5, "p95": 3.8, "p99": 3.96})

    np.testing.assert_array_equal(masks["low"], [True, True, True, False, False])
    np.testing.assert_array_equal(masks["medium"], [False, False, False, True, False])
    np.testing.assert_array_equal(masks["top_10"], [False, False, False, False, True])
    np.testing.assert_array_equal(masks["top_1"], [False, False, False, False, True])


def test_distribution_and_colored_ply_are_finite_and_parseable(tmp_path):
    stats = _distribution(np.array([0.0, 1.0, 2.0, 3.0]))
    json.dumps(stats, allow_nan=False)
    assert stats["median"] == 1.5

    path = tmp_path / "colored.ply"
    _write_colored_ply(
        path,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
        np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8),
    )
    text = path.read_text(encoding="ascii")
    assert "property uchar red" in text
    assert "element vertex 3" in text
    assert text.rstrip().endswith("3 0 1 2")

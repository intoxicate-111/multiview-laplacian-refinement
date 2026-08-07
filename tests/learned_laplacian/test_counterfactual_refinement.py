from __future__ import annotations

import numpy as np

from mlr.learned_laplacian.counterfactual_refinement import (
    symmetric_currents,
    vector_alignment_metrics,
)


def test_symmetric_currents_reflect_about_base() -> None:
    base = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    plus = base + np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
    currents = symmetric_currents(base, plus)
    np.testing.assert_allclose(currents["plus"] - base, -(currents["minus"] - base))
    np.testing.assert_array_equal(currents["base"], base)


def test_vector_alignment_metrics_recovers_direction_and_scale() -> None:
    target = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    prediction = 0.5 * target
    metrics = vector_alignment_metrics(prediction, target)
    assert np.isclose(metrics["global_cosine"], 1.0)
    assert np.isclose(metrics["mean_per_vertex_cosine"], 1.0)
    assert np.isclose(metrics["norm_ratio"], 0.5)
    assert np.isclose(metrics["alpha_star"], 2.0)

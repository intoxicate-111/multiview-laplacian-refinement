from __future__ import annotations

import torch

from mlr.learned_laplacian.scaling_diagnostics import (
    _available_queries,
    _prediction_metrics,
)


def test_prediction_metrics_reports_exact_amplitude_and_cosine() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    confidence = torch.ones(2)
    valid = torch.ones(2, dtype=torch.bool)
    metrics = _prediction_metrics(target, target, confidence, valid, zero_loss=1.0)
    assert metrics["validation_loss"] == 0.0
    assert metrics["mean_prediction_to_target_magnitude_ratio"] == 1.0
    assert metrics["high_10_cosine"] == 1.0
    assert metrics["relative_improvement_vs_zero_predictor"] == 1.0


def test_available_queries_omits_unavailable_expanded_graph() -> None:
    result = {
        "gt_query": {},
        "expanded_query": {"available": False, "reason": "not supplied"},
    }
    assert _available_queries(result) == ("gt_query",)

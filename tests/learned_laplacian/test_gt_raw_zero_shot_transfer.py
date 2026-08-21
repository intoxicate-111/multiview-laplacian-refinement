from __future__ import annotations

import math

import torch

from mlr.learned_laplacian.gt_raw_zero_shot_transfer import (
    _inference_only_sample,
    _raw_metrics_by_gt_magnitude,
    _safe_input_audit,
)


def test_gt_magnitude_groups_do_not_rank_by_prediction_error() -> None:
    target = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [100.0, 0.0, 0.0]]
    )
    prediction = target.clone()
    prediction[0, 0] += 50.0
    prediction[2, 0] += 2.0
    metrics = _raw_metrics_by_gt_magnitude(
        prediction,
        target,
        torch.ones(3),
        torch.ones(3, dtype=torch.bool),
    )
    assert metrics["top_1_raw_epe"] == 2.0
    assert metrics["top_10_raw_epe"] == 2.0
    assert math.isclose(metrics["bottom_90_raw_epe"], 25.0, abs_tol=1e-6)


def test_inference_only_sample_scrubs_targets_and_gt() -> None:
    vertices = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    target = torch.ones_like(vertices)
    sample = {
        "sample_id": "mesh__v00",
        "vertices": vertices,
        "laplacian_target": target,
        "raw_laplacian_target": target,
        "normalized_laplacian_target": target,
        "target_positions": target,
        "target_confidence": torch.full((4,), 0.5),
        "gt_vertices": target,
        "gt_faces": torch.zeros((1, 3), dtype=torch.long),
        "metadata": {
            "gt_mesh_path": "/not/allowed.obj",
            "target_constructor": "L_current@P_proxy",
            "proxy_definition": "P_proxy",
            "object_id": "mesh",
        },
    }
    safe = _inference_only_sample(sample)
    assert _safe_input_audit(safe)["passed"]
    assert torch.equal(safe["target_positions"], vertices)
    assert torch.equal(safe["target_confidence"], torch.ones(4))
    assert safe["metadata"] == {
        "object_id": "mesh",
        "inference_only_target_fields_zeroed": True,
    }

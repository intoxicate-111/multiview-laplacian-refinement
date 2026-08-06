from __future__ import annotations

import torch

from mlr.learned_laplacian.single_mesh_overfit import single_mesh_loss


def test_single_mesh_loss_variants_are_finite_and_zero_at_exact_target() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    confidence = torch.ones(2)
    for variant in (
        "base_fixed_0.01",
        "base_adaptive",
        "weighted_adaptive",
        "magnitude_direction",
    ):
        loss = single_mesh_loss(
            target, target, confidence, variant=variant, adaptive_delta=1.5
        )
        assert torch.isfinite(loss)
        assert loss.item() == 0.0


def test_weighted_loss_emphasizes_large_target_error() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    prediction = target.clone()
    prediction[1] = 0
    confidence = torch.ones(2)
    base = single_mesh_loss(
        prediction, target, confidence, variant="base_adaptive", adaptive_delta=1.0
    )
    weighted = single_mesh_loss(
        prediction,
        target,
        confidence,
        variant="weighted_adaptive",
        adaptive_delta=1.0,
        magnitude_weight_lambda=4.0,
    )
    assert weighted > base


def test_magnitude_direction_penalizes_wrong_direction() -> None:
    target = torch.tensor([[2.0, 0.0, 0.0]])
    correct = target.clone()
    opposite = -target
    confidence = torch.ones(1)
    correct_loss = single_mesh_loss(
        correct, target, confidence, variant="magnitude_direction", adaptive_delta=1.0
    )
    opposite_loss = single_mesh_loss(
        opposite, target, confidence, variant="magnitude_direction", adaptive_delta=1.0
    )
    assert opposite_loss > correct_loss

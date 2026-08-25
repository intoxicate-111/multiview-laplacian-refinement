from __future__ import annotations

import torch

from mlr.learned_laplacian.model import LearnedLaplacianModel

from .helpers import tiny_sample


def test_mesh_lambda_head_initializes_at_fixed_arm_b_value_and_has_gradients() -> None:
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
        recovery_lambda_head_enabled=True,
        recovery_lambda_head_hidden_dim=4,
        recovery_lambda_minimum=1e-3,
        recovery_lambda_maximum=1e-1,
        recovery_lambda_initial=1e-2,
    )
    output = model(tiny_sample())
    assert output.recovery_lambda is not None
    assert output.recovery_lambda_logit is not None
    torch.testing.assert_close(
        output.recovery_lambda,
        torch.tensor(1e-2),
        atol=2e-9,
        rtol=2e-7,
    )
    output.recovery_lambda.square().backward()
    gradients = [
        parameter.grad
        for parameter in model.recovery_lambda_head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(torch.linalg.vector_norm(gradient)) > 0 for gradient in gradients)


def test_model_without_lambda_head_emits_no_recovery_lambda() -> None:
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
    )
    output = model(tiny_sample())
    assert output.recovery_lambda is None
    assert output.recovery_lambda_logit is None

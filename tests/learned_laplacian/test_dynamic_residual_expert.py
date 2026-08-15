from __future__ import annotations

import torch

from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import (
    _freeze_except_dynamic_residual_expert,
    _load_initialization_checkpoint,
)

from .helpers import tiny_sample


def _model(*, expert: bool) -> LearnedLaplacianModel:
    return LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
        dynamic_residual_expert_enabled=expert,
        dynamic_residual_expert_hidden_dim=4,
        dynamic_gate_hidden_dim=5,
        dynamic_gate_initial_bias=0.1,
    )


def test_dynamic_expert_initially_preserves_base_and_has_positive_gate():
    sample = tiny_sample()
    torch.manual_seed(7)
    baseline = _model(expert=False)
    torch.manual_seed(7)
    expert = _model(expert=True)

    baseline_output = baseline(sample)
    expert_output = expert(sample)

    torch.testing.assert_close(
        expert_output.base_laplacian_prediction,
        baseline_output.predicted_laplacian,
    )
    torch.testing.assert_close(
        expert_output.predicted_laplacian,
        baseline_output.predicted_laplacian,
    )
    assert expert_output.dynamic_gate_logit is not None
    assert expert_output.dynamic_gate_signed is not None
    assert expert_output.dynamic_gate_effective is not None
    torch.testing.assert_close(
        expert_output.dynamic_gate_logit,
        torch.full_like(expert_output.dynamic_gate_logit, 0.1),
    )
    assert torch.all(expert_output.dynamic_gate_effective > 0)


def test_dynamic_gate_can_switch_expert_exactly_off():
    sample = tiny_sample()
    expert = _model(expert=True)
    with torch.no_grad():
        expert.dynamic_residual_expert[-1].bias.fill_(1.0)
        expert.dynamic_gate_head[-1].bias.fill_(-0.1)
    output = expert(sample)
    assert torch.count_nonzero(output.dynamic_gate_effective) == 0
    torch.testing.assert_close(
        output.predicted_laplacian, output.base_laplacian_prediction
    )


def test_base_checkpoint_load_and_freeze_leave_only_gate_and_residual(tmp_path):
    torch.manual_seed(7)
    baseline = _model(expert=False)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_state_dict": baseline.state_dict()}, checkpoint)

    torch.manual_seed(99)
    expert = _model(expert=True)
    _load_initialization_checkpoint(expert, checkpoint, torch.device("cpu"))
    _freeze_except_dynamic_residual_expert(expert)

    baseline_state = baseline.state_dict()
    expert_state = expert.state_dict()
    for name in baseline_state:
        torch.testing.assert_close(expert_state[name], baseline_state[name])
    trainable = {
        name for name, parameter in expert.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("dynamic_residual_expert.", "dynamic_gate_head."))
        for name in trainable
    )

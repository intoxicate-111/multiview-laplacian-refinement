from __future__ import annotations

import torch

from mlr.learned_laplacian.controlled_screening import arm_config
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import _oracle_top_magnitude_mask

from .helpers import tiny_sample


def test_oracle_top_magnitude_mask_selects_exact_per_mesh_top_ten_percent():
    target = torch.zeros((100, 3), dtype=torch.float32)
    target[:, 0] = torch.arange(100, dtype=torch.float32)
    mask = _oracle_top_magnitude_mask(target, torch.ones(100, dtype=torch.bool), 0.10)
    assert int(mask.sum()) == 10
    assert torch.all(mask[90:])
    assert not torch.any(mask[:90])


def test_residual_expert_preserves_initial_prediction_and_zeroes_bottom_rows():
    sample = tiny_sample()
    sample["oracle_high_signal_mask"] = torch.tensor([False, True, False, True])
    torch.manual_seed(7)
    baseline = LearnedLaplacianModel(
        image_feature_dim=8, hidden_dim=16, num_graph_layers=1, input_mode="coarse_only"
    )
    torch.manual_seed(7)
    expert = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
        oracle_residual_expert_enabled=True,
        oracle_residual_expert_hidden_dim=4,
    )
    baseline_output = baseline(sample)
    initial_expert_output = expert(sample)
    torch.testing.assert_close(
        initial_expert_output.predicted_laplacian, baseline_output.predicted_laplacian
    )
    assert initial_expert_output.oracle_residual_prediction is not None
    assert torch.count_nonzero(initial_expert_output.oracle_residual_prediction) == 0

    with torch.no_grad():
        expert.oracle_residual_expert[-1].bias.fill_(1.0)
    changed = expert(sample)
    assert changed.oracle_residual_prediction is not None
    assert torch.count_nonzero(changed.oracle_residual_prediction[~sample["oracle_high_signal_mask"]]) == 0
    torch.testing.assert_close(
        changed.predicted_laplacian[~sample["oracle_high_signal_mask"]],
        baseline_output.predicted_laplacian[~sample["oracle_high_signal_mask"]],
    )


def test_oracle_experiment_arms_keep_uniform_sampling_and_only_enable_e1_expert():
    base = {
        "seed": 7,
        "query_training": {
            "enabled": True,
            "normal_std_h": 0.0003,
            "tangent_std_h": 0.0003,
            "max_offset_h": 0.001,
            "apply_to_validation": True,
        },
        "model": {"hidden_dim": 64},
        "training": {},
        "multi_object_training": {"epochs": 1},
    }
    e0 = arm_config(base, "oracle_expert_e0", max_optimizer_steps=1000)
    e1 = arm_config(base, "oracle_expert_e1", max_optimizer_steps=1000)
    assert e0["training"]["vertex_sampling"] == {"mode": "full"}
    assert e1["training"]["vertex_sampling"] == {"mode": "full"}
    assert "oracle_residual_expert" not in e0["model"]
    assert e1["model"]["oracle_residual_expert"]["enabled"] is True
    assert e1["query_training"] == e0["query_training"]

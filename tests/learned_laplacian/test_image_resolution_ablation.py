from __future__ import annotations

import torch

from mlr.learned_laplacian.controlled_screening import arm_config
from mlr.learned_laplacian.image_encoder import SmallImageEncoder
from mlr.learned_laplacian.model import LearnedLaplacianModel


def test_second_stride_changes_only_feature_resolution():
    images = torch.zeros((2, 3, 96, 96))
    torch.manual_seed(7)
    baseline = SmallImageEncoder(feature_dim=16, second_stride=2)
    torch.manual_seed(7)
    high_res = SmallImageEncoder(feature_dim=16, second_stride=1)
    assert baseline(images).shape == (2, 16, 24, 24)
    assert high_res(images).shape == (2, 16, 48, 48)
    for left, right in zip(baseline.parameters(), high_res.parameters()):
        torch.testing.assert_close(left, right)


def test_model_initial_parameters_are_identical_across_feature_resolutions():
    torch.manual_seed(7)
    baseline = LearnedLaplacianModel(
        image_feature_dim=16, image_second_stride=2, hidden_dim=64
    )
    torch.manual_seed(7)
    high_res = LearnedLaplacianModel(
        image_feature_dim=16, image_second_stride=1, hidden_dim=64
    )
    assert baseline.state_dict().keys() == high_res.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, high_res.state_dict()[name])


def test_resolution_arm_configs_keep_uniform_training_and_only_change_stride():
    base = {
        "seed": 7,
        "query_training": {
            "enabled": True,
            "normal_std_h": 0.0003,
            "tangent_std_h": 0.0003,
            "max_offset_h": 0.001,
            "apply_to_validation": True,
        },
        "image_encoder": {"feature_dim": 16},
        "model": {"hidden_dim": 64},
        "training": {},
        "multi_object_training": {"epochs": 1},
    }
    f0 = arm_config(base, "image_resolution_f0", max_optimizer_steps=1000)
    f1 = arm_config(base, "image_resolution_f1", max_optimizer_steps=1000)
    assert f0["image_encoder"]["second_stride"] == 2
    assert f1["image_encoder"]["second_stride"] == 1
    assert f0["training"]["vertex_sampling"] == {"mode": "full"}
    assert f1["training"]["vertex_sampling"] == {"mode": "full"}
    assert f0["query_training"] == f1["query_training"]

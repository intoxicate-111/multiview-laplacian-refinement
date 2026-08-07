from __future__ import annotations

import torch

from mlr.learned_laplacian.controlled_screening import (
    arm_config,
    fixed_query_positions,
    target_group_masks,
)


def _base_config() -> dict:
    return {
        "seed": 7,
        "query_training": {
            "enabled": True,
            "exact_fraction": 0.2,
            "normal_std_h": 0.0003,
            "tangent_std_h": 0.0003,
            "max_offset_h": 0.001,
            "apply_to_validation": True,
        },
        "training": {},
        "multi_object_training": {"epochs": 2, "max_optimizer_steps": 20},
    }


def test_arm_configs_change_only_requested_query_or_sampling_control():
    base = _base_config()
    exact = arm_config(base, "exact_0000", max_optimizer_steps=1000)
    wide = arm_config(base, "support_0030", max_optimizer_steps=1000)
    importance = arm_config(base, "importance_0001", max_optimizer_steps=1000)
    assert exact["query_training"]["enabled"] is False
    assert exact["query_training"]["apply_to_validation"] is False
    assert wide["query_training"]["max_offset_h"] == 0.03
    assert wide["query_training"]["normal_std_h"] == 0.009
    assert importance["query_training"]["max_offset_h"] == 0.001
    assert (
        importance["training"]["vertex_sampling"]["mode"]
        == "high_laplacian_mixture_v1"
    )
    assert base["query_training"]["apply_to_validation"] is True


def test_geometry_aware_arm_configs_only_change_vertex_sampling_distribution():
    base = _base_config()
    strong = arm_config(base, "strong_importance_0001", max_optimizer_steps=1000)
    smooth = arm_config(base, "smooth_importance_0001", max_optimizer_steps=1000)
    assert strong["query_training"] == smooth["query_training"]
    assert strong["training"]["vertex_sampling"]["top_1_fraction"] == 0.25
    assert smooth["training"]["vertex_sampling"]["bottom_90_fraction"] == 0.50


def test_fixed_evaluation_queries_have_requested_relative_displacements():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    sample = {
        "sample_id": "mesh",
        "vertices": vertices,
        "vertex_normals": torch.tensor([[0.0, 0.0, 1.0]] * 4),
        "local_edge_length": torch.tensor([1.0, 2.0, 1.5, 0.5]),
    }
    queries = fixed_query_positions(sample, seed=7)
    torch.testing.assert_close(queries["exact"]["positions"], vertices)
    for name, expected in (
        ("near_0001", 0.001),
        ("moderate_0010", 0.01),
        ("expanded_0030", 0.03),
        ("large_0100", 0.10),
    ):
        torch.testing.assert_close(
            queries[name]["ratio"], torch.full((4,), expected)
        )


def test_target_groups_are_nested_and_cover_expected_counts():
    target = torch.zeros((100, 3))
    target[:, 0] = torch.arange(100, dtype=torch.float32)
    groups = target_group_masks(target, torch.ones(100, dtype=torch.bool))
    assert int(groups["lowest_10"].sum()) == 10
    assert int(groups["smooth_bottom_90"].sum()) == 90
    assert int(groups["high_top_10"].sum()) == 10
    assert int(groups["high_top_1"].sum()) == 1
    assert torch.all(groups["high_top_10"][groups["high_top_1"]])

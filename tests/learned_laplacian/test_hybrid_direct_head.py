from __future__ import annotations

import torch

from mlr.learned_laplacian.model import LearnedLaplacianModel

from .helpers import tiny_sample


def test_hybrid_direct_head_adds_one_mirrored_output_head() -> None:
    torch.manual_seed(7)
    baseline = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
    )
    torch.manual_seed(7)
    hybrid = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        input_mode="coarse_only",
        hybrid_direct_head_enabled=True,
    )
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    hybrid_parameters = sum(parameter.numel() for parameter in hybrid.parameters())
    expected_added = sum(parameter.numel() for parameter in hybrid.hybrid_direct_head.parameters())
    assert hybrid_parameters - baseline_parameters == expected_added == 323

    output = hybrid(tiny_sample())
    assert output.direct_vertex_displacement_prediction is not None
    assert output.direct_vertex_displacement_prediction.shape == output.predicted_laplacian.shape
    loss = (
        output.predicted_laplacian.square().mean()
        + output.direct_vertex_displacement_prediction.square().mean()
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in hybrid.hybrid_direct_head.parameters()
    )


def test_split_geometry_hybrid_forks_before_graph_processing() -> None:
    torch.manual_seed(7)
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=2,
        input_mode="coarse_only",
        hybrid_direct_head_enabled=True,
        split_geometry_towers_enabled=True,
    )
    assert model.hybrid_direct_head is None
    assert model.direct_predictor is not None
    groups = model.split_geometry_parameter_groups()
    lap_ids = {id(parameter) for parameter in groups["lap_tower"]}
    direct_ids = {id(parameter) for parameter in groups["direct_tower"]}
    shared_ids = {id(parameter) for parameter in groups["shared_frontend"]}
    assert lap_ids.isdisjoint(direct_ids)
    assert lap_ids.isdisjoint(shared_ids)
    assert direct_ids.isdisjoint(shared_ids)
    assert len(groups["lap_tower"]) == len(groups["direct_tower"])
    assert sum(p.numel() for p in groups["lap_tower"]) == sum(
        p.numel() for p in groups["direct_tower"]
    )

    output = model(tiny_sample())
    assert output.direct_vertex_displacement_prediction is not None
    assert output.direct_vertex_displacement_prediction.shape == (
        output.predicted_laplacian.shape
    )
    loss = (
        output.predicted_laplacian.square().mean()
        + output.direct_vertex_displacement_prediction.square().mean()
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in groups["lap_tower"])
    assert all(parameter.grad is not None for parameter in groups["direct_tower"])

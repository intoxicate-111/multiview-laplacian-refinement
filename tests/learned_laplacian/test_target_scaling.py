import math

import pytest
import torch

from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    denormalize_laplacian_by_edge_scale,
    graph_structure_statistics,
    incident_edge_length_and_valid_mask,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
    prediction_to_raw_laplacian,
    require_matching_laplacian_representations,
)
from mlr.laplacian import compute_laplacian_coordinates


def test_mean_incident_edge_length_uses_unique_undirected_edges():
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    directed_with_duplicates = torch.tensor(
        [[0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 2, 1], [1, 0, 0, 1, 2, 0, 0, 2, 2, 1, 1, 2]]
    )
    h = mean_incident_edge_length(vertices, directed_with_duplicates)
    expected = torch.tensor([1.0, (1.0 + math.sqrt(2.0)) / 2.0, (1.0 + math.sqrt(2.0)) / 2.0])
    torch.testing.assert_close(h, expected)


def test_regular_triangle_has_equal_scale_and_isolated_vertex_is_zero():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3.0) / 2.0, 0.0], [3.0, 3.0, 3.0]]
    )
    edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    h = mean_incident_edge_length(vertices, edges)
    torch.testing.assert_close(h[:3], torch.ones(3))
    assert h[3].item() == 0.0
    graph = graph_structure_statistics(edges, num_vertices=4)
    assert graph["unique_undirected_edge_count"] == 3
    assert graph["minimum_degree"] == 0
    assert graph["maximum_degree"] == 2
    assert graph["isolated_vertices"] == 1
    measured_h, valid = incident_edge_length_and_valid_mask(vertices, edges)
    torch.testing.assert_close(measured_h, h)
    torch.testing.assert_close(valid, torch.tensor([True, True, True, False]))


def test_invalid_scale_is_zeroed_instead_of_divided_by_epsilon():
    delta = torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]])
    h = torch.tensor([0.5, 0.0])
    valid = torch.tensor([True, False])

    normalized = normalize_laplacian_by_edge_scale(
        delta, h, eps=1e-12, valid_scale_mask=valid
    )

    torch.testing.assert_close(normalized[0], delta[0] / 0.25)
    torch.testing.assert_close(normalized[1], torch.zeros(3))
    assert torch.isfinite(normalized).all()


def test_normalize_then_denormalize_round_trips_with_exact_epsilon_convention():
    delta = torch.tensor([[0.2, -0.1, 0.3], [-0.5, 0.4, 0.1]], dtype=torch.float64)
    h = torch.tensor([0.5, 2.0], dtype=torch.float64)
    epsilon = 1e-4
    normalized = normalize_laplacian_by_edge_scale(delta, h, eps=epsilon)
    recovered = denormalize_laplacian_by_edge_scale(normalized, h, eps=epsilon)
    torch.testing.assert_close(recovered, delta, rtol=1e-14, atol=1e-14)


def test_prediction_conversion_is_applied_exactly_once_for_normalized_output():
    prediction = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    h = torch.tensor([0.25], dtype=torch.float64)
    epsilon = 1e-3
    raw = prediction_to_raw_laplacian(
        prediction,
        h,
        input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
        eps=epsilon,
    )
    torch.testing.assert_close(raw, prediction * (h.square() + epsilon).unsqueeze(-1))


def test_raw_prediction_is_not_multiplied_by_h2():
    prediction = torch.tensor([[2.0, -1.0, 0.5]])
    h = torch.tensor([0.25])
    raw = prediction_to_raw_laplacian(
        prediction, h, input_representation=RAW_LAPLACIAN
    )
    assert raw is prediction


def test_representation_mismatch_is_rejected_before_metrics():
    require_matching_laplacian_representations(RAW_LAPLACIAN, RAW_LAPLACIAN)
    with pytest.raises(ValueError, match="representation mismatch"):
        require_matching_laplacian_representations(
            EDGE_SCALE_NORMALIZED_LAPLACIAN, RAW_LAPLACIAN
        )


def test_global_scaling_laws_are_h_a_h2_a2_delta_a_and_delta_hat_inverse_a():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.3, 0.2, 0.7]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    edge_index = faces_to_edge_index(faces, len(vertices))
    h = mean_incident_edge_length(vertices, edge_index)
    delta = torch.from_numpy(compute_laplacian_coordinates(vertices.numpy(), faces.numpy(), "uniform"))
    delta_hat = normalize_laplacian_by_edge_scale(delta, h)

    for scale in (0.5, 2.0):
        scaled_vertices = vertices * scale
        scaled_h = mean_incident_edge_length(scaled_vertices, edge_index)
        scaled_delta = torch.from_numpy(
            compute_laplacian_coordinates(scaled_vertices.numpy(), faces.numpy(), "uniform")
        )
        scaled_delta_hat = normalize_laplacian_by_edge_scale(scaled_delta, scaled_h)
        torch.testing.assert_close(scaled_h, h * scale)
        torch.testing.assert_close(scaled_h.square(), h.square() * scale**2)
        torch.testing.assert_close(scaled_delta, delta * scale)
        torch.testing.assert_close(scaled_delta_hat, delta_hat / scale, rtol=1e-9, atol=1e-9)

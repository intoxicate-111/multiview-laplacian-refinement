import math

import torch

from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.target_scaling import (
    denormalize_laplacian_by_edge_scale,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
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


def test_normalize_then_denormalize_round_trips_when_epsilon_is_negligible():
    delta = torch.tensor([[0.2, -0.1, 0.3], [-0.5, 0.4, 0.1]])
    h = torch.tensor([0.5, 2.0])
    normalized = normalize_laplacian_by_edge_scale(delta, h, eps=1e-12)
    recovered = denormalize_laplacian_by_edge_scale(normalized, h)
    torch.testing.assert_close(recovered, delta, rtol=1e-5, atol=1e-7)


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

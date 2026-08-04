from __future__ import annotations

import torch


RAW_LAPLACIAN = "raw_laplacian"
EDGE_SCALE_NORMALIZED_LAPLACIAN = "edge_scale_normalized_laplacian"
TARGET_MODES = {RAW_LAPLACIAN, EDGE_SCALE_NORMALIZED_LAPLACIAN}
EDGE_SCALE_DEFINITION = "square_of_mean_incident_edge_length"
EDGE_SCALE_SOURCE = "input_prediction_mesh"


def mean_incident_edge_length(
    vertices: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return mean unique incident-edge length per vertex.

    Directed duplicates are canonicalised into unique undirected pairs. An
    isolated vertex receives exactly zero; callers can identify it explicitly
    and the normalisation denominator remains protected by ``eps``.
    """

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3].")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    num_vertices = vertices.shape[0]
    if edge_index.numel() == 0:
        return vertices.new_zeros((num_vertices,))
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_vertices:
        raise ValueError("edge_index contains an out-of-range vertex index.")
    pairs = _unique_undirected_pairs(edge_index)
    if pairs.numel() == 0:
        return vertices.new_zeros((num_vertices,))
    lengths = torch.linalg.vector_norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], dim=1)
    length_sum = vertices.new_zeros((num_vertices,))
    degree = vertices.new_zeros((num_vertices,))
    ones = torch.ones_like(lengths)
    length_sum.index_add_(0, pairs[:, 0], lengths)
    length_sum.index_add_(0, pairs[:, 1], lengths)
    degree.index_add_(0, pairs[:, 0], ones)
    degree.index_add_(0, pairs[:, 1], ones)
    return length_sum / degree.clamp_min(1.0)


def graph_structure_statistics(
    edge_index: torch.Tensor, num_vertices: int
) -> dict[str, float | int]:
    """Summarise unique undirected edges and incident-neighbour degrees."""

    if num_vertices < 1:
        raise ValueError("num_vertices must be positive.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.numel() > 0 and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_vertices
    ):
        raise ValueError("edge_index contains an out-of-range vertex index.")
    pairs = _unique_undirected_pairs(edge_index)
    degree = torch.zeros(num_vertices, dtype=torch.long, device=edge_index.device)
    if pairs.numel() > 0:
        ones = torch.ones(pairs.shape[0], dtype=torch.long, device=edge_index.device)
        degree.index_add_(0, pairs[:, 0], ones)
        degree.index_add_(0, pairs[:, 1], ones)
    values = degree.detach().double().cpu()
    return {
        "unique_undirected_edge_count": int(pairs.shape[0]),
        "minimum_degree": int(degree.min().item()),
        "median_degree": float(values.median().item()),
        "mean_degree": float(values.mean().item()),
        "p95_degree": float(torch.quantile(values, 0.95).item()),
        "maximum_degree": int(degree.max().item()),
        "isolated_vertices": int((degree == 0).sum().item()),
    }


def normalize_laplacian_by_edge_scale(
    laplacian: torch.Tensor,
    local_edge_length: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute delta_hat_i = delta_i / (h_i^2 + eps)."""

    _validate_transform_inputs(laplacian, local_edge_length, eps)
    return laplacian / (local_edge_length.square() + eps).unsqueeze(-1)


def denormalize_laplacian_by_edge_scale(
    normalized_laplacian: torch.Tensor,
    local_edge_length: torch.Tensor,
) -> torch.Tensor:
    """Compute graph-specific delta_i = delta_hat_i * h_i^2."""

    if normalized_laplacian.ndim != 2 or normalized_laplacian.shape[1] != 3:
        raise ValueError("normalized_laplacian must have shape [N, 3].")
    if tuple(local_edge_length.shape) != (normalized_laplacian.shape[0],):
        raise ValueError("local_edge_length must have shape [N].")
    return normalized_laplacian * local_edge_length.square().unsqueeze(-1)


def edge_scale_statistics(local_edge_length: torch.Tensor) -> dict[str, float | int]:
    if local_edge_length.ndim != 1:
        raise ValueError("local_edge_length must have shape [N].")
    values = local_edge_length.detach().double().cpu()
    scales = values.square()
    isolated = values <= 0
    return {
        "minimum_h": float(values.min().item()),
        "median_h": float(values.median().item()),
        "mean_h": float(values.mean().item()),
        "maximum_h": float(values.max().item()),
        "minimum_h2": float(scales.min().item()),
        "median_h2": float(scales.median().item()),
        "mean_h2": float(scales.mean().item()),
        "maximum_h2": float(scales.max().item()),
        "isolated_vertices": int(isolated.sum().item()),
    }


def vector_magnitude_statistics(vectors: torch.Tensor) -> dict[str, float]:
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape [N, 3].")
    magnitudes = torch.linalg.vector_norm(vectors.detach().double().cpu(), dim=-1)
    return {
        "minimum": float(magnitudes.min().item()),
        "median": float(magnitudes.median().item()),
        "mean": float(magnitudes.mean().item()),
        "p95": float(torch.quantile(magnitudes, 0.95).item()),
        "p99": float(torch.quantile(magnitudes, 0.99).item()),
        "maximum": float(magnitudes.max().item()),
    }


def _validate_transform_inputs(
    laplacian: torch.Tensor,
    local_edge_length: torch.Tensor,
    eps: float,
) -> None:
    if laplacian.ndim != 2 or laplacian.shape[1] != 3:
        raise ValueError("laplacian must have shape [N, 3].")
    if tuple(local_edge_length.shape) != (laplacian.shape[0],):
        raise ValueError("local_edge_length must have shape [N].")
    if eps <= 0:
        raise ValueError("eps must be positive.")


def _unique_undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    source, destination = edge_index.to(dtype=torch.long)
    lower = torch.minimum(source, destination)
    upper = torch.maximum(source, destination)
    pairs = torch.stack((lower, upper), dim=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.numel() == 0:
        return pairs.reshape(0, 2)
    return torch.unique(pairs, dim=0)

from __future__ import annotations

import torch
from torch import nn


def faces_to_edge_index(faces: torch.Tensor, num_vertices: int | None = None) -> torch.Tensor:
    """Build a unique directed edge list [2, E] from triangular faces."""

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    faces = faces.to(dtype=torch.long)
    if faces.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=faces.device)
    if num_vertices is not None and (int(faces.min()) < 0 or int(faces.max()) >= num_vertices):
        raise ValueError("faces contain an out-of-range vertex index.")
    undirected = torch.cat(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        dim=0,
    )
    directed = torch.cat((undirected, undirected.flip(1)), dim=0)
    directed = torch.unique(directed, dim=0)
    return directed.t().contiguous()


class GraphMessagePassingBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_vertices, channels = features.shape
        neighbour_sum = features.new_zeros((num_vertices, channels))
        neighbour_count = features.new_zeros((num_vertices, 1))
        if edge_index.numel() > 0:
            source, destination = edge_index
            neighbour_sum.index_add_(0, destination, features[source])
            neighbour_count.index_add_(
                0,
                destination,
                torch.ones((destination.numel(), 1), dtype=features.dtype, device=features.device),
            )
        neighbour_mean = neighbour_sum / neighbour_count.clamp_min(1.0)
        update = self.update(torch.cat((features, neighbour_mean), dim=-1))
        return self.activation(features + update)


class LaplacianPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_graph_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_graph_layers < 1:
            raise ValueError("input_dim, hidden_dim, and num_graph_layers must be positive.")
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            GraphMessagePassingBlock(hidden_dim, dropout=dropout)
            for _ in range(num_graph_layers)
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        vertex_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vertex_features: Tensor[N, C]
            edge_index: Tensor[2, E]

        Returns:
            predicted_laplacian: Tensor[N, 3]
        """

        if vertex_features.ndim != 2:
            raise ValueError("vertex_features must have shape [N, C].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        hidden = self.input_mlp(vertex_features)
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        return self.output_mlp(hidden)

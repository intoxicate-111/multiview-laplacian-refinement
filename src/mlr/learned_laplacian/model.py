from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .aggregation import masked_mean_aggregate
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .image_encoder import SmallImageEncoder
from .projection import sample_vertex_features
from .query_training import QUERY_FOURIER_GEOMETRY_MODE
from .target_scaling import mean_incident_edge_length


INPUT_MODES = {"coarse_only", "multiview_only", "coarse_plus_multiview"}
GEOMETRY_MODES = {"legacy", QUERY_FOURIER_GEOMETRY_MODE}


class FourierPositionEncoding(nn.Module):
    """Fourier features for shape-normalized three-dimensional query positions."""

    def __init__(self, num_frequencies: int = 6, include_input: bool = True) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive.")
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        frequencies = torch.pi * torch.pow(2.0, torch.arange(num_frequencies))
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + int(self.include_input))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        phases = positions.float().unsqueeze(-1) * self.frequencies.float()
        encoded = [
            torch.sin(phases).flatten(start_dim=1),
            torch.cos(phases).flatten(start_dim=1),
        ]
        if self.include_input:
            encoded.insert(0, positions.float())
        return torch.cat(encoded, dim=-1)


@dataclass(frozen=True)
class LearnedLaplacianOutput:
    predicted_laplacian: torch.Tensor
    vertex_features: torch.Tensor
    aggregated_image_features: torch.Tensor
    valid_view_ratio: torch.Tensor
    valid_views: torch.Tensor


class LearnedLaplacianModel(nn.Module):
    """CNN + differentiable projection + graph predictor for one prediction mesh."""

    GEOMETRY_DIM = 10  # position, normal, initial Laplacian, log(1 + degree)

    def __init__(
        self,
        image_feature_dim: int = 32,
        hidden_dim: int = 128,
        num_graph_layers: int = 3,
        dropout: float = 0.0,
        input_mode: str = "coarse_plus_multiview",
        zero_images: bool = False,
        geometry_mode: str = "legacy",
        position_num_frequencies: int = 6,
        position_include_input: bool = True,
    ) -> None:
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}, got {input_mode!r}.")
        if geometry_mode not in GEOMETRY_MODES:
            raise ValueError(
                f"geometry_mode must be one of {sorted(GEOMETRY_MODES)}, got {geometry_mode!r}."
            )
        self.image_feature_dim = image_feature_dim
        self.input_mode = input_mode
        self.zero_images = zero_images
        self.geometry_mode = geometry_mode
        self.position_encoder = FourierPositionEncoding(
            num_frequencies=position_num_frequencies,
            include_input=position_include_input,
        )
        self.image_encoder = SmallImageEncoder(image_feature_dim)
        geometry_dim = (
            self.GEOMETRY_DIM
            if geometry_mode == "legacy"
            else self.position_encoder.output_dim + 3 + 1 + 1
        )
        self.predictor = LaplacianPredictor(
            input_dim=geometry_dim + 1 + image_feature_dim,
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
        )

    def forward(self, sample: Mapping[str, Any]) -> LearnedLaplacianOutput:
        images = sample["images"]
        query_positions = sample.get("query_positions", sample["vertices"])
        if self.zero_images or self.input_mode == "coarse_only":
            feature_height = (images.shape[-2] + 1) // 2
            feature_height = (feature_height + 1) // 2
            feature_width = (images.shape[-1] + 1) // 2
            feature_width = (feature_width + 1) // 2
            feature_maps = images.new_zeros(
                (images.shape[0], self.image_feature_dim, feature_height, feature_width)
            )
        else:
            feature_maps = self.image_encoder(images)
        height, width = images.shape[-2:]
        per_view, valid, _ = sample_vertex_features(
            feature_maps=feature_maps,
            vertices=query_positions,
            intrinsics=sample["intrinsics"],
            extrinsics=sample["extrinsics"],
            image_size=(height, width),
            visibility=sample.get("visibility"),
        )
        aggregated, valid_ratio = masked_mean_aggregate(per_view, valid)

        edge_index = sample.get("edge_index")
        if edge_index is None:
            edge_index = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
        degree = sample.get("vertex_degree")
        if degree is None:
            degree = sample["vertices"].new_zeros((sample["vertices"].shape[0], 1))
            if edge_index.numel() > 0:
                degree.index_add_(
                    0,
                    edge_index[1],
                    torch.ones((edge_index.shape[1], 1), dtype=degree.dtype, device=degree.device),
                )
        if self.geometry_mode == "legacy":
            geometry = torch.cat(
                (
                    query_positions,
                    sample["vertex_normals"],
                    sample["initial_laplacian"],
                    torch.log1p(degree),
                ),
                dim=-1,
            )
        else:
            normalized_positions, position_scale = _normalize_query_positions(
                query_positions, sample
            )
            local_edge_length = sample.get("local_edge_length")
            if local_edge_length is None:
                local_edge_length = mean_incident_edge_length(
                    sample["vertices"].float(), edge_index
                )
            relative_h = local_edge_length.float() / position_scale
            geometry = torch.cat(
                (
                    self.position_encoder(normalized_positions),
                    sample["vertex_normals"].float(),
                    torch.log(relative_h.clamp_min(1e-8)).unsqueeze(-1),
                    torch.log1p(degree.float()),
                ),
                dim=-1,
            )
        ratio_feature = valid_ratio.unsqueeze(-1)
        if self.input_mode == "multiview_only":
            geometry = torch.zeros_like(geometry)
        if self.input_mode == "coarse_only":
            aggregated = torch.zeros_like(aggregated)
            ratio_feature = torch.zeros_like(ratio_feature)
        vertex_features = torch.cat((geometry, ratio_feature, aggregated), dim=-1)
        predicted = self.predictor(vertex_features, edge_index)
        return LearnedLaplacianOutput(
            predicted_laplacian=predicted,
            vertex_features=vertex_features,
            aggregated_image_features=aggregated,
            valid_view_ratio=valid_ratio,
            valid_views=valid,
        )

    def architecture_config(self) -> dict[str, Any]:
        first_linear = self.predictor.input_mlp[0]
        return {
            "image_feature_dim": self.image_feature_dim,
            "hidden_dim": first_linear.out_features,
            "num_graph_layers": len(self.predictor.blocks),
            "dropout": float(self.predictor.blocks[0].update[2].p),
            "input_mode": self.input_mode,
            "zero_images": self.zero_images,
            "geometry_mode": self.geometry_mode,
            "position_num_frequencies": self.position_encoder.num_frequencies,
            "position_include_input": self.position_encoder.include_input,
        }


def _normalize_query_positions(
    query_positions: torch.Tensor, sample: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    center = sample.get("position_normalization_center")
    scale = sample.get("position_normalization_scale")
    if center is None or scale is None:
        reference = sample["vertices"].float()
        center = 0.5 * (reference.amin(dim=0) + reference.amax(dim=0))
        scale = torch.linalg.vector_norm(reference - center, dim=-1).amax()
    center = torch.as_tensor(center, dtype=torch.float32, device=query_positions.device)
    scale = torch.as_tensor(
        scale, dtype=torch.float32, device=query_positions.device
    ).reshape(())
    scale = scale.clamp_min(1e-8)
    return (query_positions.float() - center) / scale, scale

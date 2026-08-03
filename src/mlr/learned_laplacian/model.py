from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .aggregation import masked_mean_aggregate
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .image_encoder import SmallImageEncoder
from .projection import sample_vertex_features


INPUT_MODES = {"coarse_only", "multiview_only", "coarse_plus_multiview"}


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
    ) -> None:
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}, got {input_mode!r}.")
        self.image_feature_dim = image_feature_dim
        self.input_mode = input_mode
        self.zero_images = zero_images
        self.image_encoder = SmallImageEncoder(image_feature_dim)
        self.predictor = LaplacianPredictor(
            input_dim=self.GEOMETRY_DIM + 1 + image_feature_dim,
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
        )

    def forward(self, sample: Mapping[str, Any]) -> LearnedLaplacianOutput:
        images = sample["images"]
        feature_maps = self.image_encoder(images)
        if self.zero_images or self.input_mode == "coarse_only":
            feature_maps = torch.zeros_like(feature_maps)
        height, width = images.shape[-2:]
        per_view, valid, _ = sample_vertex_features(
            feature_maps=feature_maps,
            vertices=sample["vertices"],
            intrinsics=sample["intrinsics"],
            extrinsics=sample["extrinsics"],
            image_size=(height, width),
            visibility=sample.get("visibility"),
        )
        aggregated, valid_ratio = masked_mean_aggregate(per_view, valid)

        edge_index = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
        degree = sample["vertices"].new_zeros((sample["vertices"].shape[0], 1))
        if edge_index.numel() > 0:
            degree.index_add_(
                0,
                edge_index[1],
                torch.ones((edge_index.shape[1], 1), dtype=degree.dtype, device=degree.device),
            )
        geometry = torch.cat(
            (
                sample["vertices"],
                sample["vertex_normals"],
                sample["initial_laplacian"],
                torch.log1p(degree),
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
        }

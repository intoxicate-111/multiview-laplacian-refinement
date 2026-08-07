from __future__ import annotations

import torch
from torch import nn


class SmallImageEncoder(nn.Module):
    """A small randomly initialized CNN; no pretrained weights or external backbone."""

    def __init__(self, feature_dim: int = 32, second_stride: int = 2) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if second_stride not in {1, 2}:
            raise ValueError("second_stride must be 1 or 2.")
        stem_dim = max(feature_dim // 2, 8)
        self.feature_dim = feature_dim
        self.second_stride = int(second_stride)
        self.network = nn.Sequential(
            nn.Conv2d(3, stem_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_dim,
                feature_dim,
                kernel_size=3,
                stride=self.second_stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [V, 3, H, W], got {tuple(images.shape)}.")
        return self.network(images)

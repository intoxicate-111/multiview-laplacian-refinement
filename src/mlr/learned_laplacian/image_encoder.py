from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_functional


IMAGE_FEATURE_CONSTRUCTION_MODES = {
    "original",
    "gaussian_blur",
    "original_plus_high_frequency",
}


class ImageFeatureConstructor(nn.Module):
    """Apply a fixed, parameter-free transformation to encoded image features."""

    def __init__(
        self,
        mode: str = "original",
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in IMAGE_FEATURE_CONSTRUCTION_MODES:
            raise ValueError(
                "mode must be one of "
                f"{sorted(IMAGE_FEATURE_CONSTRUCTION_MODES)}, got {mode!r}."
            )
        if gaussian_kernel_size < 1 or gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer.")
        if gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be positive.")
        self.mode = mode
        self.gaussian_kernel_size = int(gaussian_kernel_size)
        self.gaussian_sigma = float(gaussian_sigma)

        radius = self.gaussian_kernel_size // 2
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (coordinates / self.gaussian_sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        self.register_buffer(
            "gaussian_kernel",
            kernel_2d.reshape(1, 1, self.gaussian_kernel_size, self.gaussian_kernel_size),
            persistent=False,
        )

    @property
    def output_channel_multiplier(self) -> int:
        return 2 if self.mode == "original_plus_high_frequency" else 1

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        if feature_maps.ndim != 4:
            raise ValueError(
                "feature_maps must have shape [V, C, H, W], got "
                f"{tuple(feature_maps.shape)}."
            )
        if self.mode == "original":
            return feature_maps
        radius = self.gaussian_kernel_size // 2
        if feature_maps.shape[-2] <= radius or feature_maps.shape[-1] <= radius:
            raise ValueError(
                "Feature map spatial dimensions must exceed the Gaussian padding."
            )
        channels = int(feature_maps.shape[1])
        kernel = self.gaussian_kernel.to(
            device=feature_maps.device, dtype=feature_maps.dtype
        ).expand(channels, 1, -1, -1)
        padded = nn_functional.pad(
            feature_maps, (radius, radius, radius, radius), mode="reflect"
        )
        blurred = nn_functional.conv2d(padded, kernel, groups=channels)
        if self.mode == "gaussian_blur":
            return blurred
        return torch.cat((feature_maps, feature_maps - blurred), dim=1)


class SmallImageEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 32,
        first_stride: int = 2,
        second_stride: int = 2,
    ) -> None:
        super().__init__()

        if first_stride not in {1, 2}:
            raise ValueError("first_stride must be 1 or 2.")
        if second_stride not in {1, 2}:
            raise ValueError("second_stride must be 1 or 2.")

        stem_dim = max(feature_dim // 2, 8)

        self.feature_dim = feature_dim
        self.first_stride = int(first_stride)
        self.second_stride = int(second_stride)

        self.network = nn.Sequential(
            nn.Conv2d(
                3, stem_dim,
                kernel_size=5,
                stride=self.first_stride,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_dim, feature_dim,
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

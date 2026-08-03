from __future__ import annotations

import torch


def masked_mean_aggregate(
    per_view_features: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate [V, N, C] features and safely handle vertices with no valid views."""

    if per_view_features.ndim != 3:
        raise ValueError("per_view_features must have shape [V, N, C].")
    if tuple(valid.shape) != tuple(per_view_features.shape[:2]):
        raise ValueError("valid must have shape [V, N].")
    weights = valid.to(per_view_features.dtype).unsqueeze(-1)
    counts = weights.sum(dim=0)
    aggregated = (per_view_features * weights).sum(dim=0) / counts.clamp_min(1.0)
    aggregated = torch.where(counts > 0, aggregated, torch.zeros_like(aggregated))
    valid_ratio = counts.squeeze(-1) / float(max(per_view_features.shape[0], 1))
    return aggregated, valid_ratio

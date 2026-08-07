from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class HardVisibilityRecoveryMask:
    visibility_count: torch.Tensor
    visible_any: torch.Tensor
    laplacian_weight: torch.Tensor
    view_dim: int
    num_views: int


def confidence_aware_recovery_weight(
    visibility: torch.Tensor,
    confidence_prediction: torch.Tensor,
    *,
    num_vertices: int,
) -> torch.Tensor:
    """Return the canonical recovery weight ``visible * confidence``.

    Renderer visibility is always the strict gate: an invisible query receives
    exactly zero weight irrespective of the learned confidence value.
    """

    hard = hard_any_view_recovery_mask(visibility, num_vertices=num_vertices)
    confidence = torch.as_tensor(
        confidence_prediction,
        dtype=torch.float32,
        device=hard.laplacian_weight.device,
    )
    if tuple(confidence.shape) != (num_vertices,):
        raise ValueError("confidence_prediction must have shape [N].")
    if not torch.isfinite(confidence).all():
        raise ValueError("confidence_prediction must be finite.")
    return hard.laplacian_weight * confidence.clamp(0.0, 1.0)


def hard_any_view_recovery_mask(
    visibility: torch.Tensor, *, num_vertices: int
) -> HardVisibilityRecoveryMask:
    """Reduce renderer visibility to a strict binary recovery weight.

    Prepared Sofa samples use [views, vertices].  [vertices, views] is accepted
    for external callers, but ambiguous square tensors are rejected.
    """

    value = torch.as_tensor(visibility, dtype=torch.bool)
    if value.ndim != 2:
        raise ValueError("visibility must have shape [views, vertices] or [vertices, views].")
    matches = [dimension for dimension, size in enumerate(value.shape) if size == num_vertices]
    if len(matches) != 1:
        raise ValueError(
            "Exactly one visibility dimension must equal num_vertices; "
            f"got shape {tuple(value.shape)} and num_vertices={num_vertices}."
        )
    vertex_dim = matches[0]
    view_dim = 1 - vertex_dim
    count = value.sum(dim=view_dim, dtype=torch.int64)
    visible_any = count > 0
    # No epsilon is used: an all-view-invisible vertex receives exactly zero.
    weight = visible_any.to(dtype=torch.float32)
    return HardVisibilityRecoveryMask(
        visibility_count=count,
        visible_any=visible_any,
        laplacian_weight=weight,
        view_dim=view_dim,
        num_views=int(value.shape[view_dim]),
    )


def visibility_coverage_diagnostics(mask: HardVisibilityRecoveryMask) -> dict[str, Any]:
    count = mask.visibility_count.detach().cpu().numpy().astype(np.int64)
    visible = mask.visible_any.detach().cpu().numpy().astype(bool)
    histogram = np.bincount(count, minlength=mask.num_views + 1)
    return {
        "num_vertices": int(count.size),
        "num_views": int(mask.num_views),
        "num_visible_any": int(visible.sum()),
        "num_invisible_all": int((~visible).sum()),
        "visible_any_ratio": float(visible.mean()),
        "invisible_all_ratio": float((~visible).mean()),
        "mean_visible_view_count": float(count.mean()),
        "median_visible_view_count": float(np.median(count)),
        "min_visible_view_count": int(count.min()),
        "max_visible_view_count": int(count.max()),
        "visible_view_count_histogram": {
            str(index): int(value) for index, value in enumerate(histogram.tolist())
        },
        "visibility_tensor_semantics": "prepared [views, vertices] renderer-native boolean mask",
    }

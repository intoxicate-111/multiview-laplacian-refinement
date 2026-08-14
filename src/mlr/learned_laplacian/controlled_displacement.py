from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


CURRENT_GRAPH_LAPLACIAN = "current_graph_laplacian"
DIRECT_VERTEX_DISPLACEMENT = "direct_vertex_displacement"
PREDICTION_SEMANTICS = {
    CURRENT_GRAPH_LAPLACIAN,
    DIRECT_VERTEX_DISPLACEMENT,
}


def prediction_semantics(config: Mapping[str, Any]) -> str:
    """Return the output meaning while preserving legacy Laplacian configs."""

    value = str(config.get("prediction_semantics", CURRENT_GRAPH_LAPLACIAN))
    if value not in PREDICTION_SEMANTICS:
        raise ValueError(
            "prediction_semantics must be one of "
            f"{sorted(PREDICTION_SEMANTICS)}."
        )
    return value


def displacement_target(sample: Mapping[str, Any]) -> torch.Tensor:
    """Build the controlled baseline target ``P_proxy - P_current``."""

    current = sample.get("vertices")
    proxy = sample.get("target_positions")
    if not isinstance(current, torch.Tensor) or not isinstance(proxy, torch.Tensor):
        raise ValueError(
            "direct_vertex_displacement requires vertices and target_positions."
        )
    if tuple(current.shape) != tuple(proxy.shape) or current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("P_current and P_proxy must have identical [N, 3] shapes.")
    target = proxy.to(device=current.device, dtype=current.dtype) - current
    if not torch.isfinite(target).all():
        raise ValueError("Direct displacement target contains NaN or infinite values.")
    return target


def recover_direct_displacement(
    current_vertices: torch.Tensor | np.ndarray,
    displacement_prediction: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Recover ``P_refined`` by direct addition, without a Laplacian solver."""

    if isinstance(current_vertices, torch.Tensor):
        prediction = torch.as_tensor(
            displacement_prediction,
            dtype=current_vertices.dtype,
            device=current_vertices.device,
        )
        if tuple(prediction.shape) != tuple(current_vertices.shape):
            raise ValueError("Displacement prediction must match current vertex shape.")
        refined = current_vertices + prediction
        if not torch.isfinite(refined).all():
            raise ValueError("Direct displacement recovery produced non-finite vertices.")
        return refined

    current = np.asarray(current_vertices)
    prediction = np.asarray(displacement_prediction, dtype=current.dtype)
    if prediction.shape != current.shape:
        raise ValueError("Displacement prediction must match current vertex shape.")
    refined = current + prediction
    if not np.isfinite(refined).all():
        raise ValueError("Direct displacement recovery produced non-finite vertices.")
    return refined

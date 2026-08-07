from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data

from .graph_layers import faces_to_edge_index
from .target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    mean_incident_edge_length,
    prediction_to_raw_laplacian,
)
from .visibility_recovery import (
    confidence_aware_recovery_weight,
    hard_any_view_recovery_mask,
)


@dataclass(frozen=True)
class CanonicalRecoveryInputs:
    """Explicitly named tensors entering current-graph Laplacian recovery."""

    delta_hat_prediction: torch.Tensor
    delta_pred_raw: torch.Tensor
    h_current: torch.Tensor
    delta_current_raw: torch.Tensor
    visible: torch.Tensor
    confidence_prediction: torch.Tensor
    weight: torch.Tensor


def canonical_current_graph_recovery_inputs(
    current_vertices: torch.Tensor | np.ndarray,
    current_faces: torch.Tensor | np.ndarray,
    delta_hat_prediction: torch.Tensor,
    visibility: torch.Tensor,
    confidence_prediction: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> CanonicalRecoveryInputs:
    """Convert an absolute normalized prediction exactly once on the current graph."""

    vertices = torch.as_tensor(current_vertices)
    faces = torch.as_tensor(current_faces, dtype=torch.long, device=vertices.device)
    prediction = torch.as_tensor(
        delta_hat_prediction, dtype=vertices.dtype, device=vertices.device
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("current_vertices must have shape [N, 3].")
    if tuple(prediction.shape) != tuple(vertices.shape):
        raise ValueError("delta_hat_prediction must have shape [N, 3].")
    edge_index = faces_to_edge_index(faces, vertices.shape[0])
    h_current = mean_incident_edge_length(vertices, edge_index, eps=epsilon)
    delta_pred_raw = prediction_to_raw_laplacian(
        prediction,
        h_current,
        input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
        eps=epsilon,
    )
    delta_current_raw = torch.as_tensor(
        _current_uniform_laplacian(vertices, faces),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    hard = hard_any_view_recovery_mask(
        visibility, num_vertices=int(vertices.shape[0])
    )
    if confidence_prediction is None:
        confidence = torch.ones_like(h_current, dtype=torch.float32)
        weight = hard.laplacian_weight
    else:
        confidence = torch.as_tensor(
            confidence_prediction,
            dtype=torch.float32,
            device=hard.laplacian_weight.device,
        ).clamp(0.0, 1.0)
        weight = confidence_aware_recovery_weight(
            visibility, confidence, num_vertices=int(vertices.shape[0])
        )
    return CanonicalRecoveryInputs(
        delta_hat_prediction=prediction,
        delta_pred_raw=delta_pred_raw,
        h_current=h_current,
        delta_current_raw=delta_current_raw,
        visible=hard.visible_any,
        confidence_prediction=confidence,
        weight=weight,
    )


def _current_uniform_laplacian(
    vertices: torch.Tensor, faces: torch.Tensor
) -> np.ndarray:
    positions = vertices.detach().cpu().numpy().astype(np.float64, copy=False)
    triangles = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    operator = build_uniform_laplacian_data(triangles, len(positions))
    return apply_uniform_laplacian(positions, operator)

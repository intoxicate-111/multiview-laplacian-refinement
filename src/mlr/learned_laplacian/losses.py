from __future__ import annotations

import torch
from torch.nn import functional as F


def weighted_robust_laplacian_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 0.01,
    charbonnier_epsilon: float = 1e-3,
    target_magnitude_weight_lambda: float = 0.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both have shape [N, 3].")
    if tuple(confidence.shape) != (prediction.shape[0],):
        raise ValueError("confidence must have shape [N].")
    residual = prediction - target
    if loss_type == "huber":
        per_component = F.huber_loss(prediction, target, delta=huber_delta, reduction="none")
    elif loss_type == "charbonnier":
        per_component = torch.sqrt(residual.square() + charbonnier_epsilon**2) - charbonnier_epsilon
    else:
        raise ValueError("loss_type must be 'huber' or 'charbonnier'.")
    per_vertex = per_component.mean(dim=-1)
    if target_magnitude_weight_lambda < 0:
        raise ValueError("target_magnitude_weight_lambda must be non-negative.")
    weights = confidence.clamp_min(0.0)
    if target_magnitude_weight_lambda > 0:
        target_magnitude = torch.linalg.vector_norm(target.float(), dim=-1)
        weighted_mean_magnitude = (
            weights * target_magnitude
        ).sum() / weights.sum().clamp_min(1e-12)
        magnitude_weights = 1.0 + target_magnitude_weight_lambda * target_magnitude / (
            weighted_mean_magnitude + 1e-12
        )
        weights = weights * magnitude_weights
    return (weights * per_vertex).sum() / weights.sum().clamp_min(1e-12)


@torch.no_grad()
def laplacian_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match.")
    if valid_mask is not None:
        if tuple(valid_mask.shape) != (prediction.shape[0],):
            raise ValueError("valid_mask must have shape [N].")
        valid_mask = valid_mask.to(dtype=torch.bool, device=prediction.device)
        prediction = prediction[valid_mask]
        target = target[valid_mask]
        if prediction.shape[0] == 0:
            raise ValueError("valid_mask must select at least one vertex.")
    residual = prediction - target
    endpoint = torch.linalg.vector_norm(residual, dim=-1)
    pred_magnitude = torch.linalg.vector_norm(prediction, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return {
        "mse": float(residual.square().mean().item()),
        "mean_absolute_error": float(residual.abs().mean().item()),
        "vector_endpoint_error": float(endpoint.mean().item()),
        "magnitude_error": float((pred_magnitude - target_magnitude).abs().mean().item()),
        "cosine_similarity": float(cosine.mean().item()),
    }

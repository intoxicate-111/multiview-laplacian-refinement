from __future__ import annotations

import torch
from torch.nn import functional as F


def robust_laplacian_error_per_vertex(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "huber",
    huber_delta: float = 0.01,
    charbonnier_epsilon: float = 1e-3,
) -> torch.Tensor:
    """Return the robust three-component mean error for every query vertex."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must both have shape [N, 3].")
    residual = prediction - target
    if loss_type == "huber":
        per_component = F.huber_loss(
            prediction, target, delta=huber_delta, reduction="none"
        )
    elif loss_type == "charbonnier":
        per_component = (
            torch.sqrt(residual.square() + charbonnier_epsilon**2)
            - charbonnier_epsilon
        )
    else:
        raise ValueError("loss_type must be 'huber' or 'charbonnier'.")
    return per_component.mean(dim=-1)


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
    per_vertex = robust_laplacian_error_per_vertex(
        prediction,
        target,
        loss_type=loss_type,
        huber_delta=huber_delta,
        charbonnier_epsilon=charbonnier_epsilon,
    )
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


def confidence_reliability_loss(
    confidence_prediction: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision_weight: torch.Tensor,
    *,
    regularizer: float = 0.01,
    minimum_confidence: float = 1e-4,
    loss_type: str = "huber",
    huber_delta: float = 0.01,
    charbonnier_epsilon: float = 1e-3,
) -> torch.Tensor:
    """Train confidence as inverse predictive uncertainty without an oracle input.

    The detached local prediction error supplies the heteroscedastic likelihood
    signal.  ``-regularizer * log(confidence)`` prevents the trivial all-zero
    solution.  Ground-truth error is never an inference feature or recovery
    weight.
    """

    if tuple(confidence_prediction.shape) != (prediction.shape[0],):
        raise ValueError("confidence_prediction must have shape [N].")
    if tuple(supervision_weight.shape) != (prediction.shape[0],):
        raise ValueError("supervision_weight must have shape [N].")
    if regularizer <= 0:
        raise ValueError("regularizer must be positive.")
    if not 0 < minimum_confidence < 1:
        raise ValueError("minimum_confidence must be between zero and one.")
    error = robust_laplacian_error_per_vertex(
        prediction,
        target,
        loss_type=loss_type,
        huber_delta=huber_delta,
        charbonnier_epsilon=charbonnier_epsilon,
    ).detach()
    confidence = confidence_prediction.float().clamp(
        min=minimum_confidence, max=1.0
    )
    weight = supervision_weight.float().clamp_min(0.0)
    per_vertex = confidence * error - regularizer * torch.log(confidence)
    return (weight * per_vertex).sum() / weight.sum().clamp_min(1e-12)


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
    flattened_cosine = F.cosine_similarity(
        prediction.reshape(1, -1), target.reshape(1, -1), dim=-1, eps=1e-8
    )
    count = int(target_magnitude.numel())
    top10_count = max(1, int(round(0.10 * count)))
    top1_count = max(1, int(round(0.01 * count)))
    top10 = torch.topk(target_magnitude, k=top10_count).indices
    top1 = torch.topk(target_magnitude, k=top1_count).indices
    target_norm = torch.linalg.vector_norm(target)
    prediction_norm = torch.linalg.vector_norm(prediction)
    return {
        "mse": float(residual.square().mean().item()),
        "mean_absolute_error": float(residual.abs().mean().item()),
        "vector_endpoint_error": float(endpoint.mean().item()),
        "magnitude_error": float((pred_magnitude - target_magnitude).abs().mean().item()),
        "cosine_similarity": float(cosine.mean().item()),
        "mean_per_vertex_cosine": float(cosine.mean().item()),
        "global_cosine": float(flattened_cosine.item()),
        "top_10_percent_cosine": float(cosine[top10].mean().item()),
        "top_1_percent_cosine": float(cosine[top1].mean().item()),
        "top_10_percent_vector_endpoint_error": float(endpoint[top10].mean().item()),
        "top_1_percent_vector_endpoint_error": float(endpoint[top1].mean().item()),
        "prediction_to_target_norm_ratio": float(
            (prediction_norm / target_norm.clamp_min(1e-12)).item()
        ),
    }


@torch.no_grad()
def confidence_calibration_metrics(
    confidence_prediction: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    quantile_bins: int = 5,
) -> dict[str, object]:
    """Measure whether higher learned confidence corresponds to lower error."""

    confidence = confidence_prediction.float()
    error = torch.linalg.vector_norm(prediction.float() - target.float(), dim=-1)
    if tuple(confidence.shape) != tuple(error.shape):
        raise ValueError("confidence_prediction must have shape [N].")
    if valid_mask is not None:
        mask = valid_mask.to(dtype=torch.bool, device=confidence.device)
        confidence = confidence[mask]
        error = error[mask]
    if confidence.numel() < 1:
        raise ValueError("confidence calibration requires at least one valid vertex.")
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least two.")
    centered_confidence = confidence - confidence.mean()
    centered_negative_error = -error - (-error).mean()
    denominator = torch.sqrt(
        centered_confidence.square().sum()
        * centered_negative_error.square().sum()
    )
    correlation = (
        float((centered_confidence * centered_negative_error).sum().item() / denominator.item())
        if denominator.item() > 0
        else 0.0
    )
    order = torch.argsort(confidence)
    bins = []
    for index, indices in enumerate(torch.tensor_split(order, quantile_bins)):
        if indices.numel() == 0:
            continue
        bins.append(
            {
                "bin": int(index + 1),
                "mean_confidence": float(confidence[indices].mean().item()),
                "normalized_laplacian_error": float(error[indices].mean().item()),
                "vertex_count": int(indices.numel()),
            }
        )
    return {
        "mean": float(confidence.mean().item()),
        "minimum": float(confidence.min().item()),
        "maximum": float(confidence.max().item()),
        "correlation_with_negative_error": correlation,
        "bins": bins,
    }

from __future__ import annotations

import numpy as np

from .data import Array


def robust_value_and_grad(
    residual: Array,
    loss_type: str = "charbonnier",
    epsilon: float = 1e-3,
    huber_delta: float = 1e-2,
) -> tuple[float, Array]:
    residual = np.asarray(residual, dtype=np.float64)
    if loss_type == "l2":
        return float(0.5 * np.sum(residual * residual)), residual
    if loss_type == "charbonnier":
        squared = residual * residual
        value = np.sqrt(squared + epsilon * epsilon)
        grad = residual / value
        return float(np.sum(value)), grad
    if loss_type == "huber":
        abs_r = np.abs(residual)
        quadratic = abs_r <= huber_delta
        value = np.where(
            quadratic,
            0.5 * residual * residual,
            huber_delta * (abs_r - 0.5 * huber_delta),
        )
        grad = np.where(quadratic, residual, huber_delta * np.sign(residual))
        return float(np.sum(value)), grad
    raise ValueError(f"Unsupported robust loss: {loss_type}")

from __future__ import annotations

import numpy as np

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data


def initial_uniform_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute the exact uniform-Laplacian baseline used by sparse recovery."""

    positions = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    data = build_uniform_laplacian_data(triangles, len(positions))
    return apply_uniform_laplacian(positions, data)


def compose_absolute_laplacian_target(
    delta_initial: np.ndarray,
    delta_predicted_absolute: np.ndarray,
    scale: float,
    correction_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate from the initial to a predicted absolute Laplacian target.

    Visibility/precision weights gate only the learned correction.  They never
    remove the initial-geometry Laplacian equations from recovery.
    """

    initial = np.asarray(delta_initial, dtype=np.float64)
    predicted = np.asarray(delta_predicted_absolute, dtype=np.float64)
    if initial.shape != predicted.shape or initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("Initial and predicted Laplacians must share shape [N, 3].")
    weight = _correction_weight(correction_weight, len(initial))
    return initial + float(scale) * weight[:, None] * (predicted - initial)


def compose_residual_laplacian_target(
    delta_initial: np.ndarray,
    delta_predicted_residual: np.ndarray,
    scale: float,
    correction_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Add a predicted residual while retaining the initial Laplacian baseline."""

    initial = np.asarray(delta_initial, dtype=np.float64)
    residual = np.asarray(delta_predicted_residual, dtype=np.float64)
    if initial.shape != residual.shape or initial.ndim != 2 or initial.shape[1] != 3:
        raise ValueError("Initial and residual Laplacians must share shape [N, 3].")
    weight = _correction_weight(correction_weight, len(initial))
    return initial + float(scale) * weight[:, None] * residual


def same_topology_oracle_target(
    current_vertices: np.ndarray,
    target_vertices: np.ndarray,
    current_faces: np.ndarray,
    target_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact current-graph residual oracle for ordered topology pairs."""

    current = np.asarray(current_vertices, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    faces = np.asarray(current_faces, dtype=np.int64)
    other_faces = np.asarray(target_faces, dtype=np.int64)
    if current.shape != target.shape:
        raise ValueError("Control and perturbed vertices must have identical shape/order.")
    if faces.shape != other_faces.shape or not np.array_equal(faces, other_faces):
        raise ValueError("Control and perturbed faces must be exactly identical and ordered.")
    delta_initial = initial_uniform_laplacian(current, faces)
    delta_target = initial_uniform_laplacian(target, faces)
    return delta_initial, delta_target, delta_target - delta_initial


def _correction_weight(weight: np.ndarray | None, num_vertices: int) -> np.ndarray:
    if weight is None:
        return np.ones(num_vertices, dtype=np.float64)
    result = np.asarray(weight, dtype=np.float64).reshape(-1)
    if result.shape != (num_vertices,):
        raise ValueError("correction_weight must have shape [N].")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("correction_weight must be finite and non-negative.")
    return result

from __future__ import annotations

import numpy as np

from .data import Array


def correspondence_metrics(vertices: Array, gt_vertices: Array) -> dict[str, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    gt_vertices = np.asarray(gt_vertices, dtype=np.float64)
    residual = vertices - gt_vertices
    distances = np.linalg.norm(residual, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(distances * distances))),
        "mae": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }

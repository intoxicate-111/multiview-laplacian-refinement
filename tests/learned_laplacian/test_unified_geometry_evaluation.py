from __future__ import annotations

import numpy as np

from mlr.data import Mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry


def _triangle(offset: float = 0.0) -> Mesh:
    return Mesh(
        np.asarray(
            [[0.0, 0.0, offset], [1.0, 0.0, offset], [0.0, 1.0, offset]],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    ).ensure_normals()


def test_unified_geometry_metrics_are_exact_for_identical_meshes() -> None:
    metrics = evaluate_mesh_geometry(
        _triangle(), _triangle(), surface_samples=128, seed=7, fscore_threshold=0.01
    )
    assert np.isclose(metrics["chamfer"], 0.0, atol=1e-12)
    assert np.isclose(
        metrics["point_to_surface_bidirectional_mean"], 0.0, atol=1e-12
    )
    assert np.isclose(
        metrics["point_to_surface_bidirectional_p95"], 0.0, atol=1e-12
    )
    assert metrics["fscore"] == 1.0
    assert metrics["normal_consistency"] == 1.0


def test_unified_geometry_metrics_use_fixed_threshold_and_seed() -> None:
    first = evaluate_mesh_geometry(
        _triangle(0.02),
        _triangle(),
        surface_samples=128,
        seed=11,
        fscore_threshold=0.01,
    )
    second = evaluate_mesh_geometry(
        _triangle(0.02),
        _triangle(),
        surface_samples=128,
        seed=11,
        fscore_threshold=0.01,
    )
    assert first == second
    assert np.isclose(first["chamfer"], 0.02)
    assert np.isclose(first["point_to_surface_bidirectional_p95"], 0.02)
    assert first["fscore"] == 0.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sofa50_exact_solve_visibility_sweep.py"
SPEC = importlib.util.spec_from_file_location("exact_solve_visibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_sparse_solve_recovers_vertices_up_to_component_gauge() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, -0.2], [0.0, 1.0, 0.3]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    laplacian, data = MODULE.uniform_sparse_laplacian(faces, len(vertices))
    count, labels = MODULE.component_labels(data)
    centroids = MODULE.component_centroids(vertices, labels, count)
    target = laplacian @ vertices
    solved, audit = MODULE.exact_sparse_solve(
        laplacian, target, labels, count, centroids, atol=1e-13, btol=1e-13, maxiter=1000
    )
    assert audit["all_converged"]
    assert np.max(np.abs(solved - vertices)) < 1e-10


def test_initial_gauge_changes_only_component_translation() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = np.asarray([0, 0])
    shifted = MODULE.shift_component_gauge(
        vertices,
        labels,
        np.asarray([[0.5, 0.0, 0.0]]),
        np.asarray([[0.5, 2.0, -1.0]]),
    )
    np.testing.assert_allclose(shifted - vertices, [[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]])

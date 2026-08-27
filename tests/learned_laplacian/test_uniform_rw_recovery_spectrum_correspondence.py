from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_sofa50_uniform_rw_recovery_spectrum_correspondence import (  # noqa: E402
    band_subspace_overlap,
    laplacian_right_modes,
    recovery_mode_effective_frequency,
    symmetric_random_walk_similarity,
)
from diagnose_sofa50_exact_solve_visibility_sweep import (  # noqa: E402
    uniform_sparse_laplacian,
)


def _irregular_mesh_operator():
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64)
    laplacian, data = uniform_sparse_laplacian(faces, 5)
    degrees = np.asarray([len(neighbors) for neighbors in data.neighbors], dtype=np.float64)
    return laplacian, degrees


def test_random_walk_similarity_and_lambda_squared_identity() -> None:
    laplacian, degrees = _irregular_mesh_operator()
    symmetric, _, inverse_sqrt_degree = symmetric_random_walk_similarity(
        laplacian, degrees
    )
    np.testing.assert_allclose(symmetric.toarray(), symmetric.toarray().T, atol=1e-14)
    values, vectors = np.linalg.eigh(symmetric.toarray())
    selected = values > 1e-10
    values = values[selected]
    right = laplacian_right_modes(vectors[:, selected], inverse_sqrt_degree)
    np.testing.assert_allclose(
        laplacian @ right, right * values[None, :], rtol=1e-11, atol=1e-12
    )
    recovery = laplacian.T @ laplacian
    response = np.einsum("ij,ij->j", right, recovery @ right)
    np.testing.assert_allclose(response, np.square(values), rtol=1e-11, atol=1e-12)


def test_recovery_eigenvalue_is_euclidean_laplacian_response() -> None:
    laplacian, degrees = _irregular_mesh_operator()
    recovery = (laplacian.T @ laplacian).toarray()
    values, vectors = np.linalg.eigh(recovery)
    selected = values > 1e-10
    values = values[selected]
    vectors = vectors[:, selected]
    _, rms = recovery_mode_effective_frequency(vectors, laplacian, degrees)
    euclidean = np.linalg.norm(laplacian @ vectors, axis=0)
    np.testing.assert_allclose(euclidean, np.sqrt(values), rtol=1e-11, atol=1e-12)
    assert np.all(np.isfinite(rms))
    assert np.all(rms > 0)


def test_band_subspace_overlap_detects_identical_bases() -> None:
    basis = np.eye(6)
    overlap = band_subspace_overlap(basis, basis, modes_per_band=2)
    np.testing.assert_allclose(overlap, np.eye(3), atol=1e-12)

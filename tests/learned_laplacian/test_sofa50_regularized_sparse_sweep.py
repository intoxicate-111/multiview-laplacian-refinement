from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sofa50_regularized_sparse_sweep.py"
SPEC = importlib.util.spec_from_file_location("regularized_sparse_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_requested_lambda_and_family_contract() -> None:
    assert MODULE.LAMBDAS == (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    assert MODULE.FAMILIES == ("predicted_raw", "predicted_zero_mean", "exact_target")


def test_component_zero_mean_is_per_component_and_coordinate() -> None:
    values = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0], [10.0, 3.0, -1.0]])
    labels = np.array([0, 0, 1])
    projected = MODULE.component_zero_mean(values, labels, 2)
    np.testing.assert_allclose(projected[:2].mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(projected[2:].mean(axis=0), 0.0, atol=1e-14)


def test_regularized_solver_matches_closed_form() -> None:
    laplacian = csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    target = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    initial = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    labels = np.zeros(2, dtype=np.int64)
    regularization = 0.1
    solved, audit = MODULE.regularized_sparse_solve(
        laplacian,
        target,
        initial,
        labels,
        1,
        regularization,
        atol=1e-14,
        btol=1e-14,
        maxiter=1000,
    )
    matrix = laplacian.toarray()
    expected = np.linalg.solve(
        matrix.T @ matrix + regularization * np.eye(2),
        matrix.T @ target + regularization * initial,
    )
    np.testing.assert_allclose(solved, expected, atol=1e-10)
    assert audit["all_converged"]
    assert audit["system"] == "[L; sqrt(lambda) I]"

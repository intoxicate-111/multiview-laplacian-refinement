from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_sofa50_recovery_operator_spectrum import (  # noqa: E402
    _solve_shifted,
    operator_band_energies,
)


def test_operator_bands_partition_energy_and_localize_interior_modes() -> None:
    operator = diags([0.09, 0.45, 0.81], format="csr")
    bands = {
        "low": (0.0, 1.0 / 3.0),
        "mid": (1.0 / 3.0, 2.0 / 3.0),
        "high": (2.0 / 3.0, 1.0),
    }
    for index, expected_band in enumerate(bands):
        signal = np.zeros((3, 3), dtype=np.float64)
        signal[index, 0] = 1.0
        result = operator_band_energies(
            operator,
            0.9,
            signal,
            bands,
            order=128,
        )
        assert np.isclose(
            sum(result[f"{name}_energy"] for name in bands),
            result["total_energy"],
            rtol=3e-5,
            atol=3e-5,
        )
        assert result[f"{expected_band}_energy"] > 0.95


def test_shifted_solve_matches_exact_modal_transfer_identity() -> None:
    eigenvalues = np.asarray([0.0, 0.2, 2.0], dtype=np.float64)
    operator = diags(eigenvalues, format="csr")
    regularization = 0.03
    b_dagger = np.asarray(
        [[1.0, -2.0, 0.5], [0.2, 0.3, -0.4], [2.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    direct = np.asarray(
        [[-0.5, 0.7, 2.0], [1.0, -1.0, 0.2], [0.4, 0.8, 1.2]],
        dtype=np.float64,
    )
    hybrid, _ = _solve_shifted(
        operator,
        operator @ b_dagger + regularization * direct,
        regularization,
    )
    b_contribution, _ = _solve_shifted(
        operator,
        operator @ b_dagger,
        regularization,
    )
    e_contribution, _ = _solve_shifted(
        operator,
        regularization * direct,
        regularization,
    )
    b_weight = eigenvalues / (eigenvalues + regularization)
    expected = (
        b_weight[:, None] * b_dagger
        + (1.0 - b_weight)[:, None] * direct
    )
    np.testing.assert_allclose(hybrid, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        hybrid,
        b_contribution + e_contribution,
        rtol=1e-12,
        atol=1e-12,
    )

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_sofa50_recovery_operator_spectrum import (  # noqa: E402
    _indicator_coefficients_unit,
)
from analyze_sofa50_resolution_spectrum_recovery_quality import (  # noqa: E402
    mesh_geometry_statistics,
    spectral_summary_from_moments,
    standardized_coefficient,
    stochastic_chebyshev_moments,
)


def test_mesh_geometry_statistics_use_unique_edges() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    result = mesh_geometry_statistics(vertices, faces)
    assert np.isclose(result["surface_area"], 1.0)
    assert result["unique_edges"] == 5
    assert np.isclose(result["vertex_density"], 4.0)


def test_stochastic_spectral_fractions_match_exact_path_spectrum() -> None:
    size = 80
    laplacian = diags(
        (
            -np.ones(size - 1),
            np.r_[1.0, np.full(size - 2, 2.0), 1.0],
            -np.ones(size - 1),
        ),
        (-1, 0, 1),
        shape=(size, size),
        format="csr",
    )
    # Scale the squared operator so both fixed gate thresholds intersect its
    # spectrum, while preserving the component-constant nullspace.
    operator = (0.05 * laplacian.T @ laplacian).tocsr()
    exact = np.linalg.eigvalsh(operator.toarray())
    maximum = float(exact[-1]) * (1.0 + 1e-10)
    labels = np.zeros(size, dtype=np.int64)
    order = 512
    moments, norms = stochastic_chebyshev_moments(
        operator, maximum, labels, order=order, probes=64, seed=7
    )
    grid = np.unique(np.concatenate((np.geomspace(1e-8, 1e-2, 80), np.linspace(1e-2, 1.0, 160))))
    coefficients = np.stack(
        [_indicator_coefficients_unit(0.0, float(value), order) for value in grid]
    )
    estimated = spectral_summary_from_moments(
        moments,
        norms,
        maximum,
        order=order,
        cdf_coefficients=coefficients,
        cdf_grid=grid,
    )
    nonnull = exact[exact > 1e-12]
    expected = (
        np.mean(nonnull < 0.015),
        np.mean((nonnull >= 0.015) & (nonnull < 0.06)),
        np.mean(nonnull >= 0.06),
    )
    actual = (
        estimated["e_dominant_fraction"],
        estimated["transition_fraction"],
        estimated["b_dominant_fraction"],
    )
    np.testing.assert_allclose(actual, expected, atol=0.035)
    assert np.isclose(sum(actual), 1.0)


def test_standardized_coefficient_recovers_positive_effect() -> None:
    rows = [
        {"gain": 2.0 * value + 0.1 * (value % 3), "resolution": value, "control": value % 3}
        for value in np.linspace(1.0, 10.0, 30)
    ]
    coefficient = standardized_coefficient(
        rows, "gain", "resolution", ("control",)
    )
    assert coefficient > 0.9

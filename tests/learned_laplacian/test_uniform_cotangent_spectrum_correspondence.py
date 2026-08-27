from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import block_diag, diags


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_sofa50_uniform_cotangent_spectrum_correspondence import (  # noqa: E402
    component_null_basis,
    cross_basis_overlap,
    mode_correspondence,
    sampled_eigenmodes,
)


def test_sampled_modes_exclude_component_nullspace_and_cover_bands() -> None:
    # Two disconnected path-graph Laplacians give two exact component-constant
    # null modes and a known positive non-null spectrum in each block.
    blocks = []
    for size in (10, 10):
        block = diags(
            (-np.ones(size - 1), np.r_[1.0, np.full(size - 2, 2.0), 1.0], -np.ones(size - 1)),
            (-1, 0, 1),
            shape=(size, size),
            format="csr",
        )
        blocks.append(block)
    matrix = block_diag(blocks, format="csr")
    null = component_null_basis(np.repeat(np.arange(2), 10))
    sampled, vectors, bands, audit = sampled_eigenmodes(
        matrix,
        null,
        modes_per_band=2,
        tolerance=1e-10,
        maximum_iterations=10000,
        log_prefix="synthetic",
    )
    assert vectors.shape == (20, 6)
    assert list(bands) == ["low", "low", "mid", "mid", "high", "high"]
    assert np.all(sampled[:2] > 0)
    assert audit["maximum_relative_eigen_residual"] < 1e-8
    assert audit["maximum_nullspace_overlap"] < 1e-10


def test_monotone_mapping_and_band_correspondence_are_exact() -> None:
    source = np.asarray([0.1, 0.2, 0.45, 0.55, 0.8, 1.0])
    target = 3.0 * source
    bands = np.asarray(["low", "low", "mid", "mid", "high", "high"])
    result, table = mode_correspondence(source, target, bands, target_maximum=3.0)
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["band_diagonal_fraction"], 1.0)
    assert sum(row["count"] for row in table) == len(source)


def test_cross_basis_overlap_uses_mass_inner_product() -> None:
    mass = np.asarray([1.0, 2.0, 3.0])
    basis = np.eye(3)
    overlap = cross_basis_overlap(basis, basis, mass)
    np.testing.assert_allclose(overlap, np.eye(3), atol=1e-12)


def test_component_null_basis_is_orthonormal() -> None:
    labels = np.asarray([0, 0, 1, 1, 1])
    basis = component_null_basis(labels)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_sofa50_curvature_conditioned_benefit import (  # noqa: E402
    BIN_SPECS,
    area_weighted_vertex_normals,
    curvature_rank_bins,
    field_curvature_statistics,
    local_errors,
)


def test_rank_bins_are_complete_and_ordered() -> None:
    curvature = np.arange(100, dtype=np.float64)
    labels = curvature_rank_bins(curvature, np.ones(100, dtype=bool))
    assert len(BIN_SPECS) == 5
    assert [int(np.count_nonzero(labels == index)) for index in range(5)] == [25, 25, 25, 15, 10]
    assert np.all(np.diff(labels) >= 0)


def test_local_normal_error_separates_tangent_motion() -> None:
    clean = np.zeros((2, 3), dtype=np.float64)
    normals = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
    predicted = np.asarray([[3, 4, 0], [0, 0, 2]], dtype=np.float64)
    errors = local_errors(predicted, clean, normals)
    np.testing.assert_allclose(errors["vertex"], [5, 2])
    np.testing.assert_allclose(errors["normal"], [0, 2])
    np.testing.assert_allclose(errors["tangential"], [5, 0])


def test_area_weighted_normals_and_field_statistics() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    normals, valid = area_weighted_vertex_normals(vertices, faces)
    assert valid.all()
    np.testing.assert_allclose(normals, np.asarray([[0, 0, 1]] * 3))

    curvature = np.stack((np.arange(1, 11), np.zeros(10), np.zeros(10)), axis=1).astype(float)
    field = 2.0 * curvature
    stats = field_curvature_statistics(field, curvature, np.ones(10, dtype=bool))
    assert np.isclose(stats["magnitude_pearson"], 1.0)
    assert np.isclose(stats["magnitude_spearman"], 1.0)
    assert np.isclose(stats["directional_cosine_mean"], 1.0)
    assert np.isclose(stats["top10_recall"], 1.0)

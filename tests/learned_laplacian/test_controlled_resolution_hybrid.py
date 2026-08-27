from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_sofa50_controlled_resolution_hybrid import (  # noqa: E402
    _surface_area,
    nested_same_surface_levels,
)


def _square() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_nested_levels_preserve_clean_and_initial_piecewise_surfaces() -> None:
    clean, faces = _square()
    initial = clean.copy()
    initial[:, 2] = 0.1 * clean[:, 0] + 0.2 * clean[:, 1]
    levels = nested_same_surface_levels(clean, initial, faces, (4, 7, 12, 20))
    assert [len(level["clean_vertices"]) for level in levels] == [4, 7, 12, 20]
    clean_areas = [_surface_area(level["clean_vertices"], level["faces"]) for level in levels]
    initial_areas = [_surface_area(level["initial_vertices"], level["faces"]) for level in levels]
    np.testing.assert_allclose(clean_areas, clean_areas[0], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(initial_areas, initial_areas[0], rtol=0.0, atol=1e-14)
    for level in levels:
        np.testing.assert_allclose(level["clean_vertices"][:, 2], 0.0, atol=0.0)
        expected = 0.1 * level["initial_vertices"][:, 0] + 0.2 * level["initial_vertices"][:, 1]
        np.testing.assert_allclose(level["initial_vertices"][:, 2], expected, atol=1e-14)


def test_nested_levels_require_strictly_increasing_counts() -> None:
    clean, faces = _square()
    try:
        nested_same_surface_levels(clean, clean, faces, (4, 4))
    except ValueError as error:
        assert "increase strictly" in str(error)
    else:
        raise AssertionError("Expected invalid counts to fail")

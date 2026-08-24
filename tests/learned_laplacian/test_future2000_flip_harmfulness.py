from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/analyze_future2000_flip_harmfulness.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_orientation_categories_are_exhaustive() -> None:
    module = _module()
    initial = np.asarray([-0.8, 0.9, -0.7, 0.6, 0.0])
    refined = np.asarray([0.7, -0.6, -0.2, 0.8, -0.9])

    result = module._orientation_categories(initial, refined, 1e-6)

    assert result == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "wrong_to_wrong": 1,
        "correct_to_correct_or_ambiguous": 2,
    }
    assert sum(result.values()) == len(initial)


def test_change_counts_use_signed_epsilon() -> None:
    module = _module()
    initial = np.asarray([0.0, 0.0, 0.0])
    refined = np.asarray([2e-6, -2e-6, 0.5e-6])

    assert module._change_counts(initial, refined, 1e-6) == {
        "improved": 1,
        "worsened": 1,
        "unchanged_or_ambiguous": 1,
    }


def test_face_geometry_preserves_raw_orientation() -> None:
    module = _module()
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])

    cross, norm, normal, centroid = module._face_geometry(vertices, faces)

    np.testing.assert_allclose(cross, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(norm, [1.0])
    np.testing.assert_allclose(normal, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(centroid, [[1.0 / 3.0, 1.0 / 3.0, 0.0]])

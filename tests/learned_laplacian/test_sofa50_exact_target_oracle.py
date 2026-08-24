from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from mlr.data import Mesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sofa50_exact_target_oracle",
    ROOT / "scripts/diagnose_sofa50_exact_target_oracle.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_distribution = MODULE._distribution
_flip_attribution = MODULE._flip_attribution
_prediction_metrics = MODULE._prediction_metrics
_spectral_metrics = MODULE._spectral_metrics


def tetra(vertices: np.ndarray) -> Mesh:
    return Mesh(
        vertices,
        np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    ).ensure_normals()


def test_prediction_metrics_preserve_signed_vector_bias_and_visibility_groups() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    prediction = target + torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = _prediction_metrics(
        prediction,
        target,
        torch.ones(3, dtype=torch.bool),
        torch.tensor([True, False, True]),
    )
    assert np.isclose(result["raw_epe"], 0.1)
    assert np.isclose(result["mean_vector_bias_x"], 0.1)
    assert np.isclose(result["mean_vector_bias_y"], 0.0)
    assert np.isclose(result["visible_epe"], 0.1)
    assert np.isclose(result["invisible_epe"], 0.1)


def test_spectral_bands_form_an_energy_partition() -> None:
    # A sufficiently large fan-like graph avoids the tiny-mesh guard.
    vertices = np.asarray(
        [[np.cos(i * 2 * np.pi / 40), np.sin(i * 2 * np.pi / 40), 0.0] for i in range(40)]
        + [[0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[40, i, (i + 1) % 40] for i in range(40)], dtype=np.int64)
    error = np.column_stack((vertices[:, 0], vertices[:, 1], np.ones(len(vertices))))
    result = _spectral_metrics(error, faces)
    fraction_sum = (
        result["spectral_low_error_fraction"]
        + result["spectral_mid_error_fraction"]
        + result["spectral_high_error_fraction"]
    )
    assert np.isclose(fraction_sum, 1.0, atol=1e-6)
    assert result["spectral_low_error_energy"] >= 0
    assert result["spectral_high_error_energy"] >= 0


def test_flip_attribution_separates_oracle_and_prediction_only_flips() -> None:
    clean = tetra(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )
    initial = tetra(clean.vertices.copy())
    oracle = tetra(clean.vertices.copy())
    predicted_vertices = clean.vertices.copy()
    predicted_vertices[1] = [-1.0, 0.0, 0.0]
    predicted = tetra(predicted_vertices)
    row, arrays = _flip_attribution(
        "test", "sample", initial, clean, oracle, predicted, np.ones_like(clean.vertices)
    )
    assert row["oracle_introduced_flips"] == 0
    assert row["predicted_introduced_flips"] > 0
    assert row["prediction_only_flips"] == row["predicted_introduced_flips"]
    assert len(arrays["predicted_flip_area"]) == row["predicted_introduced_flips"]


def test_distribution_reports_negative_count_and_quantiles() -> None:
    result = _distribution([-1.0, 0.0, 1.0, float("nan")])
    assert result["count"] == 3
    assert result["negative_count"] == 1
    assert result["median"] == 0.0

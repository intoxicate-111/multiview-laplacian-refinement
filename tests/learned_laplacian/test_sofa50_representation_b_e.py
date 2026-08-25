from __future__ import annotations

import numpy as np

from evaluate_frozen_sofa50_b_e_ood import (
    _inference_loader_config,
    _set_execution_view_chunk_size,
)
from diagnose_sofa50_representation_b_vs_e import (
    SPECTRAL_BANDS,
    _indicator_coefficients,
    _variant,
    spectral_band_components,
)


def test_spectral_bands_partition_full_signal_energy() -> None:
    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    values = np.arange(24, dtype=np.float64).reshape(4, 6) / 17.0
    components, energy = spectral_band_components(values, faces, order=64)
    assert set(components) == set(SPECTRAL_BANDS)
    assert np.isclose(
        sum(energy[band] for band in SPECTRAL_BANDS),
        energy["total"],
        rtol=2e-5,
    )
    assert all(energy[band] >= 0 for band in SPECTRAL_BANDS)


def test_three_indicator_expansions_sum_to_identity() -> None:
    coefficients = sum(
        (
            _indicator_coefficients(low, high, 64)
            for low, high in SPECTRAL_BANDS.values()
        ),
        np.zeros(64),
    )
    assert np.isclose(coefficients[0], 1.0)
    assert np.allclose(coefficients[1:], 0.0, atol=1e-12)


def test_variant_parser_preserves_exact_recipe() -> None:
    assert _variant("uuid__A1") == "A1"
    assert _variant("uuid-with-hyphens__D2") == "D2"


def test_inference_loader_disables_only_loss_side_recovery() -> None:
    source = {
        "training": {
            "loss": "huber",
            "recovery_aware_geometry_loss": {"enabled": True, "beta": 1e-2},
        },
        "recovery": {"lambda": 1e-2},
    }
    result = _inference_loader_config(source)
    assert source["training"]["recovery_aware_geometry_loss"]["enabled"] is True
    assert result["training"]["recovery_aware_geometry_loss"]["enabled"] is False
    assert result["training"]["loss"] == "huber"
    assert result["recovery"]["lambda"] == 1e-2


def test_execution_view_chunking_changes_only_model_execution_attribute() -> None:
    class StubModel:
        image_view_chunk_size = None

    left = {"model": StubModel(), "checkpoint": "b.pt"}
    right = {"model": StubModel(), "checkpoint": "e.pt"}
    _set_execution_view_chunk_size((left, right), 2)
    assert left["model"].image_view_chunk_size == 2
    assert right["model"].image_view_chunk_size == 2
    assert left["checkpoint"] == "b.pt"
    assert right["checkpoint"] == "e.pt"

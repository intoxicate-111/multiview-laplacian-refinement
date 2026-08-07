from __future__ import annotations

import torch

from mlr.learned_laplacian.vertex_sampling import (
    FULL_VERTEX_EXPOSURE,
    HIGH_LAPLACIAN_MIXTURE,
    LAPLACIAN_MAGNITUDE_MIXTURE,
    sample_training_vertices,
    vertex_sampling_settings,
)


def test_full_vertex_exposure_returns_no_subsampling_indices():
    target = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    result = sample_training_vertices(
        target,
        torch.ones(20, dtype=torch.bool),
        vertex_sampling_settings({}),
        sample_id="mesh",
        base_seed=7,
        epoch=1,
    )
    assert result.indices is None
    assert result.diagnostics["mode"] == FULL_VERTEX_EXPOSURE


def test_high_laplacian_mixture_is_deterministic_and_has_requested_draw_counts():
    target = torch.zeros((100, 3), dtype=torch.float32)
    target[:, 0] = torch.arange(100, dtype=torch.float32)
    settings = vertex_sampling_settings(
        {
            "training": {
                "vertex_sampling": {
                    "mode": HIGH_LAPLACIAN_MIXTURE,
                    "sample_count_ratio": 1.0,
                    "uniform_fraction": 0.5,
                    "top_10_fraction": 0.25,
                    "top_1_to_10_fraction": 0.25,
                }
            }
        }
    )
    first = sample_training_vertices(
        target,
        torch.ones(100, dtype=torch.bool),
        settings,
        sample_id="mesh",
        base_seed=7,
        epoch=4,
    )
    second = sample_training_vertices(
        target,
        torch.ones(100, dtype=torch.bool),
        settings,
        sample_id="mesh",
        base_seed=7,
        epoch=4,
    )
    assert first.indices is not None
    torch.testing.assert_close(first.indices, second.indices)
    assert first.indices.shape == (100,)
    assert first.diagnostics["uniform_draw_count"] == 50
    assert first.diagnostics["top_10_draw_count"] == 25
    assert first.diagnostics["top_1_to_10_draw_count"] == 25


def test_query_support_sweep_accepts_requested_tenth_h_cap():
    from mlr.learned_laplacian.query_training import query_augmentation_settings

    settings = query_augmentation_settings(
        {
            "query_training": {
                "enabled": True,
                "normal_std_h": 0.03,
                "tangent_std_h": 0.03,
                "max_offset_h": 0.1,
            }
        }
    )
    assert settings.max_offset_h == 0.1


def test_geometry_mixture_supports_strong_and_smooth_control_pools():
    target = torch.zeros((100, 3), dtype=torch.float32)
    target[:, 0] = torch.arange(100, dtype=torch.float32)
    valid = torch.ones(100, dtype=torch.bool)
    strong = vertex_sampling_settings(
        {
            "training": {
                "vertex_sampling": {
                    "mode": LAPLACIAN_MAGNITUDE_MIXTURE,
                    "uniform_fraction": 0.25,
                    "top_10_fraction": 0.50,
                    "top_1_to_10_fraction": 0.0,
                    "top_1_fraction": 0.25,
                    "bottom_90_fraction": 0.0,
                }
            }
        }
    )
    smooth = vertex_sampling_settings(
        {
            "training": {
                "vertex_sampling": {
                    "mode": LAPLACIAN_MAGNITUDE_MIXTURE,
                    "uniform_fraction": 0.50,
                    "top_10_fraction": 0.0,
                    "top_1_to_10_fraction": 0.0,
                    "top_1_fraction": 0.0,
                    "bottom_90_fraction": 0.50,
                }
            }
        }
    )
    strong_result = sample_training_vertices(
        target, valid, strong, sample_id="mesh", base_seed=7, epoch=1
    )
    smooth_result = sample_training_vertices(
        target, valid, smooth, sample_id="mesh", base_seed=7, epoch=1
    )
    assert strong_result.diagnostics["uniform_draw_count"] == 25
    assert strong_result.diagnostics["top_10_draw_count"] == 50
    assert strong_result.diagnostics["top_1_draw_count"] == 25
    assert smooth_result.diagnostics["uniform_draw_count"] == 50
    assert smooth_result.diagnostics["bottom_90_draw_count"] == 50

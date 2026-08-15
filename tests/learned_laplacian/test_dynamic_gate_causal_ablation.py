from __future__ import annotations

import numpy as np

from mlr.learned_laplacian.dynamic_gate_causal_ablation import (
    alpha_grid,
    apply_gate_fp16,
    attribution_ratios,
    select_alpha_from_arrays,
    shuffled_gate,
)


def test_alpha_search_uses_requested_grid_and_selects_weighted_validation_optimum() -> None:
    base = np.zeros((2, 3), dtype=np.float64)
    residual = np.ones((2, 3), dtype=np.float64)
    target = np.full((2, 3), 0.17, dtype=np.float64)
    weight = np.array([1.0, 3.0])
    valid = np.array([True, True])

    grid = alpha_grid()
    selected, rows = select_alpha_from_arrays(
        base, residual, target, weight, valid, grid
    )

    assert len(grid) == 31
    assert grid[0] == 0.0
    assert grid[-1] == 0.30
    assert selected == 0.17
    assert sum(bool(row["selected"]) for row in rows) == 1
    assert all(row["split"] == "validation" for row in rows)


def test_shuffled_gate_is_deterministic_per_mesh_and_preserves_histogram() -> None:
    gate = np.linspace(0.0, 0.3, 100, dtype=np.float64)
    first = shuffled_gate(gate, "mesh_a__v00", 7)
    repeated = shuffled_gate(gate, "mesh_a__v00", 7)
    other_seed = shuffled_gate(gate, "mesh_a__v00", 17)
    other_mesh = shuffled_gate(gate, "mesh_b__v00", 7)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(np.sort(first), np.sort(gate))
    assert not np.array_equal(first, other_seed)
    assert not np.array_equal(first, other_mesh)
    assert first.mean() == gate.mean()
    assert first.std() == gate.std()
    assert first.min() == gate.min()
    assert first.max() == gate.max()


def test_apply_gate_replays_fp16_autocast_semantics() -> None:
    base = np.array([[1.0, 0.1, -0.2]], dtype=np.float32)
    residual = np.array([[0.003, -0.007, 0.011]], dtype=np.float32)
    gate = np.array([0.127], dtype=np.float32)
    expected = (
        base.astype(np.float16)
        + gate.astype(np.float16)[:, None] * residual.astype(np.float16)
    ).astype(np.float16).astype(np.float32)

    np.testing.assert_array_equal(apply_gate_fp16(base, residual, gate), expected)


def test_attribution_ratios_sum_to_one_only_for_positive_total_improvement() -> None:
    result = attribution_ratios(10.0, 7.0, 5.0)
    assert result["r_expert"] == 0.6
    assert result["r_gate"] == 0.4
    assert result["r_expert"] + result["r_gate"] == 1.0

    invalid = attribution_ratios(5.0, 6.0, 7.0)
    assert invalid["r_expert"] is None
    assert invalid["r_gate"] is None

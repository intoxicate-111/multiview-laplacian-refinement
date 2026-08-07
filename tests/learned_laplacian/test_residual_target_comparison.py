from __future__ import annotations

import numpy as np

from mlr.learned_laplacian.residual_target_comparison import (
    DIRECT,
    H2,
    RAW,
    build_comparison_targets,
    recover_prediction,
)


def _pair():
    current = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return current, target, faces


def test_targets_use_current_graph_and_h2_roundtrip() -> None:
    current, target, faces = _pair()
    built = build_comparison_targets(current, target, faces)
    np.testing.assert_allclose(built[DIRECT], target - current, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(built["raw_roundtrip"], built[RAW], atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(
        built[H2] * (built["local_edge_length"][:, None] ** 2 + 1e-12),
        built[RAW],
        atol=1e-15,
        rtol=1e-14,
    )


def test_scale_zero_is_identity_for_all_three_formulations() -> None:
    current, target, faces = _pair()
    built = build_comparison_targets(current, target, faces)
    solver = {
        "operator_type": "uniform",
        "lambda_lap": 1.0,
        "lambda_anchor": 0.01,
        "lambda_edge": 0.0,
        "num_iters": 5,
        "learning_rate": 0.01,
        "dense_vertex_limit": 5000,
    }
    for method in (DIRECT, RAW, H2):
        recovered, _ = recover_prediction(
            method,
            current,
            faces,
            built[method],
            local_edge_length=built["local_edge_length"],
            scale=0.0,
            solver_config=solver,
        )
        np.testing.assert_allclose(recovered.vertices, current, atol=0.0, rtol=0.0)

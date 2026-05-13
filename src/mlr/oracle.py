from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Array, Mesh
from .laplacian import compute_laplacian_coordinates
from .refinement import (
    RefinementConfig,
    RefinementResult,
    refine_mesh_position_only,
    refine_mesh_with_laplacian,
    zero_laplacian_smooth,
)


@dataclass
class OracleBaselineConfig:
    operator_type: str = "uniform"
    lambda_lap: float = 1.0
    lambda_anchor: float = 0.05
    lambda_position: float = 1.0
    noisy_laplacian_sigma: float = 0.01
    num_iters: int = 300
    learning_rate: float = 5e-3
    seed: int = 7


def run_oracle_baselines(
    init_mesh: Mesh,
    gt_vertices: Array,
    config: OracleBaselineConfig | None = None,
) -> dict[str, RefinementResult]:
    config = config or OracleBaselineConfig()
    gt_vertices = np.asarray(gt_vertices, dtype=np.float64)
    if gt_vertices.shape != init_mesh.vertices.shape:
        raise ValueError("Oracle baseline requires GT topology/correspondence with shape (N, 3).")

    delta_gt = compute_laplacian_coordinates(gt_vertices, init_mesh.faces, config.operator_type)
    opt = RefinementConfig(
        operator_type=config.operator_type,
        lambda_lap=config.lambda_lap,
        lambda_anchor=config.lambda_anchor,
        num_iters=config.num_iters,
        learning_rate=config.learning_rate,
    )
    pos_opt = RefinementConfig(
        operator_type=config.operator_type,
        lambda_lap=0.0,
        lambda_anchor=config.lambda_position,
        num_iters=config.num_iters,
        learning_rate=config.learning_rate,
    )

    rng = np.random.default_rng(config.seed)
    noisy_delta = delta_gt + rng.normal(scale=config.noisy_laplacian_sigma, size=delta_gt.shape)

    return {
        "position_only": refine_mesh_position_only(init_mesh, gt_vertices, config=pos_opt),
        "laplacian_only": refine_mesh_with_laplacian(
            init_mesh,
            delta_gt,
            confidence=np.ones(init_mesh.num_vertices),
            anchors=init_mesh.vertices,
            config=RefinementConfig(
                operator_type=config.operator_type,
                lambda_lap=config.lambda_lap,
                lambda_anchor=0.0,
                num_iters=config.num_iters,
                learning_rate=config.learning_rate,
            ),
        ),
        "position_plus_laplacian": refine_mesh_with_laplacian(
            init_mesh,
            delta_gt,
            confidence=np.ones(init_mesh.num_vertices),
            anchors=gt_vertices,
            config=opt,
        ),
        "zero_laplacian_smoothing": zero_laplacian_smooth(init_mesh, config=opt),
        "noisy_gt_laplacian": refine_mesh_with_laplacian(
            init_mesh,
            noisy_delta,
            confidence=np.ones(init_mesh.num_vertices),
            anchors=init_mesh.vertices,
            config=opt,
        ),
    }

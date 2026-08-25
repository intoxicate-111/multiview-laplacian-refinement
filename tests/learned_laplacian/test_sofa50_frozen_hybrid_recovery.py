from __future__ import annotations

import torch

from mlr.learned_laplacian.differentiable_sparse_recovery import (
    recovery_forward_audit,
    uniform_laplacian_apply,
)


def _cycle_graph(vertices: int) -> tuple[torch.Tensor, torch.Tensor]:
    directed = []
    for index in range(vertices):
        directed.extend(((index, (index + 1) % vertices), ((index + 1) % vertices, index)))
    edges = torch.tensor(directed, dtype=torch.long).T
    degree = torch.full((vertices, 1), 2.0, dtype=torch.float64)
    return edges, degree


def test_direct_anchor_hybrid_pcg_satisfies_normal_equation() -> None:
    generator = torch.Generator().manual_seed(7)
    direct = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    desired = torch.randn((12, 3), generator=generator, dtype=torch.float64)
    edges, degree = _cycle_graph(12)
    target = uniform_laplacian_apply(desired, edges, degree)
    regularization = 3e-2
    recovered, audit = recovery_forward_audit(
        target,
        direct,
        edges,
        degree,
        regularization=regularization,
        maximum_iterations=2048,
        tolerance=1e-10,
    )
    assert audit.converged
    lap_recovered = uniform_laplacian_apply(recovered, edges, degree)
    lap_residual = uniform_laplacian_apply(lap_recovered - target, edges.flip(0), degree)
    # For this regular degree graph L is symmetric, so L^T=L.
    normal_residual = lap_residual + regularization * (recovered - direct)
    assert torch.linalg.vector_norm(normal_residual).item() < 1e-8


def test_large_lambda_moves_hybrid_toward_direct_anchor() -> None:
    generator = torch.Generator().manual_seed(11)
    direct = torch.randn((16, 3), generator=generator, dtype=torch.float64)
    target = torch.randn((16, 3), generator=generator, dtype=torch.float64)
    edges, degree = _cycle_graph(16)
    distances = []
    for regularization in (1e-3, 1e-1, 10.0):
        recovered, audit = recovery_forward_audit(
            target,
            direct,
            edges,
            degree,
            regularization=regularization,
            maximum_iterations=2048,
            tolerance=1e-10,
        )
        assert audit.converged
        distances.append(float(torch.sqrt(torch.mean(torch.sum((recovered - direct) ** 2, dim=1)))))
    assert distances[0] > distances[1] > distances[2]

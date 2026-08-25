from __future__ import annotations

import torch

from mlr.learned_laplacian.cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
    cotangent_stiffness_apply,
    differentiable_cotangent_sparse_recovery,
)


def _dense_matrix(
    vertices: torch.Tensor, faces: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edges, weights, diagonal, _ = build_symmetric_cotangent_stiffness(vertices, faces)
    dense = torch.diag(diagonal)
    if edges.shape[1]:
        dense[edges[0], edges[1]] = -weights
        dense[edges[1], edges[0]] = -weights
    return dense, edges, weights, diagonal


def test_boundary_cotangent_stiffness_has_one_sided_weights() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    dense, _, _, _ = _dense_matrix(vertices, faces)
    expected = torch.tensor(
        [[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(dense, expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dense, dense.T, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dense.sum(dim=1), torch.zeros(3, dtype=torch.float64))


def test_obtuse_triangle_retains_negative_cotangent_weight() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.2, 0.1, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    _, weights, _, audit = build_symmetric_cotangent_stiffness(vertices, faces)
    assert audit.negative_edge_weights == 1
    assert bool(torch.any(weights < 0))


def test_near_degenerate_triangle_uses_documented_zero_contribution() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-15, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    edges, weights, diagonal, audit = build_symmetric_cotangent_stiffness(
        vertices, faces, relative_area_epsilon=1e-12
    )
    assert audit.protected_triangles == 1
    assert edges.shape == (2, 0)
    assert weights.numel() == 0
    assert torch.count_nonzero(diagonal) == 0


def test_sparse_apply_matches_dense_and_is_self_adjoint() -> None:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 1.0],
        ],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    dense, edges, weights, diagonal = _dense_matrix(vertices, faces)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((4, 3), dtype=torch.float64, generator=generator)
    y = torch.randn((4, 3), dtype=torch.float64, generator=generator)
    applied = cotangent_stiffness_apply(x, edges, weights, diagonal)
    torch.testing.assert_close(applied, dense @ x, atol=1e-12, rtol=1e-12)
    left = (applied * y).sum()
    right = (x * cotangent_stiffness_apply(y, edges, weights, diagonal)).sum()
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_cotangent_recovery_matches_dense_and_both_gradients() -> None:
    torch.manual_seed(11)
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    dense, edges, weights, diagonal = _dense_matrix(vertices, faces)
    lap = torch.randn((3, 3), dtype=torch.float64, requires_grad=True)
    direct = torch.randn((3, 3), dtype=torch.float64, requires_grad=True)
    regularization = 3e-2
    recovered = differentiable_cotangent_sparse_recovery(
        lap,
        direct,
        edges,
        weights,
        diagonal,
        regularization=regularization,
        maximum_iterations=256,
        tolerance=1e-11,
    )
    matrix = dense.T @ dense + regularization * torch.eye(3, dtype=torch.float64)
    reference = torch.linalg.solve(
        matrix, dense.T @ lap.detach() + regularization * direct.detach()
    )
    torch.testing.assert_close(recovered.detach(), reference, atol=1e-9, rtol=1e-9)
    target = torch.randn((3, 3), dtype=torch.float64)
    loss = (recovered - target).square().sum(dim=-1).mean()
    loss.backward()
    assert lap.grad is not None and direct.grad is not None
    assert torch.isfinite(lap.grad).all() and torch.isfinite(direct.grad).all()
    assert float(torch.linalg.vector_norm(lap.grad)) > 0
    assert float(torch.linalg.vector_norm(direct.grad)) > 0

    epsilon = 1e-6
    for source, gradient, index in (
        (lap, lap.grad, (1, 2)),
        (direct, direct.grad, (2, 1)),
    ):
        plus, minus = source.detach().clone(), source.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon

        def objective(candidate: torch.Tensor) -> torch.Tensor:
            lap_value = candidate if source is lap else lap.detach()
            direct_value = candidate if source is direct else direct.detach()
            value = differentiable_cotangent_sparse_recovery(
                lap_value,
                direct_value,
                edges,
                weights,
                diagonal,
                regularization=regularization,
                maximum_iterations=256,
                tolerance=1e-11,
            )
            return (value - target).square().sum(dim=-1).mean()

        finite = float((objective(plus) - objective(minus)) / (2 * epsilon))
        analytic = float(gradient[index])
        relative = abs(analytic - finite) / max(abs(finite), 1e-12)
        assert relative < 1e-6

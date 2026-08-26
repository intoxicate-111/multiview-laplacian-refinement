from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

from mlr.learned_laplacian.differentiable_sparse_recovery import (
    _pcg_solve,
    differentiable_regularized_sparse_recovery,
    differentiable_regularized_sparse_recovery_with_audit,
    recovery_forward_audit,
    uniform_laplacian_apply,
    uniform_laplacian_transpose_apply,
)


def _cycle_graph() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor([1, 3, 0, 2, 1, 3, 2, 0], dtype=torch.long)
    destination = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    return torch.stack((source, destination)), torch.full((4, 1), 2.0)


def test_float32_pcg_rechecks_true_residual_before_stopping() -> None:
    vertices = 1_000
    source = torch.arange(vertices - 1, dtype=torch.long)
    destination = source + 1
    edge_index = torch.stack(
        (
            torch.cat((source, destination)),
            torch.cat((destination, source)),
        )
    )
    degree = torch.bincount(edge_index[1], minlength=vertices).float().unsqueeze(1)
    generator = torch.Generator().manual_seed(4)
    right_hand_side = torch.randn((vertices, 3), generator=generator)
    right_hand_side[:, 1] *= 1e-3
    right_hand_side[:, 2] *= 1e-6

    _, audit = _pcg_solve(
        right_hand_side,
        edge_index,
        degree,
        1e-2,
        maximum_iterations=256,
        tolerance=1e-4,
    )

    assert audit.converged
    assert audit.relative_residual <= 1e-4


def test_uniform_laplacian_and_transpose_are_adjoint() -> None:
    edge_index, degree = _cycle_graph()
    generator = torch.Generator().manual_seed(7)
    x = torch.randn((4, 3), generator=generator, dtype=torch.double)
    y = torch.randn((4, 3), generator=generator, dtype=torch.double)
    left = (uniform_laplacian_apply(x, edge_index, degree) * y).sum()
    right = (x * uniform_laplacian_transpose_apply(y, edge_index, degree)).sum()
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_forward_matches_lsmr_and_gradient_is_finite_nonzero() -> None:
    edge_index, degree = _cycle_graph()
    initial = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.double,
    )
    prediction = torch.tensor(
        [[-0.8, -0.9, 0.1], [0.9, -0.7, -0.1], [0.8, 0.9, 0.2], [-0.9, 0.7, -0.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    regularization = 1e-2
    recovered, audit = recovery_forward_audit(
        prediction,
        initial,
        edge_index,
        degree,
        regularization=regularization,
        maximum_iterations=256,
        tolerance=1e-10,
    )
    laplacian = np.eye(4)
    for source, destination in edge_index.t().numpy():
        laplacian[destination, source] -= 0.5
    system = vstack(
        (csr_matrix(laplacian), np.sqrt(regularization) * eye(4)), format="csr"
    )
    rhs = np.vstack(
        (prediction.detach().numpy(), np.sqrt(regularization) * initial.numpy())
    )
    expected = np.column_stack(
        [lsmr(system, rhs[:, axis], atol=1e-12, btol=1e-12)[0] for axis in range(3)]
    )
    assert audit.converged
    np.testing.assert_allclose(
        recovered.detach().numpy(), expected, atol=1e-8, rtol=1e-8
    )

    autograd_recovered = differentiable_regularized_sparse_recovery(
        prediction,
        initial,
        edge_index,
        degree,
        regularization=regularization,
        maximum_iterations=256,
        tolerance=1e-10,
    )
    loss = (autograd_recovered - 0.5 * initial).square().sum(dim=-1).mean()
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0


def test_differentiable_audit_uses_the_same_forward_and_keeps_gradients() -> None:
    edge_index, degree = _cycle_graph()
    initial = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.double,
    )
    prediction = torch.randn((4, 3), dtype=torch.double, requires_grad=True)
    recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
        prediction,
        initial,
        edge_index,
        degree,
        regularization=1e-3,
        maximum_iterations=256,
        tolerance=1e-10,
    )
    expected = differentiable_regularized_sparse_recovery(
        prediction.detach(),
        initial,
        edge_index,
        degree,
        regularization=1e-3,
        maximum_iterations=256,
        tolerance=1e-10,
    )
    torch.testing.assert_close(recovered, expected, atol=0, rtol=0)
    assert audit.converged
    assert audit.iterations > 0
    assert audit.relative_residual <= 1.05e-10
    recovered.square().mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0


def test_regularization_scalar_receives_correct_finite_nonzero_gradient() -> None:
    edge_index, degree = _cycle_graph()
    initial = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.double,
    )
    prediction = torch.tensor(
        [[-0.8, -0.9, 0.1], [0.9, -0.7, -0.1], [0.8, 0.9, 0.2], [-0.9, 0.7, -0.2]],
        dtype=torch.double,
    )
    target = 0.65 * initial
    regularization = torch.tensor(1e-2, dtype=torch.double, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_regularized_sparse_recovery(
            prediction,
            initial,
            edge_index,
            degree,
            regularization=value,
            maximum_iterations=256,
            tolerance=1e-11,
        )
        return (recovered - target).square().mean()

    loss = objective(regularization)
    loss.backward()
    assert regularization.grad is not None
    assert torch.isfinite(regularization.grad)
    assert abs(float(regularization.grad)) > 0
    epsilon = 1e-6
    finite_difference = (
        objective(torch.tensor(1e-2 + epsilon, dtype=torch.double))
        - objective(torch.tensor(1e-2 - epsilon, dtype=torch.double))
    ) / (2 * epsilon)
    torch.testing.assert_close(
        regularization.grad, finite_difference, atol=2e-7, rtol=2e-6
    )


def test_hybrid_solve_has_correct_gradients_to_lap_and_direct_branches() -> None:
    edge_index, degree = _cycle_graph()
    generator = torch.Generator().manual_seed(17)
    lap = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    direct = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    target = torch.randn((4, 3), generator=generator, dtype=torch.double)

    def objective(lap_value: torch.Tensor, direct_value: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_regularized_sparse_recovery(
            lap_value,
            direct_value,
            edge_index,
            degree,
            regularization=3e-2,
            maximum_iterations=256,
            tolerance=1e-11,
        )
        return (recovered - target).square().sum(dim=-1).mean()

    objective(lap, direct).backward()
    assert lap.grad is not None and direct.grad is not None
    assert torch.isfinite(lap.grad).all() and torch.isfinite(direct.grad).all()
    assert float(torch.linalg.vector_norm(lap.grad)) > 0
    assert float(torch.linalg.vector_norm(direct.grad)) > 0

    epsilon = 1e-6
    for value, gradient, index in (
        (lap.detach(), lap.grad, (1, 2)),
        (direct.detach(), direct.grad, (2, 0)),
    ):
        plus = value.clone()
        minus = value.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        if value.data_ptr() == lap.detach().data_ptr():  # pragma: no cover - tensors differ
            finite = (objective(plus, direct.detach()) - objective(minus, direct.detach())) / (2 * epsilon)
        else:
            finite = (objective(lap.detach(), plus) - objective(lap.detach(), minus)) / (2 * epsilon)
        torch.testing.assert_close(gradient[index], finite, atol=3e-7, rtol=3e-6)

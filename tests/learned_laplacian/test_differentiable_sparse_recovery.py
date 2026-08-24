from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

from mlr.learned_laplacian.differentiable_sparse_recovery import (
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

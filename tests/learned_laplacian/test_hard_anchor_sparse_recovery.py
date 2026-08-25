from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from mlr.learned_laplacian.hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
    differentiable_hard_anchor_sparse_recovery,
    differentiable_hard_anchor_sparse_recovery_with_audit,
)


def _two_component_graph() -> tuple[torch.Tensor, torch.Tensor]:
    # Components {0,1,2} and {3,4}; both directions are explicit.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 0, 2, 3, 4],
         [1, 2, 0, 1, 0, 2, 0, 1, 1, 2, 2, 0, 4, 3]],
        dtype=torch.long,
    )
    degree = torch.bincount(edge_index[1], minlength=5).to(torch.float64)
    return edge_index, degree


def _dense_laplacian(edge_index: torch.Tensor, degree: torch.Tensor) -> np.ndarray:
    n = int(degree.numel())
    source = edge_index[0].numpy()
    destination = edge_index[1].numpy()
    matrix = np.eye(n, dtype=np.float64)
    np.add.at(
        matrix,
        (destination, source),
        -1.0 / degree.numpy()[destination],
    )
    return matrix


def test_component_anchors_are_lowest_global_indices() -> None:
    edge_index, _ = _two_component_graph()
    anchors = deterministic_component_anchor_indices(edge_index, 5)
    assert anchors.tolist() == [0, 3]


def test_hard_anchor_forward_matches_reduced_lsmr() -> None:
    edge_index, degree = _two_component_graph()
    generator = torch.Generator().manual_seed(9)
    initial = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    prediction = torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.1
    anchors = torch.tensor([0, 3], dtype=torch.long)
    recovered, audit = differentiable_hard_anchor_sparse_recovery_with_audit(
        prediction,
        initial,
        edge_index,
        degree,
        anchors,
        maximum_iterations=256,
        tolerance=1e-11,
    )
    laplacian = _dense_laplacian(edge_index, degree)
    free = np.array([1, 2, 4], dtype=np.int64)
    anchor_values = np.zeros((5, 3), dtype=np.float64)
    anchor_values[anchors.numpy()] = initial.numpy()[anchors.numpy()]
    rhs = prediction.numpy() - laplacian @ anchor_values
    expected = anchor_values.copy()
    for axis in range(3):
        expected[free, axis] = lsmr(
            coo_matrix(laplacian[:, free]).tocsr(),
            rhs[:, axis],
            atol=1e-14,
            btol=1e-14,
            maxiter=1000,
        )[0]
    assert audit.converged
    assert torch.equal(recovered[anchors], initial[anchors])
    np.testing.assert_allclose(recovered.detach().numpy(), expected, atol=1e-9, rtol=1e-9)


def test_hard_anchor_prediction_gradient_matches_finite_difference() -> None:
    edge_index, degree = _two_component_graph()
    generator = torch.Generator().manual_seed(17)
    initial = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    prediction = (
        torch.randn(5, 3, generator=generator, dtype=torch.float64) * 0.05
    ).requires_grad_(True)
    target = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    anchors = torch.tensor([0, 3], dtype=torch.long)

    def objective(values: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_hard_anchor_sparse_recovery(
            values,
            initial,
            edge_index,
            degree,
            anchors,
            maximum_iterations=256,
            tolerance=1e-11,
        )
        return ((recovered - target) ** 2).mean()

    loss = objective(prediction)
    analytic = torch.autograd.grad(loss, prediction)[0]
    epsilon = 1e-6
    for row, column in ((0, 0), (2, 1), (4, 2)):
        plus = prediction.detach().clone()
        minus = prediction.detach().clone()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        assert torch.isfinite(analytic[row, column])
        torch.testing.assert_close(
            analytic[row, column], numerical, atol=2e-7, rtol=2e-6
        )

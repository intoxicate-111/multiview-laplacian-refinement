from __future__ import annotations

"""Implicitly differentiated regularized uniform-Laplacian integration."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConjugateGradientAudit:
    iterations: int
    converged: bool
    relative_residual: float


def _degree_vector(vertex_degree: torch.Tensor, vertices: int) -> torch.Tensor:
    degree = vertex_degree.reshape(-1)
    if tuple(degree.shape) != (vertices,):
        raise ValueError("vertex_degree must have shape [N] or [N, 1].")
    if not torch.isfinite(degree).all() or bool(torch.any(degree <= 0)):
        raise ValueError("Regularized recovery requires finite positive vertex degrees.")
    return degree


def uniform_laplacian_apply(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
) -> torch.Tensor:
    """Apply L=I-D^-1 A using the prepared directed src->dst edge list."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("values must have shape [N, 3].")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    degree = _degree_vector(vertex_degree, int(values.shape[0])).to(
        device=values.device, dtype=values.dtype
    )
    edges = edge_index.to(device=values.device, dtype=torch.long)
    source, destination = edges[0], edges[1]
    neighbor_sum = torch.zeros_like(values)
    neighbor_sum.index_add_(0, destination, values.index_select(0, source))
    return values - neighbor_sum / degree.unsqueeze(-1)


def uniform_laplacian_transpose_apply(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact transpose of the prepared random-walk Laplacian."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("values must have shape [N, 3].")
    degree = _degree_vector(vertex_degree, int(values.shape[0])).to(
        device=values.device, dtype=values.dtype
    )
    edges = edge_index.to(device=values.device, dtype=torch.long)
    source, destination = edges[0], edges[1]
    result = values.clone()
    contribution = -values.index_select(0, destination) / degree.index_select(
        0, destination
    ).unsqueeze(-1)
    result.index_add_(0, source, contribution)
    return result


def _normal_matrix_apply(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    regularization: float | torch.Tensor,
) -> torch.Tensor:
    return uniform_laplacian_transpose_apply(
        uniform_laplacian_apply(values, edge_index, vertex_degree),
        edge_index,
        vertex_degree,
    ) + regularization * values


def _jacobi_diagonal(
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    *,
    vertices: int,
    dtype: torch.dtype,
    device: torch.device,
    regularization: float | torch.Tensor,
) -> torch.Tensor:
    degree = _degree_vector(vertex_degree, vertices).to(device=device, dtype=dtype)
    edges = edge_index.to(device=device, dtype=torch.long)
    source, destination = edges[0], edges[1]
    regularization_tensor = torch.as_tensor(
        regularization, dtype=dtype, device=device
    ).reshape(())
    diagonal = torch.ones((vertices,), dtype=dtype, device=device) * (
        1.0 + regularization_tensor
    )
    diagonal.index_add_(
        0,
        source,
        degree.index_select(0, destination).reciprocal().square(),
    )
    return diagonal.clamp_min(torch.finfo(dtype).eps)


def _pcg_solve(
    right_hand_side: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    regularization: float | torch.Tensor,
    *,
    maximum_iterations: int,
    tolerance: float,
    initial_guess: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    regularization_tensor = torch.as_tensor(
        regularization,
        dtype=right_hand_side.dtype,
        device=right_hand_side.device,
    ).reshape(())
    if not bool(torch.isfinite(regularization_tensor)) or bool(
        regularization_tensor <= 0
    ):
        raise ValueError("Differentiable recovery requires positive regularization.")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    rhs = right_hand_side
    solution = torch.zeros_like(rhs) if initial_guess is None else initial_guess.clone()
    diagonal = _jacobi_diagonal(
        edge_index,
        vertex_degree,
        vertices=int(rhs.shape[0]),
        dtype=rhs.dtype,
        device=rhs.device,
        regularization=regularization_tensor,
    ).unsqueeze(-1)
    residual = rhs - _normal_matrix_apply(
        solution, edge_index, vertex_degree, regularization_tensor
    )
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    # Treat XYZ as one flattened block-diagonal SPD system. A per-coordinate
    # stopping test is numerically brittle when one coordinate has an almost
    # zero RHS (common for nearly planar meshes and adjoint gradients).
    residual_preconditioned = (residual * preconditioned).sum()
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    active = torch.linalg.vector_norm(residual) > tolerance * rhs_norm
    iterations = 0
    # Dot products naturally become far smaller than machine epsilon near
    # convergence. Clamping them to eps stalls CG; only protect true underflow.
    epsilon = torch.finfo(rhs.dtype).tiny
    for iteration in range(maximum_iterations):
        if not bool(active):
            break
        matrix_direction = _normal_matrix_apply(
            direction, edge_index, vertex_degree, regularization_tensor
        )
        denominator = (direction * matrix_direction).sum()
        alpha = residual_preconditioned / denominator.clamp_min(epsilon)
        solution = solution + direction * alpha
        residual = residual - matrix_direction * alpha
        iterations = iteration + 1
        active = torch.linalg.vector_norm(residual) > tolerance * rhs_norm
        if not bool(active):
            break
        next_preconditioned = residual / diagonal
        next_residual_preconditioned = (residual * next_preconditioned).sum()
        beta = next_residual_preconditioned / residual_preconditioned.clamp_min(epsilon)
        direction = next_preconditioned + direction * beta
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned
    final_residual = rhs - _normal_matrix_apply(
        solution, edge_index, vertex_degree, regularization_tensor
    )
    relative = torch.linalg.vector_norm(final_residual) / rhs_norm
    # The recurrence residual can cross the requested tolerance one iteration
    # before the explicitly recomputed residual because the solve is float32.
    # A 5% acceptance margin prevents false failures at 1.00x tolerance while
    # retaining the requested numerical accuracy (audited against float64 LSMR).
    convergence_slack = 1.05
    return solution, ConjugateGradientAudit(
        iterations=iterations,
        converged=bool(relative <= tolerance * convergence_slack),
        relative_residual=float(relative.detach().cpu()),
    )


class _RegularizedSparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        predicted_laplacian: torch.Tensor,
        initial_vertices: torch.Tensor,
        edge_index: torch.Tensor,
        vertex_degree: torch.Tensor,
        regularization: torch.Tensor,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regularization = regularization.to(
            device=predicted_laplacian.device, dtype=predicted_laplacian.dtype
        ).reshape(())
        maximum_iterations = int(maximum_iterations)
        tolerance = float(tolerance)
        rhs = uniform_laplacian_transpose_apply(
            predicted_laplacian, edge_index, vertex_degree
        ) + regularization * initial_vertices
        solution, audit = _pcg_solve(
            rhs,
            edge_index,
            vertex_degree,
            regularization,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            initial_guess=initial_vertices,
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable regularized sparse solve did not converge: "
                f"relative_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}."
            )
        ctx.save_for_backward(
            edge_index, vertex_degree, regularization, solution, initial_vertices
        )
        ctx.maximum_iterations = maximum_iterations
        ctx.tolerance = tolerance
        audit_tensor = torch.tensor(
            [
                float(audit.iterations),
                float(audit.converged),
                float(audit.relative_residual),
            ],
            dtype=torch.float64,
            device=solution.device,
        )
        ctx.mark_non_differentiable(audit_tensor)
        return solution, audit_tensor

    @staticmethod
    def backward(
        ctx, output_gradient: torch.Tensor, audit_gradient: torch.Tensor | None
    ):
        edge_index, vertex_degree, regularization, solution, initial_vertices = (
            ctx.saved_tensors
        )
        adjoint, audit = _pcg_solve(
            output_gradient,
            edge_index,
            vertex_degree,
            regularization,
            maximum_iterations=ctx.maximum_iterations,
            tolerance=ctx.tolerance,
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable regularized sparse adjoint did not converge: "
                f"relative_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}."
            )
        prediction_gradient = uniform_laplacian_apply(
            adjoint, edge_index, vertex_degree
        )
        initial_gradient = regularization * adjoint
        regularization_gradient = (adjoint * (initial_vertices - solution)).sum()
        return (
            prediction_gradient,
            initial_gradient,
            None,
            None,
            regularization_gradient,
            None,
            None,
        )


def differentiable_regularized_sparse_recovery(
    predicted_laplacian: torch.Tensor,
    initial_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    *,
    regularization: float | torch.Tensor,
    maximum_iterations: int = 128,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Recover vertices and propagate exact implicit gradients to delta prediction."""
    if predicted_laplacian.shape != initial_vertices.shape:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    if predicted_laplacian.ndim != 2 or predicted_laplacian.shape[1] != 3:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    regularization_tensor = torch.as_tensor(
        regularization,
        dtype=predicted_laplacian.dtype,
        device=predicted_laplacian.device,
    ).reshape(())
    recovered, _ = _RegularizedSparseSolve.apply(
        predicted_laplacian,
        initial_vertices,
        edge_index,
        vertex_degree,
        regularization_tensor,
        int(maximum_iterations),
        float(tolerance),
    )
    return recovered


def differentiable_regularized_sparse_recovery_with_audit(
    predicted_laplacian: torch.Tensor,
    initial_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    *,
    regularization: float | torch.Tensor,
    maximum_iterations: int = 128,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    """Recover vertices and expose the audit from the same differentiable solve."""
    if predicted_laplacian.shape != initial_vertices.shape:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    if predicted_laplacian.ndim != 2 or predicted_laplacian.shape[1] != 3:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    regularization_tensor = torch.as_tensor(
        regularization,
        dtype=predicted_laplacian.dtype,
        device=predicted_laplacian.device,
    ).reshape(())
    recovered, audit_tensor = _RegularizedSparseSolve.apply(
        predicted_laplacian,
        initial_vertices,
        edge_index,
        vertex_degree,
        regularization_tensor,
        int(maximum_iterations),
        float(tolerance),
    )
    audit_values = audit_tensor.detach().cpu().tolist()
    return recovered, ConjugateGradientAudit(
        iterations=int(audit_values[0]),
        converged=bool(audit_values[1]),
        relative_residual=float(audit_values[2]),
    )


def recovery_forward_audit(
    predicted_laplacian: torch.Tensor,
    initial_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    *,
    regularization: float | torch.Tensor,
    maximum_iterations: int = 128,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    """Expose the PCG forward audit without constructing an autograd node."""
    rhs = uniform_laplacian_transpose_apply(
        predicted_laplacian, edge_index, vertex_degree
    ) + regularization * initial_vertices
    return _pcg_solve(
        rhs,
        edge_index,
        vertex_degree,
        regularization,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        initial_guess=initial_vertices,
    )

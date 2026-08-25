from __future__ import annotations

"""Symmetric cotangent-stiffness construction and implicit sparse recovery."""

from dataclasses import dataclass

import numpy as np
import torch

from .differentiable_sparse_recovery import ConjugateGradientAudit


@dataclass(frozen=True)
class CotangentConstructionAudit:
    faces: int
    unique_edges: int
    protected_triangles: int
    negative_edge_weights: int
    nonfinite_edge_weights: int
    maximum_absolute_weight: float


def build_symmetric_cotangent_stiffness(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    relative_area_epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CotangentConstructionAudit]:
    """Build C with C_ij=-w_ij and C_ii=sum_j w_ij.

    Each valid incident triangle contributes half of its opposite-angle
    cotangent. Boundary edges therefore receive one contribution, ordinary
    interior edges receive two, and non-manifold edges retain every real
    incident-face contribution. Negative cotangents are deliberately retained.

    A triangle whose twice-area is at most ``relative_area_epsilon`` times its
    maximum squared edge length contributes zero to all three edges. This is a
    scale-invariant protection for undefined cotangents; topology is unchanged.
    """

    if relative_area_epsilon <= 0:
        raise ValueError("relative_area_epsilon must be positive.")
    xyz = np.asarray(vertices.detach().cpu(), dtype=np.float64)
    tri = np.asarray(faces.detach().cpu(), dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3].")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if tri.size and (tri.min() < 0 or tri.max() >= len(xyz)):
        raise ValueError("faces contain an out-of-range vertex index.")

    if len(tri) == 0:
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_weights = torch.empty((0,), dtype=torch.float64)
        diagonal = torch.zeros((len(xyz),), dtype=torch.float64)
        audit = CotangentConstructionAudit(0, 0, 0, 0, 0, 0.0)
        return empty_edges, empty_weights, diagonal, audit

    i, j, k = tri.T
    vi, vj, vk = xyz[i], xyz[j], xyz[k]
    edge_ij = vj - vi
    edge_ik = vk - vi
    edge_jk = vk - vj
    twice_area = np.linalg.norm(np.cross(edge_ij, edge_ik), axis=1)
    max_edge_squared = np.maximum.reduce(
        (
            np.einsum("ij,ij->i", edge_ij, edge_ij),
            np.einsum("ij,ij->i", edge_ik, edge_ik),
            np.einsum("ij,ij->i", edge_jk, edge_jk),
        )
    )
    protected = twice_area <= relative_area_epsilon * np.maximum(
        max_edge_squared, np.finfo(np.float64).tiny
    )
    denominator = np.where(protected, 1.0, twice_area)
    cot_i = np.einsum("ij,ij->i", edge_ij, edge_ik) / denominator
    cot_j = np.einsum("ij,ij->i", -edge_ij, edge_jk) / denominator
    cot_k = np.einsum("ij,ij->i", -edge_ik, -edge_jk) / denominator
    cot_i[protected] = 0.0
    cot_j[protected] = 0.0
    cot_k[protected] = 0.0

    edge_u = np.concatenate((j, i, i))
    edge_v = np.concatenate((k, k, j))
    contributions = 0.5 * np.concatenate((cot_i, cot_j, cot_k))
    lower = np.minimum(edge_u, edge_v)
    upper = np.maximum(edge_u, edge_v)
    order = np.lexsort((upper, lower))
    lower, upper, contributions = lower[order], upper[order], contributions[order]
    starts = np.r_[True, (lower[1:] != lower[:-1]) | (upper[1:] != upper[:-1])]
    start_indices = np.flatnonzero(starts)
    unique_lower = lower[start_indices]
    unique_upper = upper[start_indices]
    weights = np.add.reduceat(contributions, start_indices)

    nonfinite = ~np.isfinite(weights)
    nonfinite_count = int(nonfinite.sum())
    if nonfinite_count:
        raise FloatingPointError(
            f"Cotangent construction produced {nonfinite_count} non-finite edge weights."
        )
    keep = weights != 0.0
    unique_lower, unique_upper, weights = (
        unique_lower[keep],
        unique_upper[keep],
        weights[keep],
    )
    diagonal = np.zeros((len(xyz),), dtype=np.float64)
    np.add.at(diagonal, unique_lower, weights)
    np.add.at(diagonal, unique_upper, weights)
    edge_index = torch.from_numpy(
        np.stack((unique_lower, unique_upper), axis=0).astype(np.int64, copy=False)
    )
    edge_weight = torch.from_numpy(weights.astype(np.float64, copy=False))
    diagonal_tensor = torch.from_numpy(diagonal)
    audit = CotangentConstructionAudit(
        faces=int(len(tri)),
        unique_edges=int(len(weights)),
        protected_triangles=int(protected.sum()),
        negative_edge_weights=int((weights < 0).sum()),
        nonfinite_edge_weights=nonfinite_count,
        maximum_absolute_weight=float(np.max(np.abs(weights), initial=0.0)),
    )
    return edge_index, edge_weight, diagonal_tensor, audit


def cotangent_stiffness_apply(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    """Apply the symmetric cotangent stiffness matrix without densifying."""

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("values must have shape [N, 3].")
    vertices = int(values.shape[0])
    if tuple(edge_index.shape[:1]) != (2,) or edge_index.ndim != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if tuple(edge_weight.shape) != (edge_index.shape[1],):
        raise ValueError("edge_weight must have shape [E].")
    if tuple(diagonal.shape) != (vertices,):
        raise ValueError("diagonal must have shape [N].")
    edges = edge_index.to(device=values.device, dtype=torch.long)
    weights = edge_weight.to(device=values.device, dtype=values.dtype)
    diag = diagonal.to(device=values.device, dtype=values.dtype)
    u, v = edges[0], edges[1]
    result = diag.unsqueeze(-1) * values
    result.index_add_(0, u, -weights.unsqueeze(-1) * values.index_select(0, v))
    result.index_add_(0, v, -weights.unsqueeze(-1) * values.index_select(0, u))
    return result


def _normal_apply(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
    regularization: torch.Tensor,
) -> torch.Tensor:
    return cotangent_stiffness_apply(
        cotangent_stiffness_apply(values, edge_index, edge_weight, diagonal),
        edge_index,
        edge_weight,
        diagonal,
    ) + regularization * values


def _normal_diagonal(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
    regularization: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    edges = edge_index.to(device=device, dtype=torch.long)
    weight_squared = edge_weight.to(device=device, dtype=dtype).square()
    result = diagonal.to(device=device, dtype=dtype).square() + regularization
    result = result.clone()
    result.index_add_(0, edges[0], weight_squared)
    result.index_add_(0, edges[1], weight_squared)
    return result.clamp_min(torch.finfo(dtype).eps).unsqueeze(-1)


def _pcg_solve(
    right_hand_side: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
    regularization: torch.Tensor,
    *,
    maximum_iterations: int,
    tolerance: float,
    initial_guess: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    if maximum_iterations < 1 or tolerance <= 0:
        raise ValueError("PCG iteration budget and tolerance must be positive.")
    if not bool(torch.isfinite(regularization)) or bool(regularization <= 0):
        raise ValueError("Cotangent recovery requires positive finite regularization.")
    solution = (
        torch.zeros_like(right_hand_side)
        if initial_guess is None
        else initial_guess.clone()
    )
    preconditioner = _normal_diagonal(
        edge_index,
        edge_weight,
        diagonal,
        regularization,
        dtype=right_hand_side.dtype,
        device=right_hand_side.device,
    )
    residual = right_hand_side - _normal_apply(
        solution, edge_index, edge_weight, diagonal, regularization
    )
    preconditioned = residual / preconditioner
    direction = preconditioned.clone()
    residual_preconditioned = (residual * preconditioned).sum()
    rhs_norm = torch.linalg.vector_norm(right_hand_side).clamp_min(
        torch.finfo(right_hand_side.dtype).eps
    )
    iterations = 0
    tiny = torch.finfo(right_hand_side.dtype).tiny
    for iteration in range(maximum_iterations):
        if bool(torch.linalg.vector_norm(residual) <= tolerance * rhs_norm):
            break
        matrix_direction = _normal_apply(
            direction, edge_index, edge_weight, diagonal, regularization
        )
        denominator = (direction * matrix_direction).sum()
        if bool(torch.abs(denominator) <= tiny):
            break
        alpha = residual_preconditioned / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * matrix_direction
        iterations = iteration + 1
        if bool(torch.linalg.vector_norm(residual) <= tolerance * rhs_norm):
            break
        next_preconditioned = residual / preconditioner
        next_residual_preconditioned = (residual * next_preconditioned).sum()
        if bool(torch.abs(residual_preconditioned) <= tiny):
            break
        beta = next_residual_preconditioned / residual_preconditioned
        direction = next_preconditioned + beta * direction
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned
    final_residual = right_hand_side - _normal_apply(
        solution, edge_index, edge_weight, diagonal, regularization
    )
    relative = torch.linalg.vector_norm(final_residual) / rhs_norm
    return solution, ConjugateGradientAudit(
        iterations=iterations,
        converged=bool(relative <= tolerance * 1.05),
        relative_residual=float(relative.detach().cpu()),
    )


class _CotangentSparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        predicted_laplacian: torch.Tensor,
        direct_vertices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        diagonal: torch.Tensor,
        regularization: torch.Tensor,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        regularization = regularization.to(
            device=predicted_laplacian.device, dtype=predicted_laplacian.dtype
        ).reshape(())
        rhs = cotangent_stiffness_apply(
            predicted_laplacian, edge_index, edge_weight, diagonal
        ) + regularization * direct_vertices
        solution, audit = _pcg_solve(
            rhs,
            edge_index,
            edge_weight,
            diagonal,
            regularization,
            maximum_iterations=int(maximum_iterations),
            tolerance=float(tolerance),
            initial_guess=direct_vertices,
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable cotangent solve did not converge: "
                f"relative_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}."
            )
        ctx.save_for_backward(edge_index, edge_weight, diagonal, regularization)
        ctx.maximum_iterations = int(maximum_iterations)
        ctx.tolerance = float(tolerance)
        audit_tensor = torch.tensor(
            [float(audit.iterations), 1.0, float(audit.relative_residual)],
            dtype=torch.float64,
            device=solution.device,
        )
        ctx.mark_non_differentiable(audit_tensor)
        return solution, audit_tensor

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor, audit_gradient: torch.Tensor | None):
        edge_index, edge_weight, diagonal, regularization = ctx.saved_tensors
        adjoint, audit = _pcg_solve(
            output_gradient,
            edge_index,
            edge_weight,
            diagonal,
            regularization,
            maximum_iterations=ctx.maximum_iterations,
            tolerance=ctx.tolerance,
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable cotangent adjoint did not converge: "
                f"relative_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}."
            )
        prediction_gradient = cotangent_stiffness_apply(
            adjoint, edge_index, edge_weight, diagonal
        )
        direct_gradient = regularization * adjoint
        return prediction_gradient, direct_gradient, None, None, None, None, None, None


def differentiable_cotangent_sparse_recovery_with_audit(
    predicted_laplacian: torch.Tensor,
    direct_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    regularization: float,
    maximum_iterations: int = 2048,
    tolerance: float = 1e-8,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    if predicted_laplacian.shape != direct_vertices.shape:
        raise ValueError("predicted_laplacian and direct_vertices must share [N, 3].")
    regularization_tensor = torch.as_tensor(
        regularization,
        dtype=predicted_laplacian.dtype,
        device=predicted_laplacian.device,
    ).reshape(())
    recovered, audit_tensor = _CotangentSparseSolve.apply(
        predicted_laplacian,
        direct_vertices,
        edge_index,
        edge_weight,
        diagonal,
        regularization_tensor,
        int(maximum_iterations),
        float(tolerance),
    )
    values = audit_tensor.detach().cpu().tolist()
    return recovered, ConjugateGradientAudit(
        iterations=int(values[0]),
        converged=bool(values[1]),
        relative_residual=float(values[2]),
    )


def differentiable_cotangent_sparse_recovery(
    predicted_laplacian: torch.Tensor,
    direct_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    regularization: float,
    maximum_iterations: int = 2048,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    recovered, _ = differentiable_cotangent_sparse_recovery_with_audit(
        predicted_laplacian,
        direct_vertices,
        edge_index,
        edge_weight,
        diagonal,
        regularization=regularization,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    return recovered

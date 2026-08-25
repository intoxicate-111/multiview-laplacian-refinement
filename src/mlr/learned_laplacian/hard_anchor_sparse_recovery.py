from __future__ import annotations

"""Differentiable zero-regularization Laplacian recovery with hard anchors."""

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr, splu

from .differentiable_sparse_recovery import (
    ConjugateGradientAudit,
    _jacobi_diagonal,
    uniform_laplacian_apply,
    uniform_laplacian_transpose_apply,
)


def deterministic_component_anchor_indices(
    edge_index: torch.Tensor,
    num_vertices: int,
) -> torch.Tensor:
    """Return the lowest global vertex index in every undirected component."""
    vertices = int(num_vertices)
    if vertices < 1:
        raise ValueError("num_vertices must be positive.")
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu().numpy()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edges.size and (edges.min() < 0 or edges.max() >= vertices):
        raise ValueError("edge_index contains an out-of-range vertex index.")
    adjacency = coo_matrix(
        (
            np.ones(edges.shape[1], dtype=np.float64),
            (edges[0], edges[1]),
        ),
        shape=(vertices, vertices),
    ).tocsr()
    count, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    anchors = np.full(int(count), vertices, dtype=np.int64)
    np.minimum.at(anchors, labels, np.arange(vertices, dtype=np.int64))
    if np.any(anchors >= vertices):
        raise RuntimeError("Failed to assign one deterministic anchor per component.")
    return torch.from_numpy(anchors)


def hard_anchor_sparse_recovery_lsmr(
    laplacian: csr_matrix,
    predicted_laplacian: np.ndarray,
    initial_vertices: np.ndarray,
    anchor_indices: np.ndarray,
    *,
    atol: float = 1e-12,
    btol: float = 1e-12,
    maxiter: int = 100000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Float64 reference solve after explicitly eliminating anchored unknowns."""
    matrix = csr_matrix(laplacian, dtype=np.float64)
    target = np.asarray(predicted_laplacian, dtype=np.float64)
    initial = np.asarray(initial_vertices, dtype=np.float64)
    anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    if matrix.shape[0] != matrix.shape[1] or target.shape != initial.shape:
        raise ValueError("Invalid Laplacian, prediction or initial-vertex shape.")
    if target.shape != (matrix.shape[0], 3):
        raise ValueError("Prediction and initial vertices must have shape [N, 3].")
    if len(anchors) < 1 or len(np.unique(anchors)) != len(anchors):
        raise ValueError("At least one unique hard anchor is required.")
    anchor_mask = np.zeros(matrix.shape[0], dtype=bool)
    anchor_mask[anchors] = True
    free = np.flatnonzero(~anchor_mask)
    anchor_only = np.zeros_like(initial)
    anchor_only[anchors] = initial[anchors]
    reduced = matrix[:, free]
    rhs = target - matrix @ anchor_only
    solution = anchor_only.copy()
    axes: list[dict[str, float | int]] = []
    for axis in range(3):
        result = lsmr(
            reduced,
            rhs[:, axis],
            atol=atol,
            btol=btol,
            conlim=1e12,
            maxiter=maxiter,
        )
        solution[free, axis] = result[0]
        axes.append(
            {
                "axis": axis,
                "istop": int(result[1]),
                "iterations": int(result[2]),
                "norm_residual": float(result[3]),
                "norm_normal_residual": float(result[4]),
                "operator_norm": float(result[5]),
                "condition_estimate": float(result[6]),
                "solution_norm": float(result[7]),
            }
        )
    solution[anchors] = initial[anchors]
    residual = matrix @ solution - target
    residual_norm = np.linalg.norm(residual, axis=1)
    return solution, {
        "axes": axes,
        "all_converged": all(row["istop"] in (1, 2, 4, 5) for row in axes),
        "maximum_iterations": max(int(row["iterations"]) for row in axes),
        "maximum_condition_estimate": max(
            float(row["condition_estimate"]) for row in axes
        ),
        "laplacian_residual_rms": float(
            np.sqrt(np.mean(np.square(residual_norm)))
        ),
        "laplacian_residual_max": float(residual_norm.max(initial=0.0)),
        "anchor_max_abs_error": float(
            np.max(np.abs(solution[anchors] - initial[anchors]), initial=0.0)
        ),
        "hidden_regularization": False,
        "centroid_constraint": False,
    }


def _validate_anchor_indices(
    anchor_indices: torch.Tensor,
    vertices: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors = torch.as_tensor(
        anchor_indices, dtype=torch.long, device=device
    ).reshape(-1)
    if anchors.numel() < 1:
        raise ValueError("At least one hard anchor is required.")
    if bool(torch.any(anchors < 0)) or bool(torch.any(anchors >= vertices)):
        raise ValueError("hard anchor index is out of range.")
    if int(torch.unique(anchors).numel()) != int(anchors.numel()):
        raise ValueError("hard anchor indices must be unique.")
    anchor_mask = torch.zeros(vertices, dtype=torch.bool, device=device)
    anchor_mask[anchors] = True
    free = torch.nonzero(~anchor_mask, as_tuple=False).reshape(-1)
    if free.numel() < 1:
        raise ValueError("Hard anchors cannot eliminate every vertex.")
    return anchors, free


def _scatter_free(
    free_values: torch.Tensor,
    free_indices: torch.Tensor,
    vertices: int,
) -> torch.Tensor:
    full = torch.zeros(
        (vertices, free_values.shape[1]),
        dtype=free_values.dtype,
        device=free_values.device,
    )
    full.index_copy_(0, free_indices, free_values)
    return full


def _reduced_normal_apply(
    free_values: torch.Tensor,
    free_indices: torch.Tensor,
    vertices: int,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
) -> torch.Tensor:
    full = _scatter_free(free_values, free_indices, vertices)
    normal = uniform_laplacian_transpose_apply(
        uniform_laplacian_apply(full, edge_index, vertex_degree),
        edge_index,
        vertex_degree,
    )
    return normal.index_select(0, free_indices)


def _reduced_pcg_solve(
    right_hand_side: torch.Tensor,
    free_indices: torch.Tensor,
    vertices: int,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    *,
    maximum_iterations: int,
    tolerance: float,
    initial_guess: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    rhs = right_hand_side
    solution = torch.zeros_like(rhs) if initial_guess is None else initial_guess.clone()
    diagonal = _jacobi_diagonal(
        edge_index,
        vertex_degree,
        vertices=vertices,
        dtype=rhs.dtype,
        device=rhs.device,
        regularization=0.0,
    ).index_select(0, free_indices).unsqueeze(-1)
    residual = rhs - _reduced_normal_apply(
        solution, free_indices, vertices, edge_index, vertex_degree
    )
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_preconditioned = (residual * preconditioned).sum()
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    active = torch.linalg.vector_norm(residual) > tolerance * rhs_norm
    iterations = 0
    tiny = torch.finfo(rhs.dtype).tiny
    for iteration in range(maximum_iterations):
        if not bool(active):
            break
        matrix_direction = _reduced_normal_apply(
            direction, free_indices, vertices, edge_index, vertex_degree
        )
        denominator = (direction * matrix_direction).sum()
        if not bool(torch.isfinite(denominator)) or bool(denominator <= tiny):
            break
        alpha = residual_preconditioned / denominator
        solution = solution + direction * alpha
        residual = residual - matrix_direction * alpha
        iterations = iteration + 1
        active = torch.linalg.vector_norm(residual) > tolerance * rhs_norm
        if not bool(active):
            break
        next_preconditioned = residual / diagonal
        next_residual_preconditioned = (residual * next_preconditioned).sum()
        if not bool(torch.isfinite(next_residual_preconditioned)):
            break
        beta = next_residual_preconditioned / residual_preconditioned.clamp_min(tiny)
        direction = next_preconditioned + direction * beta
        preconditioned = next_preconditioned
        residual_preconditioned = next_residual_preconditioned
    final_residual = rhs - _reduced_normal_apply(
        solution, free_indices, vertices, edge_index, vertex_degree
    )
    relative = torch.linalg.vector_norm(final_residual) / rhs_norm
    return solution, ConjugateGradientAudit(
        iterations=iterations,
        converged=bool(torch.isfinite(relative) and relative <= tolerance * 1.05),
        relative_residual=float(relative.detach().cpu()),
    )


class _HardAnchorSparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        predicted_laplacian: torch.Tensor,
        initial_vertices: torch.Tensor,
        edge_index: torch.Tensor,
        vertex_degree: torch.Tensor,
        anchor_indices: torch.Tensor,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vertices = int(predicted_laplacian.shape[0])
        anchors, free = _validate_anchor_indices(
            anchor_indices, vertices, predicted_laplacian.device
        )
        anchor_only = torch.zeros_like(initial_vertices)
        anchor_only.index_copy_(
            0, anchors, initial_vertices.index_select(0, anchors)
        )
        shifted_target = predicted_laplacian - uniform_laplacian_apply(
            anchor_only, edge_index, vertex_degree
        )
        rhs = uniform_laplacian_transpose_apply(
            shifted_target, edge_index, vertex_degree
        ).index_select(0, free)
        free_solution, audit = _reduced_pcg_solve(
            rhs,
            free,
            vertices,
            edge_index,
            vertex_degree,
            maximum_iterations=int(maximum_iterations),
            tolerance=float(tolerance),
            initial_guess=initial_vertices.index_select(0, free),
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable hard-anchor sparse solve did not converge: "
                f"relative_normal_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}. No damping was added."
            )
        solution = anchor_only.index_copy(0, free, free_solution)
        # Enforce and audit exact bitwise anchor coordinates in the returned mesh.
        solution.index_copy_(0, anchors, initial_vertices.index_select(0, anchors))
        ctx.save_for_backward(edge_index, vertex_degree, anchors, free)
        ctx.vertices = vertices
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
    def backward(
        ctx,
        output_gradient: torch.Tensor,
        audit_gradient: torch.Tensor | None,
    ):
        edge_index, vertex_degree, anchors, free = ctx.saved_tensors
        adjoint_free, audit = _reduced_pcg_solve(
            output_gradient.index_select(0, free),
            free,
            ctx.vertices,
            edge_index,
            vertex_degree,
            maximum_iterations=ctx.maximum_iterations,
            tolerance=ctx.tolerance,
        )
        if not audit.converged:
            raise RuntimeError(
                "Differentiable hard-anchor sparse adjoint did not converge: "
                f"relative_normal_residual={audit.relative_residual:.6g}, "
                f"iterations={audit.iterations}. No damping was added."
            )
        adjoint_full = _scatter_free(adjoint_free, free, ctx.vertices)
        prediction_gradient = uniform_laplacian_apply(
            adjoint_full, edge_index, vertex_degree
        )
        initial_gradient = torch.zeros_like(output_gradient)
        normal_adjoint = uniform_laplacian_transpose_apply(
            prediction_gradient, edge_index, vertex_degree
        )
        anchor_gradient = output_gradient.index_select(0, anchors) - (
            normal_adjoint.index_select(0, anchors)
        )
        initial_gradient.index_copy_(0, anchors, anchor_gradient)
        return prediction_gradient, initial_gradient, None, None, None, None, None


def _scipy_uniform_laplacian(
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    vertices: int,
) -> csr_matrix:
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu().numpy()
    degree = (
        torch.as_tensor(vertex_degree, dtype=torch.float64)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if degree.shape != (vertices,) or not np.isfinite(degree).all() or np.any(degree <= 0):
        raise ValueError("vertex_degree must contain N finite positive entries.")
    source, destination = edges
    rows = np.concatenate((np.arange(vertices, dtype=np.int64), destination))
    columns = np.concatenate((np.arange(vertices, dtype=np.int64), source))
    values = np.concatenate(
        (np.ones(vertices, dtype=np.float64), -1.0 / degree[destination])
    )
    return csr_matrix((values, (rows, columns)), shape=(vertices, vertices))


class _HardAnchorSparseDirectSolve(torch.autograd.Function):
    """Exact reduced normal-equation solve using an undamped sparse LU factor."""

    @staticmethod
    def forward(
        ctx,
        predicted_laplacian: torch.Tensor,
        initial_vertices: torch.Tensor,
        edge_index: torch.Tensor,
        vertex_degree: torch.Tensor,
        anchor_indices: torch.Tensor,
        maximum_iterations: int,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if predicted_laplacian.dtype != torch.float64:
            raise ValueError("Hard-anchor direct sparse recovery requires float64 input.")
        vertices = int(predicted_laplacian.shape[0])
        anchors_t, free_t = _validate_anchor_indices(
            anchor_indices, vertices, predicted_laplacian.device
        )
        anchors = anchors_t.detach().cpu().numpy()
        free = free_t.detach().cpu().numpy()
        prediction = predicted_laplacian.detach().cpu().numpy()
        initial = initial_vertices.detach().to(dtype=torch.float64, device="cpu").numpy()
        matrix = _scipy_uniform_laplacian(edge_index, vertex_degree, vertices)
        reduced = matrix[:, free].tocsr()
        anchored = matrix[:, anchors].tocsr()
        rhs_observation = prediction - anchored @ initial[anchors]
        normal = (reduced.T @ reduced).tocsc()
        rhs = reduced.T @ rhs_observation
        try:
            factor = splu(
                normal,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Reduced hard-anchor system is singular; no damping was added."
            ) from error
        free_solution = np.asarray(factor.solve(np.asarray(rhs)), dtype=np.float64)
        normal_residual = normal @ free_solution - rhs
        rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).eps)
        relative_residual = float(np.linalg.norm(normal_residual) / rhs_norm)
        if not np.isfinite(relative_residual) or relative_residual > float(tolerance) * 1.05:
            raise RuntimeError(
                "Direct hard-anchor sparse solve failed its residual audit: "
                f"relative_normal_residual={relative_residual:.6g}. No damping was added."
            )
        solution = initial.copy()
        solution[free] = free_solution
        solution[anchors] = initial[anchors]
        recovered = torch.from_numpy(solution).to(
            device=predicted_laplacian.device, dtype=predicted_laplacian.dtype
        )
        ctx.factor = factor
        ctx.reduced = reduced
        ctx.anchored = anchored
        ctx.input_device = predicted_laplacian.device
        ctx.input_dtype = predicted_laplacian.dtype
        ctx.anchors = anchors
        ctx.free = free
        audit_tensor = torch.tensor(
            [1.0, 1.0, relative_residual],
            dtype=torch.float64,
            device=predicted_laplacian.device,
        )
        ctx.mark_non_differentiable(audit_tensor)
        return recovered, audit_tensor

    @staticmethod
    def backward(
        ctx,
        output_gradient: torch.Tensor,
        audit_gradient: torch.Tensor | None,
    ):
        output_cpu = output_gradient.detach().to(dtype=torch.float64, device="cpu").numpy()
        adjoint_free = np.asarray(
            ctx.factor.solve(output_cpu[ctx.free], trans="T"), dtype=np.float64
        )
        prediction_gradient = np.asarray(ctx.reduced @ adjoint_free, dtype=np.float64)
        initial_gradient = np.zeros_like(output_cpu)
        initial_gradient[ctx.anchors] = output_cpu[ctx.anchors] - np.asarray(
            ctx.anchored.T @ prediction_gradient, dtype=np.float64
        )
        prediction_tensor = torch.from_numpy(prediction_gradient).to(
            device=ctx.input_device, dtype=ctx.input_dtype
        )
        initial_tensor = torch.from_numpy(initial_gradient).to(
            device=ctx.input_device, dtype=ctx.input_dtype
        )
        return prediction_tensor, initial_tensor, None, None, None, None, None


def differentiable_hard_anchor_sparse_recovery_with_audit(
    predicted_laplacian: torch.Tensor,
    initial_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    maximum_iterations: int = 2048,
    tolerance: float = 1e-4,
) -> tuple[torch.Tensor, ConjugateGradientAudit]:
    """Solve the true lambda=0 constrained problem with implicit gradients."""
    if predicted_laplacian.shape != initial_vertices.shape:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    if predicted_laplacian.ndim != 2 or predicted_laplacian.shape[1] != 3:
        raise ValueError("predicted_laplacian and initial_vertices must have shape [N, 3].")
    # Jacobi-PCG on the undamped normal equations is too ill-conditioned on
    # real Sofa50 meshes.  Use a direct, undamped reduced sparse factorization;
    # maximum_iterations remains in the API only to preserve trainer settings.
    recovered, audit_tensor = _HardAnchorSparseDirectSolve.apply(
        predicted_laplacian,
        initial_vertices,
        edge_index,
        vertex_degree,
        anchor_indices,
        int(maximum_iterations),
        float(tolerance),
    )
    values = audit_tensor.detach().cpu().tolist()
    return recovered, ConjugateGradientAudit(
        iterations=int(values[0]),
        converged=bool(values[1]),
        relative_residual=float(values[2]),
    )


def differentiable_hard_anchor_sparse_recovery(
    predicted_laplacian: torch.Tensor,
    initial_vertices: torch.Tensor,
    edge_index: torch.Tensor,
    vertex_degree: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    maximum_iterations: int = 2048,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    recovered, _ = differentiable_hard_anchor_sparse_recovery_with_audit(
        predicted_laplacian,
        initial_vertices,
        edge_index,
        vertex_degree,
        anchor_indices,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    return recovered

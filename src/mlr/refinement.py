from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Array, Mesh
from .laplacian import LaplacianOperator, build_laplacian, unique_edges
from .losses import robust_value_and_grad


@dataclass
class RefinementConfig:
    operator_type: str = "uniform"
    lambda_lap: float = 1.0
    lambda_anchor: float = 0.05
    lambda_edge: float = 0.0
    lambda_unseen_anchor: float = 0.0
    num_iters: int = 200
    learning_rate: float = 1e-2
    robust_loss: str = "charbonnier"
    charbonnier_epsilon: float = 1e-3
    huber_delta: float = 1e-2
    freeze_pseudo_target: bool = True
    log_every: int = 25
    print_every: int = 0


@dataclass
class RefinementResult:
    mesh: Mesh
    vertices: Array
    history: list[dict[str, float]] = field(default_factory=list)
    operator: LaplacianOperator | None = None


def refine_mesh_with_laplacian(
    mesh: Mesh,
    delta_target: Array,
    confidence: Array | None = None,
    anchors: Array | None = None,
    config: RefinementConfig | None = None,
    *,
    laplacian_weight: Array | None = None,
) -> RefinementResult:
    config = config or RefinementConfig()
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    delta_target = np.asarray(delta_target, dtype=np.float64)
    if delta_target.shape != vertices.shape:
        raise ValueError("delta_target must have shape (N, 3).")

    operator = build_laplacian(vertices, mesh.faces, config.operator_type)
    lap = operator.matrix
    anchors = np.asarray(mesh.vertices if anchors is None else anchors, dtype=np.float64)
    if anchors.shape != vertices.shape:
        raise ValueError("anchors must have shape (N, 3).")

    if confidence is None:
        conf = np.ones((mesh.num_vertices, 1), dtype=np.float64)
    else:
        conf = np.asarray(confidence, dtype=np.float64).reshape(mesh.num_vertices, 1)
        conf = np.clip(conf, 0.0, None)
        max_conf = conf.max(initial=0.0)
        if max_conf > 0:
            conf = conf / max_conf

    if laplacian_weight is None:
        lap_weight = np.ones((mesh.num_vertices, 1), dtype=np.float64)
    else:
        lap_weight = np.asarray(laplacian_weight, dtype=np.float64).reshape(
            mesh.num_vertices, 1
        )
        if not np.isfinite(lap_weight).all() or np.any(lap_weight < 0):
            raise ValueError("laplacian_weight must be finite and non-negative.")

    edge_pairs = unique_edges(mesh.faces)
    target_edge_lengths = None
    if config.lambda_edge > 0 and len(edge_pairs) > 0:
        target_edge_lengths = np.linalg.norm(
            mesh.vertices[edge_pairs[:, 0]] - mesh.vertices[edge_pairs[:, 1]], axis=1
        )

    m = np.zeros_like(vertices)
    v = np.zeros_like(vertices)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history: list[dict[str, float]] = []

    for step in range(1, config.num_iters + 1):
        loss, grad, parts = _loss_and_grad(
            vertices=vertices,
            lap=lap,
            delta_target=delta_target,
            confidence=conf,
            laplacian_weight=lap_weight,
            anchors=anchors,
            edge_pairs=edge_pairs,
            target_edge_lengths=target_edge_lengths,
            config=config,
        )
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        vertices -= config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if step == 1 or step == config.num_iters or step % config.log_every == 0:
            history.append({"iter": float(step), "loss": float(loss), **parts})
        if config.print_every > 0 and (step == 1 or step == config.num_iters or step % config.print_every == 0):
            extras = ", ".join(f"{name}={value:.6f}" for name, value in parts.items())
            suffix = f", {extras}" if extras else ""
            if step == 1:
                tag = "start"
            elif step == config.num_iters:
                tag = "final"
            else:
                tag = "iter"
            print(f"{tag} step={step} loss={loss:.6f}{suffix}")

    refined = mesh.with_vertices(vertices)
    return RefinementResult(mesh=refined, vertices=vertices, history=history, operator=operator)


def refine_mesh_position_only(
    mesh: Mesh,
    target_vertices: Array,
    confidence: Array | None = None,
    config: RefinementConfig | None = None,
) -> RefinementResult:
    config = config or RefinementConfig(lambda_lap=0.0, lambda_anchor=1.0)
    vertices = np.array(mesh.vertices, dtype=np.float64, copy=True)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    if confidence is None:
        conf = np.ones((mesh.num_vertices, 1), dtype=np.float64)
    else:
        conf = np.asarray(confidence, dtype=np.float64).reshape(mesh.num_vertices, 1)
    m = np.zeros_like(vertices)
    v = np.zeros_like(vertices)
    history: list[dict[str, float]] = []
    for step in range(1, config.num_iters + 1):
        residual = (vertices - target_vertices) * conf
        value, grad_robust = robust_value_and_grad(
            residual,
            loss_type=config.robust_loss,
            epsilon=config.charbonnier_epsilon,
            huber_delta=config.huber_delta,
        )
        grad = config.lambda_anchor * grad_robust * conf
        loss = config.lambda_anchor * value
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        m_hat = m / (1.0 - 0.9**step)
        v_hat = v / (1.0 - 0.999**step)
        vertices -= config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        if step == 1 or step == config.num_iters or step % config.log_every == 0:
            history.append({"iter": float(step), "loss": float(loss), "position": float(loss)})
        if config.print_every > 0 and (step == 1 or step == config.num_iters or step % config.print_every == 0):
            if step == 1:
                tag = "start"
            elif step == config.num_iters:
                tag = "final"
            else:
                tag = "iter"
            print(f"{tag} step={step} loss={loss:.6f} position={loss:.6f}")
    return RefinementResult(mesh=mesh.with_vertices(vertices), vertices=vertices, history=history)


def zero_laplacian_smooth(mesh: Mesh, config: RefinementConfig | None = None) -> RefinementResult:
    config = config or RefinementConfig(lambda_lap=1.0, lambda_anchor=0.05)
    delta_zero = np.zeros_like(mesh.vertices)
    return refine_mesh_with_laplacian(
        mesh=mesh,
        delta_target=delta_zero,
        confidence=np.ones(mesh.num_vertices),
        anchors=mesh.vertices,
        config=config,
    )


def _loss_and_grad(
    vertices: Array,
    lap: Array,
    delta_target: Array,
    confidence: Array,
    laplacian_weight: Array,
    anchors: Array,
    edge_pairs: Array,
    target_edge_lengths: Array | None,
    config: RefinementConfig,
) -> tuple[float, Array, dict[str, float]]:
    grad = np.zeros_like(vertices)
    total = 0.0
    parts: dict[str, float] = {}

    if config.lambda_lap > 0:
        sqrt_weight = np.sqrt(laplacian_weight)
        residual_lap = visibility_weighted_laplacian_residual(
            lap @ vertices - delta_target, laplacian_weight
        ) * confidence
        lap_value, lap_grad_robust = robust_value_and_grad(
            residual_lap,
            loss_type=config.robust_loss,
            epsilon=config.charbonnier_epsilon,
            huber_delta=config.huber_delta,
        )
        grad += config.lambda_lap * (
            lap.T @ (lap_grad_robust * confidence * sqrt_weight)
        )
        total += config.lambda_lap * lap_value
        parts["laplacian"] = float(config.lambda_lap * lap_value)

    if config.lambda_anchor > 0:
        residual_anchor = vertices - anchors
        anchor_value = 0.5 * float(np.sum(residual_anchor * residual_anchor))
        grad += config.lambda_anchor * residual_anchor
        total += config.lambda_anchor * anchor_value
        parts["anchor"] = float(config.lambda_anchor * anchor_value)

    if config.lambda_unseen_anchor > 0:
        unseen = (laplacian_weight <= 0).astype(np.float64)
        unseen_residual = (vertices - anchors) * unseen
        unseen_value = 0.5 * float(np.sum(unseen_residual * unseen_residual))
        grad += config.lambda_unseen_anchor * unseen_residual
        total += config.lambda_unseen_anchor * unseen_value
        parts["unseen_anchor"] = float(config.lambda_unseen_anchor * unseen_value)

    if config.lambda_edge > 0 and target_edge_lengths is not None and len(edge_pairs) > 0:
        edge_value, edge_grad = _edge_length_value_and_grad(vertices, edge_pairs, target_edge_lengths)
        grad += config.lambda_edge * edge_grad
        total += config.lambda_edge * edge_value
        parts["edge"] = float(config.lambda_edge * edge_value)

    return total, grad, parts


def visibility_weighted_laplacian_residual(
    laplacian_residual: Array, laplacian_weight: Array
) -> Array:
    """Apply sqrt(W) to complete Laplacian equation residual rows."""

    residual = np.asarray(laplacian_residual)
    weight = np.asarray(laplacian_weight, dtype=residual.dtype).reshape(-1, 1)
    if residual.ndim != 2 or residual.shape[0] != weight.shape[0]:
        raise ValueError("residual must be [N, C] and laplacian_weight must contain N values.")
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("laplacian_weight must be finite and non-negative.")
    return residual * np.sqrt(weight)


def _edge_length_value_and_grad(vertices: Array, edges: Array, target_lengths: Array) -> tuple[float, Array]:
    grad = np.zeros_like(vertices)
    diff = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    lengths = np.linalg.norm(diff, axis=1)
    residual = lengths - target_lengths
    value = 0.5 * float(np.sum(residual * residual))
    direction = diff / np.maximum(lengths[:, None], 1e-12)
    edge_grad = residual[:, None] * direction
    np.add.at(grad, edges[:, 0], edge_grad)
    np.add.at(grad, edges[:, 1], -edge_grad)
    return value, grad

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Array


@dataclass(frozen=True)
class LaplacianOperator:
    matrix: Array
    operator_type: str

    def apply(self, vertices: Array) -> Array:
        return self.matrix @ np.asarray(vertices, dtype=np.float64)


def build_laplacian(vertices: Array, faces: Array, operator_type: str = "uniform") -> LaplacianOperator:
    if operator_type == "uniform":
        return LaplacianOperator(build_uniform_laplacian(faces, len(vertices)), operator_type)
    if operator_type == "cotangent":
        return LaplacianOperator(build_cotangent_laplacian(vertices, faces), operator_type)
    raise ValueError(f"Unsupported Laplacian operator: {operator_type}")


def compute_laplacian_coordinates(vertices: Array, faces: Array, operator_type: str = "uniform") -> Array:
    operator = build_laplacian(vertices, faces, operator_type)
    return operator.apply(vertices)


def compute_laplacian_target(positions: Array, faces: Array, operator_type: str = "uniform") -> Array:
    return compute_laplacian_coordinates(positions, faces, operator_type)


def build_uniform_laplacian(faces: Array, num_vertices: int) -> Array:
    neighbors = vertex_neighbors(faces, num_vertices)
    lap = np.zeros((num_vertices, num_vertices), dtype=np.float64)
    for idx, nbrs in enumerate(neighbors):
        lap[idx, idx] = 1.0
        if not nbrs:
            continue
        weight = -1.0 / len(nbrs)
        for nbr in nbrs:
            lap[idx, nbr] = weight
    return lap


def build_cotangent_laplacian(vertices: Array, faces: Array) -> Array:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)
    weights = np.zeros((n, n), dtype=np.float64)

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cot_i = _cotangent(vj - vi, vk - vi)
        cot_j = _cotangent(vi - vj, vk - vj)
        cot_k = _cotangent(vi - vk, vj - vk)
        _add_symmetric_weight(weights, j, k, cot_i)
        _add_symmetric_weight(weights, i, k, cot_j)
        _add_symmetric_weight(weights, i, j, cot_k)

    lap = np.zeros((n, n), dtype=np.float64)
    row_sums = weights.sum(axis=1)
    for i in range(n):
        if row_sums[i] <= 1e-12:
            lap[i, i] = 1.0
            continue
        lap[i, i] = 1.0
        lap[i, :] -= weights[i, :] / row_sums[i]
        lap[i, i] += weights[i, i] / row_sums[i]
    return lap


def vertex_neighbors(faces: Array, num_vertices: int) -> list[set[int]]:
    neighbors = [set() for _ in range(num_vertices)]
    for a, b, c in np.asarray(faces, dtype=np.int64):
        neighbors[a].update((int(b), int(c)))
        neighbors[b].update((int(a), int(c)))
        neighbors[c].update((int(a), int(b)))
    return neighbors


def edge_lengths(vertices: Array, faces: Array) -> Array:
    edges = unique_edges(faces)
    diff = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    return np.linalg.norm(diff, axis=1)


def unique_edges(faces: Array) -> Array:
    edges = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add(tuple(sorted((int(u), int(v)))))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def _cotangent(a: Array, b: Array) -> float:
    cross_norm = np.linalg.norm(np.cross(a, b))
    if cross_norm < 1e-12:
        return 0.0
    return float(np.dot(a, b) / cross_norm)


def _add_symmetric_weight(weights: Array, i: int, j: int, value: float) -> None:
    value = max(0.0, 0.5 * float(value))
    weights[i, j] += value
    weights[j, i] += value

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Array, Mesh
from .laplacian import compute_laplacian_coordinates
from .io import save_mesh
from .refinement import RefinementConfig, RefinementResult, refine_mesh_with_laplacian


@dataclass(frozen=True)
class GTLaplacianTargetConfig:
    operator_type: str = "uniform"
    distance_confidence_scale: float | None = None
    min_confidence: float = 0.0


@dataclass
class InterpolatedLaplacianTarget:
    delta_target: Array
    confidence: Array
    closest_points: Array
    face_indices: Array
    barycentric: Array
    distances: Array


@dataclass
class GTLaplacianRefinementResult:
    mesh: Mesh
    vertices: Array
    history: list[dict[str, float]]
    target: InterpolatedLaplacianTarget
    refinement: RefinementResult


def compute_coarse_graph_gt_laplacian_target(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    config: GTLaplacianTargetConfig | None = None,
    gt_laplacian_values: Array | None = None,
) -> InterpolatedLaplacianTarget:
    """Compute coarse-graph Laplacian targets from GT-projected coarse vertices."""

    config = config or GTLaplacianTargetConfig()
    if gt_laplacian_values is not None:
        raise ValueError("gt_laplacian_values is no longer supported for this target.")

    closest = closest_points_on_mesh(coarse_mesh.vertices, gt_mesh.vertices, gt_mesh.faces)
    projected = closest.points
    delta_target = compute_laplacian_coordinates(
        projected,
        coarse_mesh.faces,
        config.operator_type,
    )
    assert projected.shape == coarse_mesh.vertices.shape
    assert delta_target.shape == coarse_mesh.vertices.shape
    assert len(closest.face_indices) == coarse_mesh.num_vertices
    assert len(closest.distances) == coarse_mesh.num_vertices
    confidence = _distance_confidence(
        closest.distances,
        distance_confidence_scale=config.distance_confidence_scale,
        min_confidence=config.min_confidence,
    )
    return InterpolatedLaplacianTarget(
        delta_target=delta_target,
        confidence=confidence,
        closest_points=projected,
        face_indices=closest.face_indices,
        barycentric=closest.barycentric,
        distances=closest.distances,
    )


def interpolate_gt_laplacian_to_coarse(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    config: GTLaplacianTargetConfig | None = None,
    gt_laplacian_values: Array | None = None,
) -> InterpolatedLaplacianTarget:
    return compute_coarse_graph_gt_laplacian_target(
        coarse_mesh=coarse_mesh,
        gt_mesh=gt_mesh,
        config=config,
        gt_laplacian_values=gt_laplacian_values,
    )


def refine_coarse_mesh_with_gt_laplacian(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    target_config: GTLaplacianTargetConfig | None = None,
    refinement_config: RefinementConfig | None = None,
    gt_laplacian_values: Array | None = None,
    anchors: Array | None = None,
) -> GTLaplacianRefinementResult:
    """Optimize coarse mesh vertices against interpolated GT Laplacian targets."""

    target_config = target_config or GTLaplacianTargetConfig()
    target = interpolate_gt_laplacian_to_coarse(
        coarse_mesh,
        gt_mesh,
        config=target_config,
        gt_laplacian_values=gt_laplacian_values,
    )
    projected_gt_on_coarse_path = "projected_gt_on_coarse.obj"
    projected_points = target.closest_points
    disp = np.linalg.norm(projected_points - coarse_mesh.vertices, axis=1)
    print("P shape:", projected_points.shape)
    print("V shape:", coarse_mesh.vertices.shape)
    print("mean |P - V|:", float(disp.mean()))
    print("median |P - V|:", float(np.median(disp)))
    print("max |P - V|:", float(disp.max(initial=0.0)))
    print("num unchanged:", int(np.sum(disp < 1e-12)), "/", len(disp))
    print("mean projection distance:", float(target.distances.mean()))
    print("max projection distance:", float(target.distances.max(initial=0.0)))
    assert projected_points.shape == coarse_mesh.vertices.shape
    if not np.allclose(disp, target.distances, atol=1e-8):
        print("disp:", disp)
        print("target.distances:", target.distances)
    assert np.allclose(disp, target.distances, atol=1e-8)
    projected_mesh = Mesh(vertices=projected_points.copy(), faces=coarse_mesh.faces.copy())
    save_mesh(projected_mesh, projected_gt_on_coarse_path)
    if target.delta_target.shape != coarse_mesh.vertices.shape:
        raise ValueError("Interpolated GT Laplacian target must have shape (N_coarse, 3).")

    if refinement_config is None:
        refinement_config = RefinementConfig(operator_type=target_config.operator_type)
    result = refine_mesh_with_laplacian(
        coarse_mesh,
        delta_target=target.delta_target,
        confidence=target.confidence,
        anchors=anchors,
        config=refinement_config,
    )
    return GTLaplacianRefinementResult(
        mesh=result.mesh,
        vertices=result.vertices,
        history=result.history,
        target=target,
        refinement=result,
    )


@dataclass
class ClosestMeshPoints:
    points: Array
    face_indices: Array
    barycentric: Array
    distances: Array


def closest_points_on_mesh(points: Array, vertices: Array, faces: Array) -> ClosestMeshPoints:
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3).")
    if len(faces) == 0:
        raise ValueError("Cannot interpolate on a mesh without faces.")

    tris = vertices[faces]
    closest_points = np.zeros_like(points)
    face_indices = np.zeros(points.shape[0], dtype=np.int64)
    barycentric = np.zeros((points.shape[0], 3), dtype=np.float64)
    distances = np.zeros(points.shape[0], dtype=np.float64)

    for point_idx, point in enumerate(points):
        candidate_points, candidate_bary = _closest_points_on_triangles(point, tris)
        diff = candidate_points - point[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        best = int(np.argmin(dist2))
        closest_points[point_idx] = candidate_points[best]
        face_indices[point_idx] = best
        barycentric[point_idx] = candidate_bary[best]
        distances[point_idx] = float(np.sqrt(dist2[best]))

    return ClosestMeshPoints(
        points=closest_points,
        face_indices=face_indices,
        barycentric=barycentric,
        distances=distances,
    )


def _closest_points_on_triangles(point: Array, triangles: Array) -> tuple[Array, Array]:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    num_tris = triangles.shape[0]
    closest = np.zeros((num_tris, 3), dtype=np.float64)
    bary = np.zeros((num_tris, 3), dtype=np.float64)
    assigned = np.zeros(num_tris, dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    _assign(mask, assigned, closest, bary, a, np.array([1.0, 0.0, 0.0]))

    mask = (d3 >= 0.0) & (d4 <= d3)
    _assign(mask, assigned, closest, bary, b, np.array([0.0, 1.0, 0.0]))

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    edge_mask = mask & ~assigned
    if np.any(edge_mask):
        v = d1[edge_mask] / np.maximum(d1[edge_mask] - d3[edge_mask], 1e-12)
        closest[edge_mask] = a[edge_mask] + v[:, None] * ab[edge_mask]
        bary[edge_mask, 0] = 1.0 - v
        bary[edge_mask, 1] = v
        assigned[edge_mask] = True

    mask = (d6 >= 0.0) & (d5 <= d6)
    _assign(mask, assigned, closest, bary, c, np.array([0.0, 0.0, 1.0]))

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    edge_mask = mask & ~assigned
    if np.any(edge_mask):
        w = d2[edge_mask] / np.maximum(d2[edge_mask] - d6[edge_mask], 1e-12)
        closest[edge_mask] = a[edge_mask] + w[:, None] * ac[edge_mask]
        bary[edge_mask, 0] = 1.0 - w
        bary[edge_mask, 2] = w
        assigned[edge_mask] = True

    va = d3 * d6 - d5 * d4
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    edge_mask = mask & ~assigned
    if np.any(edge_mask):
        w = (d4[edge_mask] - d3[edge_mask]) / np.maximum(
            (d4[edge_mask] - d3[edge_mask]) + (d5[edge_mask] - d6[edge_mask]),
            1e-12,
        )
        closest[edge_mask] = b[edge_mask] + w[:, None] * (c[edge_mask] - b[edge_mask])
        bary[edge_mask, 1] = 1.0 - w
        bary[edge_mask, 2] = w
        assigned[edge_mask] = True

    interior = ~assigned
    if np.any(interior):
        denom = np.maximum(va[interior] + vb[interior] + vc[interior], 1e-12)
        v = vb[interior] / denom
        w = vc[interior] / denom
        u = 1.0 - v - w
        closest[interior] = (
            u[:, None] * a[interior]
            + v[:, None] * b[interior]
            + w[:, None] * c[interior]
        )
        bary[interior, 0] = u
        bary[interior, 1] = v
        bary[interior, 2] = w

    return closest, bary


def _assign(
    mask: Array,
    assigned: Array,
    closest: Array,
    bary: Array,
    points: Array,
    bary_value: Array,
) -> None:
    active = mask & ~assigned
    if not np.any(active):
        return
    closest[active] = points[active]
    bary[active] = bary_value
    assigned[active] = True


def _distance_confidence(
    distances: Array,
    distance_confidence_scale: float | None,
    min_confidence: float,
) -> Array:
    distances = np.asarray(distances, dtype=np.float64)
    if distance_confidence_scale is None:
        confidence = np.ones_like(distances)
    else:
        scale = max(float(distance_confidence_scale), 1e-12)
        confidence = np.exp(-((distances / scale) ** 2))
    return np.clip(confidence, float(min_confidence), 1.0)

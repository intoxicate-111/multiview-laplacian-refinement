from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mlr.data import Mesh
from mlr.laplacian import unique_edges


@dataclass(frozen=True)
class CoarsePerturbationConfig:
    enabled: bool = True
    seed: int = 20260806
    distribution: str = "gaussian"
    use_local_edge_scale: bool = True
    normal_std_h: float = 0.10
    tangent_std_h: float = 0.03
    max_offset_h: float = 0.25
    smoothing_steps: int = 5
    smoothing_alpha: float = 0.5
    boundary_scale: float = 0.5
    topology_safe_altitude_ratio: float | None = None
    preserve_centroid: bool = True
    remove_global_translation: bool = True

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "CoarsePerturbationConfig":
        return cls(**({} if values is None else dict(values)))

    def validate(self) -> None:
        if self.distribution != "gaussian":
            raise ValueError("Only gaussian coarse perturbations are supported.")
        if min(self.normal_std_h, self.tangent_std_h, self.max_offset_h) < 0:
            raise ValueError("Perturbation magnitudes must be non-negative.")
        if self.smoothing_steps < 0:
            raise ValueError("smoothing_steps must be non-negative.")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must lie in [0, 1].")
        if not 0.0 <= self.boundary_scale <= 1.0:
            raise ValueError("boundary_scale must lie in [0, 1].")
        if (
            self.topology_safe_altitude_ratio is not None
            and self.topology_safe_altitude_ratio <= 0
        ):
            raise ValueError("topology_safe_altitude_ratio must be positive when set.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoarsePerturbationResult:
    mesh: Mesh
    displacement: np.ndarray
    local_edge_length: np.ndarray
    boundary_mask: np.ndarray
    metadata: dict[str, Any]


def perturb_coarse_mesh(
    coarse_mesh: Mesh,
    config: CoarsePerturbationConfig | Mapping[str, Any],
) -> CoarsePerturbationResult:
    """Apply one deterministic, graph-smoothed displacement to a coarse mesh.

    This routine deliberately receives no GT mesh, correspondence, or GT normal.
    It uses only the coarse vertices/faces and their derived normals/edge lengths.
    """

    if not isinstance(config, CoarsePerturbationConfig):
        config = CoarsePerturbationConfig.from_mapping(config)
    config.validate()
    coarse = Mesh(coarse_mesh.vertices.copy(), coarse_mesh.faces.copy()).ensure_normals()
    h = local_mean_edge_length(coarse)
    boundary = boundary_vertex_mask(coarse.faces, coarse.num_vertices)
    identity = (
        not config.enabled
        or config.max_offset_h == 0.0
        or (config.normal_std_h == 0.0 and config.tangent_std_h == 0.0)
    )
    if identity:
        displacement = np.zeros_like(coarse.vertices)
    else:
        rng = np.random.default_rng(config.seed)
        field = rng.normal(size=coarse.vertices.shape)
        field = graph_smooth_vectors(
            field,
            coarse.faces,
            steps=config.smoothing_steps,
            alpha=config.smoothing_alpha,
        )
        if config.remove_global_translation:
            field = field - field.mean(axis=0, keepdims=True)
        normal_scalar = np.einsum("ij,ij->i", field, coarse.normals)
        normal = normal_scalar[:, None] * coarse.normals
        tangent = field - normal
        normal_unit = _normalize_rows_allow_zero(normal)
        tangent_unit = _normalize_rows_allow_zero(tangent)
        scale = h if config.use_local_edge_scale else np.ones_like(h)
        displacement = (
            config.normal_std_h * scale[:, None] * normal_unit
            + config.tangent_std_h * scale[:, None] * tangent_unit
        )
        displacement[boundary] *= config.boundary_scale
        if config.topology_safe_altitude_ratio is not None:
            altitude_cap = (
                config.topology_safe_altitude_ratio
                * minimum_incident_triangle_altitude(coarse)
            )
            magnitude = np.linalg.norm(displacement, axis=1)
            scale_to_cap = np.minimum(
                1.0, altitude_cap / np.maximum(magnitude, 1e-15)
            )
            displacement *= scale_to_cap[:, None]
        # A mean-free field preserves the mesh centroid.  A single global clamp
        # then preserves that zero mean while enforcing every local h_i bound.
        if config.preserve_centroid:
            displacement -= displacement.mean(axis=0, keepdims=True)
        cap = config.max_offset_h * h
        magnitude = np.linalg.norm(displacement, axis=1)
        active = magnitude > 1e-15
        global_ratio = 1.0
        if np.any(active):
            global_ratio = float(
                min(1.0, np.min(cap[active] / np.maximum(magnitude[active], 1e-15)))
            )
        displacement *= global_ratio
    perturbed = Mesh(
        coarse.vertices + displacement, coarse.faces.copy()
    ).ensure_normals()
    magnitude = np.linalg.norm(displacement, axis=1)
    cap = config.max_offset_h * h
    if np.any(magnitude > cap + 1e-12):
        raise AssertionError("Coarse perturbation exceeded max_offset_h * h_i.")
    if config.preserve_centroid and not np.allclose(
        perturbed.vertices.mean(axis=0), coarse.vertices.mean(axis=0), atol=2e-12
    ):
        raise AssertionError("Coarse perturbation failed centroid preservation.")
    return CoarsePerturbationResult(
        mesh=perturbed,
        displacement=displacement,
        local_edge_length=h,
        boundary_mask=boundary,
        metadata={
            "config": config.as_dict(),
            "applied_once": bool(not identity),
            "geometry_inputs": "coarse_vertices_faces_normals_and_local_edge_lengths_only",
            "gt_used": False,
            "centroid_shift": float(
                np.linalg.norm(
                    perturbed.vertices.mean(axis=0) - coarse.vertices.mean(axis=0)
                )
            ),
            "global_clamp_ratio": float(global_ratio if not identity else 1.0),
            "boundary_vertices": int(boundary.sum()),
            "displacement_mean": float(magnitude.mean()),
            "displacement_median": float(np.median(magnitude)),
            "displacement_max": float(magnitude.max(initial=0.0)),
        },
    )


def graph_smooth_vectors(
    vectors: np.ndarray,
    faces: np.ndarray,
    *,
    steps: int,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vectors must have shape [N, 3].")
    edges = unique_edges(np.asarray(faces, dtype=np.int64))
    for _ in range(int(steps)):
        neighbour_sum = np.zeros_like(values)
        degree = np.zeros(len(values), dtype=np.float64)
        if len(edges):
            np.add.at(neighbour_sum, edges[:, 0], values[edges[:, 1]])
            np.add.at(neighbour_sum, edges[:, 1], values[edges[:, 0]])
            np.add.at(degree, edges[:, 0], 1.0)
            np.add.at(degree, edges[:, 1], 1.0)
        mean = values.copy()
        valid = degree > 0
        mean[valid] = neighbour_sum[valid] / degree[valid, None]
        values = (1.0 - alpha) * values + alpha * mean
    return values


def local_mean_edge_length(mesh: Mesh) -> np.ndarray:
    edges = unique_edges(mesh.faces)
    totals = np.zeros(mesh.num_vertices, dtype=np.float64)
    degree = np.zeros(mesh.num_vertices, dtype=np.float64)
    if len(edges):
        lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        np.add.at(totals, edges[:, 0], lengths)
        np.add.at(totals, edges[:, 1], lengths)
        np.add.at(degree, edges[:, 0], 1.0)
        np.add.at(degree, edges[:, 1], 1.0)
    result = np.zeros(mesh.num_vertices, dtype=np.float64)
    valid = degree > 0
    result[valid] = totals[valid] / degree[valid]
    return result


def boundary_vertex_mask(faces: np.ndarray, num_vertices: int) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            counts[key] = counts.get(key, 0) + 1
    result = np.zeros(int(num_vertices), dtype=bool)
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if boundary_edges:
        result[np.unique(np.asarray(boundary_edges, dtype=np.int64))] = True
    return result


def minimum_incident_triangle_altitude(mesh: Mesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    doubled_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    opposite_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
        ),
        axis=1,
    )
    altitudes = doubled_area[:, None] / np.maximum(opposite_lengths, 1e-15)
    result = np.full(mesh.num_vertices, np.inf, dtype=np.float64)
    for corner in range(3):
        np.minimum.at(result, mesh.faces[:, corner], altitudes[:, corner])
    result[~np.isfinite(result)] = 0.0
    return result


def expand_perturbed_coarse(
    perturbed_coarse: Mesh,
    control_expanded: Mesh,
    subdivision_mapping_path: str | Path,
) -> Mesh:
    """Re-run the saved midpoint expansion with perturbed coarse positions."""

    mapping_path = Path(subdivision_mapping_path)
    with np.load(mapping_path, allow_pickle=False) as mapping:
        parent_edges = np.asarray(mapping["parent_edges"], dtype=np.int64)
        new_indices = np.asarray(mapping["new_vertex_indices"], dtype=np.int64)
        final_to_pre = np.asarray(mapping["final_to_pre_compaction"], dtype=np.int64)
    if parent_edges.shape != (len(new_indices), 2):
        raise ValueError("Invalid subdivision parent-edge mapping.")
    if np.any(parent_edges < 0) or np.any(parent_edges >= perturbed_coarse.num_vertices):
        raise ValueError("Subdivision mapping references a non-coarse vertex.")
    pre_count = int(max(new_indices.max(initial=-1), perturbed_coarse.num_vertices - 1) + 1)
    pre = np.full((pre_count, 3), np.nan, dtype=np.float64)
    pre[: perturbed_coarse.num_vertices] = perturbed_coarse.vertices
    pre[new_indices] = 0.5 * (
        perturbed_coarse.vertices[parent_edges[:, 0]]
        + perturbed_coarse.vertices[parent_edges[:, 1]]
    )
    if not np.isfinite(pre).all():
        raise ValueError("Subdivision mapping left unassigned pre-compaction vertices.")
    expanded_vertices = pre[final_to_pre]
    if expanded_vertices.shape != control_expanded.vertices.shape:
        raise ValueError("Perturbed expansion changed the expanded vertex count.")
    return Mesh(expanded_vertices, control_expanded.faces.copy()).ensure_normals()


def apply_delta_scale(raw_prediction: np.ndarray, scale: float) -> np.ndarray:
    prediction = np.asarray(raw_prediction)
    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("raw_prediction must have shape [N, 3].")
    result = float(scale) * prediction
    if scale == 0.0:
        result = np.zeros_like(prediction)
    return result


def _normalize_rows_allow_zero(values: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(values, axis=1, keepdims=True)
    result = np.zeros_like(values)
    valid = magnitude[:, 0] > 1e-15
    result[valid] = values[valid] / magnitude[valid]
    return result

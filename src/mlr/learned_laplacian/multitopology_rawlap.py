from __future__ import annotations

"""Geometry primitives for the Sofa50 clean-topology raw-Laplacian dataset.

The functions in this module are deliberately independent of image rendering and
dataset I/O.  This keeps the supervision contract directly testable: construct a
clean topology, compute its native uniform Laplacian, then corrupt only a copy of
the clean vertex positions.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Mesh
from mlr.learned_laplacian.coarse_perturbation import (
    CoarsePerturbationConfig,
    perturb_coarse_mesh,
)


VARIANT_NAMES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")
UNSEEN_VARIANT_NAMES = ("U1", "U2", "U3", "U4", "U5")
LEGACY_SMOOTHING_PROFILE = "legacy_v1"
STRONG_SMOOTHING_PROFILE = "strong_smooth_v2"
DEFAULT_SMOOTHING_PROFILE = STRONG_SMOOTHING_PROFILE


@dataclass(frozen=True)
class DegradationConfig:
    name: str
    normal_std_h: float
    tangent_std_h: float
    max_offset_h: float
    field_smoothing_steps: int
    field_smoothing_alpha: float
    topology_safe_altitude_ratio: float
    mesh_smoothing_iterations: int
    mesh_smoothing_strength: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LEGACY_MILD_DEGRADATION = DegradationConfig(
    name="mild",
    normal_std_h=0.06,
    tangent_std_h=0.02,
    max_offset_h=0.12,
    field_smoothing_steps=3,
    field_smoothing_alpha=0.5,
    topology_safe_altitude_ratio=0.25,
    mesh_smoothing_iterations=2,
    mesh_smoothing_strength=0.08,
)


LEGACY_STRONG_DEGRADATION = DegradationConfig(
    name="strong",
    normal_std_h=0.12,
    tangent_std_h=0.04,
    max_offset_h=0.24,
    field_smoothing_steps=5,
    field_smoothing_alpha=0.5,
    topology_safe_altitude_ratio=0.35,
    mesh_smoothing_iterations=4,
    mesh_smoothing_strength=0.12,
)


LEGACY_UNSEEN_DEGRADATION = DegradationConfig(
    name="unseen_intermediate",
    normal_std_h=0.09,
    tangent_std_h=0.03,
    max_offset_h=0.18,
    field_smoothing_steps=4,
    field_smoothing_alpha=0.5,
    topology_safe_altitude_ratio=0.30,
    mesh_smoothing_iterations=3,
    mesh_smoothing_strength=0.10,
)


# The v1 smoothing pass left the generated current/coarse meshes visually too
# close to their clean references.  v2 deliberately changes only the final
# mesh-smoothing budget.  Perturbation magnitudes, topology construction,
# random seeds and raw-Laplacian targets remain unchanged.
MILD_DEGRADATION = DegradationConfig(
    name="mild",
    normal_std_h=LEGACY_MILD_DEGRADATION.normal_std_h,
    tangent_std_h=LEGACY_MILD_DEGRADATION.tangent_std_h,
    max_offset_h=LEGACY_MILD_DEGRADATION.max_offset_h,
    field_smoothing_steps=LEGACY_MILD_DEGRADATION.field_smoothing_steps,
    field_smoothing_alpha=LEGACY_MILD_DEGRADATION.field_smoothing_alpha,
    topology_safe_altitude_ratio=LEGACY_MILD_DEGRADATION.topology_safe_altitude_ratio,
    mesh_smoothing_iterations=6,
    mesh_smoothing_strength=0.12,
)


STRONG_DEGRADATION = DegradationConfig(
    name="strong",
    normal_std_h=LEGACY_STRONG_DEGRADATION.normal_std_h,
    tangent_std_h=LEGACY_STRONG_DEGRADATION.tangent_std_h,
    max_offset_h=LEGACY_STRONG_DEGRADATION.max_offset_h,
    field_smoothing_steps=LEGACY_STRONG_DEGRADATION.field_smoothing_steps,
    field_smoothing_alpha=LEGACY_STRONG_DEGRADATION.field_smoothing_alpha,
    topology_safe_altitude_ratio=LEGACY_STRONG_DEGRADATION.topology_safe_altitude_ratio,
    mesh_smoothing_iterations=10,
    mesh_smoothing_strength=0.15,
)


UNSEEN_DEGRADATION = DegradationConfig(
    name="unseen_intermediate",
    normal_std_h=LEGACY_UNSEEN_DEGRADATION.normal_std_h,
    tangent_std_h=LEGACY_UNSEEN_DEGRADATION.tangent_std_h,
    max_offset_h=LEGACY_UNSEEN_DEGRADATION.max_offset_h,
    field_smoothing_steps=LEGACY_UNSEEN_DEGRADATION.field_smoothing_steps,
    field_smoothing_alpha=LEGACY_UNSEEN_DEGRADATION.field_smoothing_alpha,
    topology_safe_altitude_ratio=LEGACY_UNSEEN_DEGRADATION.topology_safe_altitude_ratio,
    mesh_smoothing_iterations=8,
    mesh_smoothing_strength=0.135,
)


DEGRADATION_PROFILES: dict[str, dict[str, DegradationConfig]] = {
    LEGACY_SMOOTHING_PROFILE: {
        "mild": LEGACY_MILD_DEGRADATION,
        "strong": LEGACY_STRONG_DEGRADATION,
        "unseen": LEGACY_UNSEEN_DEGRADATION,
    },
    STRONG_SMOOTHING_PROFILE: {
        "mild": MILD_DEGRADATION,
        "strong": STRONG_DEGRADATION,
        "unseen": UNSEEN_DEGRADATION,
    },
}


def smoothing_high_frequency_attenuation(iterations: int, strength: float) -> float:
    """Return a simple repeated-pass attenuation proxy for profile auditing."""

    if iterations < 0 or not 0.0 <= strength <= 1.0:
        raise ValueError("Invalid Laplacian smoothing settings.")
    return float(1.0 - (1.0 - float(strength)) ** int(iterations))


# Percentiles are computed per object after area / bbox_diagonal**2 and edge /
# bbox_diagonal normalization.  This makes every threshold scale independent.
TOPOLOGY_RECIPES: dict[str, dict[str, Any]] = {
    "A1": {"family": "original", "degradation": "mild"},
    "A2": {"family": "original", "degradation": "strong"},
    "B1": {"family": "uniform_midpoint", "levels": 1, "degradation": "mild"},
    "B2": {"family": "uniform_midpoint", "levels": 1, "degradation": "strong"},
    "C1": {"family": "area", "area_quantile": 0.25, "degradation": "mild"},
    "C2": {"family": "area", "area_quantile": 0.50, "degradation": "strong"},
    "C3": {"family": "area", "area_quantile": 0.75, "degradation": "mild"},
    "C4": {"family": "area", "area_quantile": 0.90, "degradation": "strong"},
    "D1": {
        "family": "area_or_edge",
        "area_quantile": 0.75,
        "edge_quantile": 0.50,
        "degradation": "mild",
    },
    "D2": {
        "family": "area_or_edge",
        "area_quantile": 0.90,
        "edge_quantile": 0.80,
        "degradation": "strong",
    },
    # Evaluation-only recipes. None is present in the 500-sample training set.
    "U1": {"family": "area", "area_quantile": 0.625, "degradation": "unseen"},
    "U2": {"family": "area", "area_quantile": 0.97, "degradation": "unseen"},
    "U3": {
        "family": "area_or_edge",
        "area_quantile": 0.85,
        "edge_quantile": 0.70,
        "degradation": "unseen",
    },
    "U4": {"family": "original", "degradation": "unseen"},
    "U5": {"family": "uniform_midpoint", "levels": 2, "degradation": "unseen"},
}


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def face_max_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    return lengths.max(axis=1)


def bbox_diagonal(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    value = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not np.isfinite(value) or value <= 1e-12:
        raise ValueError("Mesh bounding-box diagonal must be finite and positive.")
    return value


def unique_sorted_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def split_marked_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    marked_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformingly split marked edges without changing the represented surface."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    marked_edges = np.asarray(marked_edges, dtype=np.int64)
    if len(marked_edges) == 0:
        return vertices.copy(), faces.copy()
    marked_edges = np.unique(np.sort(marked_edges, axis=1), axis=0)
    if marked_edges.min() < 0 or marked_edges.max() >= len(vertices):
        raise ValueError("Marked edge index is outside the vertex array.")

    new_vertices = np.concatenate(
        (vertices, 0.5 * (vertices[marked_edges[:, 0]] + vertices[marked_edges[:, 1]])),
        axis=0,
    )
    edge_to_midpoint = {
        (int(edge[0]), int(edge[1])): len(vertices) + index
        for index, edge in enumerate(marked_edges)
    }
    output: list[tuple[int, int, int]] = []
    for a_value, b_value, c_value in faces:
        a, b, c = int(a_value), int(b_value), int(c_value)
        ab = edge_to_midpoint.get((min(a, b), max(a, b)))
        bc = edge_to_midpoint.get((min(b, c), max(b, c)))
        ca = edge_to_midpoint.get((min(c, a), max(c, a)))
        code = (ab is not None) + 2 * (bc is not None) + 4 * (ca is not None)
        if code == 0:
            output.append((a, b, c))
        elif code == 1:
            output.extend(((a, ab, c), (ab, b, c)))
        elif code == 2:
            output.extend(((b, bc, a), (bc, c, a)))
        elif code == 4:
            output.extend(((c, ca, b), (ca, a, b)))
        elif code == 3:
            output.extend(((a, ab, c), (ab, bc, c), (ab, b, bc)))
        elif code == 6:
            output.extend(((b, bc, a), (bc, ca, a), (bc, c, ca)))
        elif code == 5:
            output.extend(((c, ca, b), (ca, ab, b), (ca, a, ab)))
        else:
            output.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return new_vertices, np.asarray(output, dtype=np.int64)


def uniform_midpoint_subdivide(
    vertices: np.ndarray, faces: np.ndarray, *, levels: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    output_vertices = np.asarray(vertices, dtype=np.float64).copy()
    output_faces = np.asarray(faces, dtype=np.int64).copy()
    for _ in range(int(levels)):
        output_vertices, output_faces = split_marked_edges(
            output_vertices, output_faces, unique_sorted_edges(output_faces)
        )
    return output_vertices, output_faces


def adaptive_subdivide(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tau_area: float,
    tau_edge: float | None = None,
    max_iters: int = 8,
    max_vertices: int = 500_000,
    require_convergence: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not np.isfinite(tau_area) or tau_area <= 0:
        raise ValueError("tau_area must be finite and positive.")
    if tau_edge is not None and (not np.isfinite(tau_edge) or tau_edge <= 0):
        raise ValueError("tau_edge must be finite and positive when supplied.")
    current_vertices = np.asarray(vertices, dtype=np.float64).copy()
    current_faces = np.asarray(faces, dtype=np.int64).copy()
    history: list[dict[str, Any]] = []
    for iteration in range(max_iters + 1):
        areas = triangle_areas(current_vertices, current_faces)
        area_mask = areas > tau_area * (1.0 + 1e-12)
        if tau_edge is None:
            edge_mask = np.zeros(len(current_faces), dtype=bool)
        else:
            edge_mask = face_max_edge_lengths(current_vertices, current_faces) > (
                tau_edge * (1.0 + 1e-12)
            )
        selected = area_mask | edge_mask
        history.append(
            {
                "iteration": int(iteration),
                "vertices": int(len(current_vertices)),
                "faces": int(len(current_faces)),
                "area_selected_faces": int(area_mask.sum()),
                "edge_selected_faces": int(edge_mask.sum()),
                "edge_only_selected_faces": int((edge_mask & ~area_mask).sum()),
                "selected_faces": int(selected.sum()),
                "max_area": float(areas.max(initial=0.0)),
                "max_edge": float(
                    face_max_edge_lengths(current_vertices, current_faces).max(initial=0.0)
                ),
            }
        )
        if not np.any(selected):
            return current_vertices, current_faces, history
        if iteration == max_iters:
            if require_convergence:
                raise RuntimeError("Adaptive subdivision did not converge within max_iters.")
            return current_vertices, current_faces, history
        marked = unique_sorted_edges(current_faces[selected])
        if len(current_vertices) + len(marked) > max_vertices:
            raise RuntimeError(
                f"Adaptive subdivision would exceed max_vertices={max_vertices}."
            )
        current_vertices, current_faces = split_marked_edges(
            current_vertices, current_faces, marked
        )
    raise AssertionError("unreachable")


def construct_clean_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    variant: str,
    *,
    max_iters: int = 8,
    max_vertices: int = 500_000,
) -> tuple[Mesh, dict[str, Any]]:
    if variant not in TOPOLOGY_RECIPES:
        raise ValueError(f"Unknown topology variant: {variant}")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    recipe = dict(TOPOLOGY_RECIPES[variant])
    diagonal = bbox_diagonal(vertices)
    normalized_areas = triangle_areas(vertices, faces) / diagonal**2
    normalized_edges = face_max_edge_lengths(vertices, faces) / diagonal
    family = str(recipe["family"])
    history: list[dict[str, Any]] = []
    tau_area = None
    tau_edge = None
    if family == "original":
        clean_vertices, clean_faces = vertices.copy(), faces.copy()
    elif family == "uniform_midpoint":
        clean_vertices, clean_faces = uniform_midpoint_subdivide(
            vertices, faces, levels=int(recipe["levels"])
        )
    else:
        tau_area_normalized = float(np.quantile(normalized_areas, recipe["area_quantile"]))
        tau_area = tau_area_normalized * diagonal**2
        if family == "area_or_edge":
            tau_edge_normalized = float(np.quantile(normalized_edges, recipe["edge_quantile"]))
            tau_edge = tau_edge_normalized * diagonal
        clean_vertices, clean_faces, history = adaptive_subdivide(
            vertices,
            faces,
            tau_area=tau_area,
            tau_edge=tau_edge,
            # One threshold-selection/subdivision round is intentional.  The
            # threshold is evaluated on the original GT topology; recursively
            # enforcing a low original-area quantile can grow a mesh without a
            # useful upper bound and is not the experiment requested here.
            max_iters=1,
            max_vertices=max_vertices,
            require_convergence=False,
        )
    clean = Mesh(clean_vertices, clean_faces).ensure_normals()
    metadata = {
        "variant": variant,
        "family": family,
        "recipe": recipe,
        "scale_definition": "bbox_diagonal",
        "bbox_diagonal": diagonal,
        "tau_area": tau_area,
        "tau_area_normalized": None if tau_area is None else tau_area / diagonal**2,
        "tau_edge": tau_edge,
        "tau_edge_normalized": None if tau_edge is None else tau_edge / diagonal,
        "original_vertices": int(len(vertices)),
        "original_faces": int(len(faces)),
        "clean_vertices": clean.num_vertices,
        "clean_faces": clean.num_faces,
        "vertex_ratio_vs_gt": clean.num_vertices / len(vertices),
        "face_ratio_vs_gt": clean.num_faces / len(faces),
        "adaptive_history": history,
        "adaptive_subdivision_rounds": 0 if not history else len(history) - 1,
        "adaptive_threshold_evaluation": "one round on original GT topology",
        "edge_only_selected_faces_total": int(
            sum(row["edge_only_selected_faces"] for row in history[:-1])
        ),
    }
    return clean, metadata


def raw_uniform_laplacian(mesh: Mesh) -> np.ndarray:
    operator = build_uniform_laplacian_data(mesh.faces, mesh.num_vertices)
    return apply_uniform_laplacian(mesh.vertices, operator)


def laplacian_smooth_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    strength: float,
    preserve_centroid: bool = True,
) -> np.ndarray:
    if iterations < 0 or not 0.0 <= strength <= 1.0:
        raise ValueError("Invalid Laplacian smoothing settings.")
    output = np.asarray(vertices, dtype=np.float64).copy()
    center = output.mean(axis=0, keepdims=True)
    operator = build_uniform_laplacian_data(faces, len(output))
    for _ in range(iterations):
        output -= strength * apply_uniform_laplacian(output, operator)
    if preserve_centroid:
        output += center - output.mean(axis=0, keepdims=True)
    return output


def corrupt_clean_reference(
    clean: Mesh,
    degradation: DegradationConfig,
    *,
    seed: int,
) -> tuple[Mesh, dict[str, Any]]:
    perturbation = perturb_coarse_mesh(
        clean,
        CoarsePerturbationConfig(
            seed=int(seed),
            normal_std_h=degradation.normal_std_h,
            tangent_std_h=degradation.tangent_std_h,
            max_offset_h=degradation.max_offset_h,
            smoothing_steps=degradation.field_smoothing_steps,
            smoothing_alpha=degradation.field_smoothing_alpha,
            topology_safe_altitude_ratio=degradation.topology_safe_altitude_ratio,
        ),
    )
    smoothed_vertices = laplacian_smooth_positions(
        perturbation.mesh.vertices,
        clean.faces,
        iterations=degradation.mesh_smoothing_iterations,
        strength=degradation.mesh_smoothing_strength,
    )
    corrupted = Mesh(smoothed_vertices, clean.faces.copy()).ensure_normals()
    displacement = np.linalg.norm(corrupted.vertices - clean.vertices, axis=1)
    smoothing_displacement = np.linalg.norm(
        corrupted.vertices - perturbation.mesh.vertices, axis=1
    )
    diagonal = bbox_diagonal(clean.vertices)
    if not np.isfinite(corrupted.vertices).all():
        raise RuntimeError("Corrupted mesh contains NaN or infinite positions.")
    metadata = {
        "order": "clean_reference -> perturb -> smooth -> input_mesh",
        "seed": int(seed),
        "degradation": degradation.as_dict(),
        "perturbation": perturbation.metadata,
        "mesh_smoothing_high_frequency_attenuation_proxy": (
            smoothing_high_frequency_attenuation(
                degradation.mesh_smoothing_iterations,
                degradation.mesh_smoothing_strength,
            )
        ),
        "clean_to_input_displacement_mean": float(displacement.mean()),
        "clean_to_input_displacement_median": float(np.median(displacement)),
        "clean_to_input_displacement_max": float(displacement.max(initial=0.0)),
        "clean_to_input_displacement_p95": float(np.quantile(displacement, 0.95)),
        "clean_to_input_displacement_mean_over_bbox_diagonal": float(
            displacement.mean() / diagonal
        ),
        "clean_to_input_displacement_p95_over_bbox_diagonal": float(
            np.quantile(displacement, 0.95) / diagonal
        ),
        "perturb_to_input_smoothing_displacement_mean": float(
            smoothing_displacement.mean()
        ),
        "perturb_to_input_smoothing_displacement_mean_over_bbox_diagonal": float(
            smoothing_displacement.mean() / diagonal
        ),
    }
    return corrupted, metadata


def degradation_for_variant(
    variant: str,
    smoothing_profile: str = DEFAULT_SMOOTHING_PROFILE,
) -> DegradationConfig:
    if smoothing_profile not in DEGRADATION_PROFILES:
        raise ValueError(f"Unknown smoothing profile: {smoothing_profile}")
    name = str(TOPOLOGY_RECIPES[variant]["degradation"])
    profile = DEGRADATION_PROFILES[smoothing_profile]
    if name not in profile:
        raise ValueError(f"Unknown degradation regime: {name}")
    return profile[name]

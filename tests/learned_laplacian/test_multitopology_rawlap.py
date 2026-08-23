from __future__ import annotations

import numpy as np

from mlr.data import Mesh
from mlr.learned_laplacian.multitopology_rawlap import (
    LEGACY_SMOOTHING_PROFILE,
    STRONG_SMOOTHING_PROFILE,
    DegradationConfig,
    adaptive_subdivide,
    construct_clean_topology,
    corrupt_clean_reference,
    degradation_for_variant,
    raw_uniform_laplacian,
    smoothing_high_frequency_attenuation,
    triangle_areas,
    uniform_midpoint_subdivide,
)


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    return vertices, faces


def irregular_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((0, 0, 0), (4, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.2, 0.2)),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2), (0, 4, 2)),
        dtype=np.int64,
    )
    return vertices, faces


def test_uniform_midpoint_is_deterministic_and_preserves_surface() -> None:
    vertices, faces = tetrahedron()
    first_v, first_f = uniform_midpoint_subdivide(vertices, faces)
    second_v, second_f = uniform_midpoint_subdivide(vertices, faces)
    assert np.array_equal(first_v, second_v)
    assert np.array_equal(first_f, second_f)
    assert len(first_f) == 4 * len(faces)
    assert np.isclose(triangle_areas(first_v, first_f).sum(), triangle_areas(vertices, faces).sum())


def test_adaptive_area_or_edge_reports_edge_only_selections() -> None:
    vertices, faces = irregular_mesh()
    area = triangle_areas(vertices, faces)
    output_v, output_f, history = adaptive_subdivide(
        vertices,
        faces,
        tau_area=float(np.quantile(area, 0.9)),
        tau_edge=1.0,
        max_iters=8,
    )
    assert len(output_v) > len(vertices)
    assert len(output_f) > len(faces)
    assert sum(row["edge_only_selected_faces"] for row in history[:-1]) > 0
    assert history[-1]["selected_faces"] == 0


def test_variant_thresholds_are_scale_invariant() -> None:
    vertices, faces = irregular_mesh()
    clean_a, metadata_a = construct_clean_topology(vertices, faces, "C2")
    clean_b, metadata_b = construct_clean_topology(vertices * 13.0, faces, "C2")
    assert np.array_equal(clean_a.faces, clean_b.faces)
    assert np.allclose(clean_a.vertices * 13.0, clean_b.vertices)
    assert np.isclose(metadata_a["tau_area_normalized"], metadata_b["tau_area_normalized"])


def test_target_is_native_to_clean_topology_and_corruption_is_topology_preserving() -> None:
    vertices, faces = tetrahedron()
    clean_vertices, clean_faces = uniform_midpoint_subdivide(vertices, faces)
    clean = Mesh(clean_vertices, clean_faces).ensure_normals()
    target = raw_uniform_laplacian(clean)
    degradation = DegradationConfig(
        name="test",
        normal_std_h=0.04,
        tangent_std_h=0.01,
        max_offset_h=0.08,
        field_smoothing_steps=2,
        field_smoothing_alpha=0.5,
        topology_safe_altitude_ratio=0.2,
        mesh_smoothing_iterations=1,
        mesh_smoothing_strength=0.05,
    )
    corrupted, metadata = corrupt_clean_reference(clean, degradation, seed=123)
    assert np.array_equal(clean.faces, corrupted.faces)
    assert clean.num_vertices == corrupted.num_vertices
    assert target.shape == clean.vertices.shape
    assert np.array_equal(target, raw_uniform_laplacian(clean))
    assert not np.allclose(target, raw_uniform_laplacian(corrupted))
    assert metadata["order"] == "clean_reference -> perturb -> smooth -> input_mesh"
    assert metadata["clean_to_input_displacement_mean_over_bbox_diagonal"] > 0.0
    assert metadata["perturb_to_input_smoothing_displacement_mean_over_bbox_diagonal"] > 0.0


def test_v2_profile_only_increases_mesh_smoothing_budget() -> None:
    for variant in ("A1", "A2", "U1"):
        legacy = degradation_for_variant(variant, LEGACY_SMOOTHING_PROFILE)
        stronger = degradation_for_variant(variant, STRONG_SMOOTHING_PROFILE)
        assert (
            legacy.normal_std_h,
            legacy.tangent_std_h,
            legacy.max_offset_h,
            legacy.field_smoothing_steps,
            legacy.field_smoothing_alpha,
            legacy.topology_safe_altitude_ratio,
        ) == (
            stronger.normal_std_h,
            stronger.tangent_std_h,
            stronger.max_offset_h,
            stronger.field_smoothing_steps,
            stronger.field_smoothing_alpha,
            stronger.topology_safe_altitude_ratio,
        )
        legacy_attenuation = smoothing_high_frequency_attenuation(
            legacy.mesh_smoothing_iterations, legacy.mesh_smoothing_strength
        )
        stronger_attenuation = smoothing_high_frequency_attenuation(
            stronger.mesh_smoothing_iterations, stronger.mesh_smoothing_strength
        )
        assert stronger.mesh_smoothing_iterations > legacy.mesh_smoothing_iterations
        assert stronger.mesh_smoothing_strength > legacy.mesh_smoothing_strength
        assert stronger_attenuation > legacy_attenuation


def test_unseen_recipes_are_not_training_recipe_duplicates() -> None:
    vertices, faces = irregular_mesh()
    u1, u1_meta = construct_clean_topology(vertices, faces, "U1")
    u2, u2_meta = construct_clean_topology(vertices, faces, "U2")
    u5, u5_meta = construct_clean_topology(vertices, faces, "U5")
    assert u1_meta["recipe"]["area_quantile"] == 0.625
    assert u2_meta["recipe"]["area_quantile"] == 0.97
    assert u1_meta["tau_area_normalized"] < u2_meta["tau_area_normalized"]
    assert u1.num_faces >= u2.num_faces
    assert u5_meta["recipe"]["levels"] == 2
    assert u5.num_faces == 16 * len(faces)

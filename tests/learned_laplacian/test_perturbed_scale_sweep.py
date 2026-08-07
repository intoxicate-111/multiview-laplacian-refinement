from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mlr.data import Mesh
from mlr.learned_laplacian.coarse_perturbation import (
    CoarsePerturbationConfig,
    apply_delta_scale,
    expand_perturbed_coarse,
    perturb_coarse_mesh,
)
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.perturbed_scale_sweep import (
    PANEL_SIZE,
    _contact_sheet,
    _render_panel,
    fixed_visualization_cameras,
    normalize_scales,
    scale_sweep_jobs,
    scale_token,
    validate_variant_visibility_contract,
)
from mlr.learned_laplacian.target_scaling import incident_edge_length_and_valid_mask


def grid_mesh() -> Mesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.15],
        ]
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return Mesh(vertices, faces).ensure_normals()


def default_config(**overrides) -> CoarsePerturbationConfig:
    values = CoarsePerturbationConfig().__dict__ | overrides
    return CoarsePerturbationConfig(**values)


def midpoint_expansion(mesh: Mesh, path: Path) -> Mesh:
    edges = np.array(
        sorted(
            {
                tuple(sorted((int(a), int(b))))
                for face in mesh.faces
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            }
        ),
        dtype=np.int64,
    )
    new_indices = np.arange(mesh.num_vertices, mesh.num_vertices + len(edges))
    edge_to_new = {tuple(edge): int(index) for edge, index in zip(edges, new_indices)}
    faces = []
    for a, b, c in mesh.faces:
        ab = edge_to_new[tuple(sorted((int(a), int(b))))]
        bc = edge_to_new[tuple(sorted((int(b), int(c))))]
        ca = edge_to_new[tuple(sorted((int(c), int(a))))]
        faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    vertices = np.concatenate(
        (mesh.vertices, 0.5 * (mesh.vertices[edges[:, 0]] + mesh.vertices[edges[:, 1]]))
    )
    np.savez(
        path,
        parent_edges=edges,
        new_vertex_indices=new_indices,
        pre_compaction_to_final=np.arange(len(vertices)),
        final_to_pre_compaction=np.arange(len(vertices)),
    )
    return Mesh(vertices, np.asarray(faces)).ensure_normals()


def test_disabled_perturbation_is_exact_identity() -> None:
    mesh = grid_mesh()
    result = perturb_coarse_mesh(mesh, default_config(enabled=False))
    np.testing.assert_array_equal(result.mesh.vertices, mesh.vertices)
    np.testing.assert_array_equal(result.mesh.faces, mesh.faces)


def test_same_seed_is_reproducible_and_different_seed_changes_vertices() -> None:
    mesh = grid_mesh()
    first = perturb_coarse_mesh(mesh, default_config(seed=13)).mesh.vertices
    second = perturb_coarse_mesh(mesh, default_config(seed=13)).mesh.vertices
    other = perturb_coarse_mesh(mesh, default_config(seed=14)).mesh.vertices
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_perturbation_preserves_counts_connectivity_order_and_centroid() -> None:
    mesh = grid_mesh()
    result = perturb_coarse_mesh(mesh, default_config())
    assert result.mesh.num_vertices == mesh.num_vertices
    assert result.mesh.num_faces == mesh.num_faces
    np.testing.assert_array_equal(result.mesh.faces, mesh.faces)
    np.testing.assert_allclose(result.mesh.vertices.mean(0), mesh.vertices.mean(0), atol=2e-12)


def test_max_displacement_obeys_local_h_bound() -> None:
    result = perturb_coarse_mesh(grid_mesh(), default_config(max_offset_h=0.08))
    magnitude = np.linalg.norm(result.displacement, axis=1)
    assert np.all(magnitude <= 0.08 * result.local_edge_length + 1e-12)


def test_zero_strength_is_identity() -> None:
    mesh = grid_mesh()
    result = perturb_coarse_mesh(
        mesh,
        default_config(normal_std_h=0.0, tangent_std_h=0.0),
    )
    np.testing.assert_array_equal(result.mesh.vertices, mesh.vertices)


def test_boundary_scale_changes_boundary_displacement() -> None:
    mesh = grid_mesh()
    full = perturb_coarse_mesh(mesh, default_config(boundary_scale=1.0))
    reduced = perturb_coarse_mesh(mesh, default_config(boundary_scale=0.0))
    boundary = full.boundary_mask
    assert boundary.any()
    assert not np.allclose(full.displacement[boundary], reduced.displacement[boundary])
    assert np.linalg.norm(reduced.displacement[boundary], axis=1).mean() < np.linalg.norm(
        full.displacement[boundary], axis=1
    ).mean()


def test_uniform_altitude_cap_prevents_face_flips() -> None:
    mesh = grid_mesh()
    safe = perturb_coarse_mesh(
        mesh, default_config(topology_safe_altitude_ratio=0.30)
    ).mesh
    original_triangles = mesh.vertices[mesh.faces]
    safe_triangles = safe.vertices[safe.faces]
    original_normals = np.cross(
        original_triangles[:, 1] - original_triangles[:, 0],
        original_triangles[:, 2] - original_triangles[:, 0],
    )
    safe_normals = np.cross(
        safe_triangles[:, 1] - safe_triangles[:, 0],
        safe_triangles[:, 2] - safe_triangles[:, 0],
    )
    assert np.all(np.einsum("ij,ij->i", original_normals, safe_normals) >= 0)


def test_expansion_uses_perturbed_coarse_once_and_keeps_topology(tmp_path: Path) -> None:
    coarse = grid_mesh()
    mapping = tmp_path / "mapping.npz"
    control = midpoint_expansion(coarse, mapping)
    perturbed = perturb_coarse_mesh(coarse, default_config()).mesh
    expanded = expand_perturbed_coarse(perturbed, control, mapping)
    np.testing.assert_allclose(expanded.vertices[: coarse.num_vertices], perturbed.vertices)
    np.testing.assert_array_equal(expanded.faces, control.faces)
    assert expanded.num_vertices == control.num_vertices
    # Re-expansion is deterministic; it does not apply random noise a second time.
    second = expand_perturbed_coarse(perturbed, control, mapping)
    np.testing.assert_array_equal(expanded.vertices, second.vertices)


def test_perturbation_api_cannot_receive_gt_geometry() -> None:
    parameters = inspect.signature(perturb_coarse_mesh).parameters
    assert set(parameters) == {"coarse_mesh", "config"}


@pytest.mark.parametrize(
    ("scale", "expected"),
    [(1.0, 1.0), (0.0, 0.0), (-1.0, -1.0), (-0.5, -0.5), (2.0, 2.0)],
)
def test_delta_scale_is_applied_exactly_once(scale: float, expected: float) -> None:
    prediction = np.arange(12, dtype=np.float64).reshape(4, 3) + 1.0
    scaled = apply_delta_scale(prediction, scale)
    np.testing.assert_array_equal(scaled, expected * prediction)


def test_scale_zero_is_strict_all_zero() -> None:
    scaled = apply_delta_scale(np.full((5, 3), np.nan_to_num(1.0)), 0.0)
    assert np.count_nonzero(scaled) == 0


def test_scale_sweep_requires_zero_and_one_and_stable_tokens() -> None:
    assert normalize_scales([-1, 0, 1]) == (-1.0, 0.0, 1.0)
    assert scale_token(-0.5) == "neg0p5"
    assert scale_token(0.125) == "0p125"
    with pytest.raises(ValueError, match="include"):
        normalize_scales([0.5, 1.0])


def test_cached_prediction_source_is_not_mutated_by_scales() -> None:
    prediction = np.arange(15, dtype=np.float64).reshape(5, 3)
    original = prediction.copy()
    outputs = [apply_delta_scale(prediction, scale) for scale in (-1, 0, 1, 2)]
    np.testing.assert_array_equal(prediction, original)
    assert all(not np.shares_memory(output, prediction) for output in outputs)


def test_all_scales_share_prediction_visibility_and_solver_config() -> None:
    prediction = np.ones((5, 3))
    visibility = np.ones(5)
    solver = {"num_iters": 2}
    jobs = scale_sweep_jobs(prediction, visibility, solver, [-1, 0, 1])
    assert all(job["raw_prediction"] is prediction for job in jobs)
    assert all(job["visibility_weight"] is visibility for job in jobs)
    assert all(job["solver_config"] is solver for job in jobs)


def test_control_and_perturbed_visibility_are_separate_and_shape_checked() -> None:
    control = np.ones((3, 5), dtype=bool)
    perturbed = control.copy()
    validate_variant_visibility_contract(control, perturbed, 5, 5)
    with pytest.raises(ValueError, match="reuse"):
        validate_variant_visibility_contract(control, control, 5, 5)
    with pytest.raises(ValueError, match="shape"):
        validate_variant_visibility_contract(control, perturbed[:, :-1], 5, 5)


def test_identity_placeholder_is_not_used_for_oracle_or_error(tmp_path: Path) -> None:
    mesh = grid_mesh()
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    local_h, valid = incident_edge_length_and_valid_mask(
        vertices, faces_to_edge_index(faces)
    )
    sample = {
        "vertices": vertices,
        "faces": faces,
        "laplacian_target": torch.full_like(vertices, 999.0),
        "raw_laplacian_target": torch.full_like(vertices, 999.0),
        "target_confidence": torch.ones(len(vertices)),
        "local_edge_length": local_h,
        "valid_scale_mask": valid,
    }
    metrics = reconstruct_and_evaluate(
        sample,
        torch.zeros_like(vertices),
        tmp_path,
        {"operator_type": "uniform", "num_iters": 1, "evaluate_oracle": False},
        evaluate_laplacian_prediction=False,
    )
    assert metrics["laplacian_prediction"] is None
    assert metrics["reconstruction"]["oracle_evaluated"] is False
    assert not (tmp_path / "delta_target.npy").exists()
    assert not (tmp_path / "laplacian_error.npy").exists()


def test_visualization_cameras_are_shared_and_panels_are_960(tmp_path: Path) -> None:
    mesh = grid_mesh()
    cameras, _ = fixed_visualization_cameras(mesh, mesh, mesh, PANEL_SIZE)
    assert set(cameras) == {"front", "side", "perspective"}
    panel = tmp_path / "panel.png"
    _render_panel(
        mesh,
        cameras["perspective"],
        panel,
        "mesh | Control | GT | perspective",
        (180, 200, 220),
        "cpu",
    )
    with Image.open(panel) as image:
        assert image.size == (960, 960)
        values = np.asarray(image)
        assert values.min() < values.max()


def test_composite_contact_sheet_is_nonempty_and_labelled(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (960, 960), (120, 160, 200))
    image.save(source)
    destination = tmp_path / "sheet.png"
    _contact_sheet([("correct label", source)], destination, columns=1)
    with Image.open(destination) as sheet:
        assert sheet.size == (480, 480)
        assert np.asarray(sheet).std() > 0

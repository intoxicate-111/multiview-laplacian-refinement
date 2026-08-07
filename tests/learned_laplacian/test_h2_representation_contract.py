import numpy as np
import torch

from mlr.data import Mesh
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.recovery_targets import (
    compose_absolute_laplacian_target,
    initial_uniform_laplacian,
    same_topology_oracle_target,
)
from mlr.learned_laplacian.target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    incident_edge_length_and_valid_mask,
    prediction_to_raw_laplacian,
)
from mlr.refinement import RefinementConfig, refine_mesh_with_laplacian


def _subdivided_triangle():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
    return vertices, faces


def test_expanded_midpoints_receive_h_from_their_own_current_graph_neighborhood():
    vertices, faces = _subdivided_triangle()
    h, valid = incident_edge_length_and_valid_mask(
        vertices, faces_to_edge_index(faces, len(vertices))
    )
    assert valid.all()
    expected_midpoint_h = torch.tensor(
        [
            (3.0 + 2.0**0.5) / 4.0,
            (2.0 + 2.0 * 2.0**0.5) / 4.0,
            (3.0 + 2.0**0.5) / 4.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(h[3:], expected_midpoint_h)


def test_inference_conversion_uses_supplied_current_expanded_h_not_stale_h():
    prediction = torch.ones((2, 3), dtype=torch.float64)
    current_h = torch.tensor([0.25, 0.5], dtype=torch.float64)
    stale_h = torch.tensor([2.0, 2.0], dtype=torch.float64)
    current_raw = prediction_to_raw_laplacian(
        prediction,
        current_h,
        input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
    )
    stale_raw = prediction_to_raw_laplacian(
        prediction,
        stale_h,
        input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
    )
    assert not torch.allclose(current_raw, stale_raw)
    torch.testing.assert_close(
        current_raw, prediction * (current_h.square() + 1e-12).unsqueeze(-1)
    )


def test_scale_zero_composition_remains_exact_after_representation_conversion():
    vertices, faces = _subdivided_triangle()
    initial = initial_uniform_laplacian(vertices.numpy(), faces.numpy())
    prediction_hat = torch.randn_like(vertices)
    prediction_raw = prediction_to_raw_laplacian(
        prediction_hat,
        incident_edge_length_and_valid_mask(
            vertices, faces_to_edge_index(faces, len(vertices))
        )[0],
        input_representation=EDGE_SCALE_NORMALIZED_LAPLACIAN,
    ).numpy()
    target = compose_absolute_laplacian_target(initial, prediction_raw, 0.0)
    np.testing.assert_array_equal(target, initial)


def test_identity_recovery_still_has_zero_update():
    vertices, faces = _subdivided_triangle()
    mesh = Mesh(vertices.numpy(), faces.numpy())
    delta = initial_uniform_laplacian(mesh.vertices, mesh.faces)
    result = refine_mesh_with_laplacian(
        mesh,
        delta,
        config=RefinementConfig(num_iters=5, learning_rate=0.01),
    )
    np.testing.assert_array_equal(result.vertices, mesh.vertices)


def test_same_topology_oracle_target_is_representation_independent():
    vertices, faces = _subdivided_triangle()
    target = vertices.numpy().copy()
    target[4, 2] = 0.25
    delta_initial, delta_target, residual = same_topology_oracle_target(
        vertices.numpy(), target, faces.numpy(), faces.numpy()
    )
    np.testing.assert_allclose(delta_initial + residual, delta_target)

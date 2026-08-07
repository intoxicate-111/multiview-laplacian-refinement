import numpy as np
import torch

from mlr.data import Mesh
from mlr.learned_laplacian.canonical_pipeline import (
    canonical_current_graph_recovery_inputs,
)
from mlr.refinement import RefinementConfig, refine_mesh_with_laplacian


def _mesh():
    return (
        torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([[0, 1, 2]], dtype=torch.long),
    )


def test_canonical_inference_recomputes_current_h_and_converts_exactly_once():
    vertices, faces = _mesh()
    delta_hat_prediction = torch.tensor(
        [[2.0, -1.0, 0.5], [1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    epsilon = 1e-3
    result = canonical_current_graph_recovery_inputs(
        vertices,
        faces,
        delta_hat_prediction,
        torch.ones((2, 3), dtype=torch.bool),
        torch.tensor([0.2, 0.5, 1.0]),
        epsilon=epsilon,
    )

    expected_h = torch.tensor(
        [(2.0 + 1.0) / 2.0, (2.0 + np.sqrt(5.0)) / 2.0, (1.0 + np.sqrt(5.0)) / 2.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.h_current, expected_h)
    torch.testing.assert_close(
        result.delta_pred_raw,
        delta_hat_prediction * (expected_h.square() + epsilon).unsqueeze(-1),
    )
    torch.testing.assert_close(result.weight, torch.tensor([0.2, 0.5, 1.0]))


def test_renderer_visibility_is_a_strict_confidence_gate():
    vertices, faces = _mesh()
    visibility = torch.tensor([[True, False, False], [False, False, True]])
    result = canonical_current_graph_recovery_inputs(
        vertices,
        faces,
        torch.ones_like(vertices),
        visibility,
        torch.ones(3),
    )
    torch.testing.assert_close(result.weight, torch.tensor([1.0, 0.0, 1.0]))
    assert result.weight[1].item() == 0.0


def test_h_is_recomputed_when_current_geometry_changes():
    vertices, faces = _mesh()
    visibility = torch.ones((1, 3), dtype=torch.bool)
    first = canonical_current_graph_recovery_inputs(
        vertices, faces, torch.ones_like(vertices), visibility
    )
    changed = vertices.clone()
    changed[1, 0] = 4.0
    second = canonical_current_graph_recovery_inputs(
        changed, faces, torch.ones_like(changed), visibility
    )
    assert not torch.equal(first.h_current, second.h_current)


def test_zero_learned_precision_preserves_x0_and_does_not_drive_lx_to_zero():
    vertices, faces = _mesh()
    mesh = Mesh(vertices.numpy(), faces.numpy())
    deliberately_wrong_target = np.full_like(mesh.vertices, 1000.0)
    result = refine_mesh_with_laplacian(
        mesh,
        deliberately_wrong_target,
        laplacian_weight=np.zeros(len(vertices)),
        anchors=mesh.vertices,
        config=RefinementConfig(
            lambda_lap=1.0,
            lambda_anchor=0.01,
            num_iters=10,
            learning_rate=0.01,
            robust_loss="huber",
        ),
    )
    np.testing.assert_array_equal(result.vertices, mesh.vertices)

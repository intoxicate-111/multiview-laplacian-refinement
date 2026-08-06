import numpy as np
import torch

from mlr.data import Mesh
from mlr.refinement import (
    RefinementConfig,
    refine_mesh_with_laplacian,
    visibility_weighted_laplacian_residual,
)
from mlr.learned_laplacian.visibility_recovery import hard_any_view_recovery_mask


def test_all_view_invisible_vertex_has_exact_zero_recovery_weight():
    visibility = torch.tensor(
        [
            [False, True, False],
            [False, False, True],
            [False, True, True],
            [False, False, False],
        ]
    )
    mask = hard_any_view_recovery_mask(visibility, num_vertices=3)

    assert mask.visibility_count.tolist() == [0, 2, 2]
    assert mask.visible_any.tolist() == [False, True, True]
    assert mask.laplacian_weight.tolist() == [0.0, 1.0, 1.0]


def test_one_visible_view_produces_unit_recovery_weight():
    visibility = torch.tensor([[False, False, True, False]])
    mask = hard_any_view_recovery_mask(visibility, num_vertices=1)

    assert mask.visibility_count.item() == 1
    assert mask.laplacian_weight.item() == 1.0


def test_unseen_row_is_removed_instead_of_receiving_zero_laplacian_target():
    laplacian_at_x = np.array([[3.0, -2.0, 1.0], [1.0, 2.0, 3.0]])
    delta_prediction = np.array([[4.0, 5.0, -6.0], [-2.0, 1.0, 4.0]])
    residual = laplacian_at_x - delta_prediction
    weighted = visibility_weighted_laplacian_residual(residual, [0.0, 1.0])

    np.testing.assert_array_equal(weighted[0], np.zeros(3))
    np.testing.assert_array_equal(weighted[1], residual[1])
    assert not np.array_equal(weighted[0], laplacian_at_x[0])


def test_all_visible_recovery_matches_baseline():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    mesh = Mesh(vertices, np.array([[0, 1, 2]], dtype=np.int64))
    delta = np.array([[0.2, 0.1, 0.0], [-0.1, 0.3, 0.0], [0.4, -0.2, 0.0]])
    config = RefinementConfig(num_iters=8, learning_rate=1e-3, robust_loss="huber")

    baseline = refine_mesh_with_laplacian(mesh, delta, config=config)
    weighted = refine_mesh_with_laplacian(
        mesh, delta, config=config, laplacian_weight=np.ones(3)
    )

    np.testing.assert_allclose(weighted.vertices, baseline.vertices, rtol=0.0, atol=1e-12)


def test_view_permutation_keeps_hard_recovery_mask_unchanged():
    visibility = torch.tensor(
        [
            [False, True, False, True, False],
            [True, False, False, True, False],
            [False, False, True, False, False],
            [True, True, False, False, False],
        ]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    original = hard_any_view_recovery_mask(visibility, num_vertices=5)
    shuffled = hard_any_view_recovery_mask(
        visibility[permutation], num_vertices=5
    )

    torch.testing.assert_close(original.visibility_count, shuffled.visibility_count)
    torch.testing.assert_close(original.visible_any, shuffled.visible_any)
    torch.testing.assert_close(original.laplacian_weight, shuffled.laplacian_weight)

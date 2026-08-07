import numpy as np
import pytest

from mlr.learned_laplacian.recovery_targets import (
    compose_absolute_laplacian_target,
    compose_residual_laplacian_target,
    initial_uniform_laplacian,
    same_topology_oracle_target,
)


def _mesh():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return vertices, faces


def test_scale_zero_absolute_target_is_exact_identity_target():
    vertices, faces = _mesh()
    initial = initial_uniform_laplacian(vertices, faces)
    predicted = np.full_like(initial, 123.0)
    actual = compose_absolute_laplacian_target(initial, predicted, 0.0)
    np.testing.assert_array_equal(actual, initial)


def test_residual_target_composition():
    initial = np.arange(12, dtype=np.float64).reshape(4, 3)
    residual = np.full((4, 3), 2.0)
    weights = np.array([1.0, 0.0, 0.5, 1.0])
    actual = compose_residual_laplacian_target(initial, residual, 0.25, weights)
    np.testing.assert_allclose(actual, initial + 0.25 * weights[:, None] * residual)


def test_visibility_gates_correction_without_deleting_initial_geometry():
    initial = np.arange(12, dtype=np.float64).reshape(4, 3)
    predicted = initial + 10.0
    visible = np.array([1.0, 0.0, 1.0, 0.0])
    actual = compose_absolute_laplacian_target(initial, predicted, 1.0, visible)
    np.testing.assert_array_equal(actual[~visible.astype(bool)], initial[~visible.astype(bool)])
    np.testing.assert_array_equal(actual[visible.astype(bool)], predicted[visible.astype(bool)])


def test_same_topology_oracle_target_construction():
    current, faces = _mesh()
    target = current.copy()
    target[3] += np.array([0.1, -0.2, 0.3])
    delta_initial, delta_target, residual = same_topology_oracle_target(
        current, target, faces, faces.copy()
    )
    np.testing.assert_allclose(delta_initial + residual, delta_target)
    np.testing.assert_allclose(delta_target, initial_uniform_laplacian(target, faces))


def test_same_topology_oracle_rejects_order_or_topology_mismatch():
    current, faces = _mesh()
    with pytest.raises(ValueError, match="vertices"):
        same_topology_oracle_target(current, current[:-1], faces, faces)
    reordered = faces.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    with pytest.raises(ValueError, match="faces"):
        same_topology_oracle_target(current, current, faces, reordered)

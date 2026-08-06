import numpy as np

from mlr.data import Mesh
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
from mlr.learned_laplacian.sample_io import corrupt_same_topology_mesh

from .helpers import tiny_sample


def test_same_topology_corruption_is_deterministic_and_preserves_faces():
    sample = tiny_sample()
    mesh = Mesh(sample["vertices"].numpy(), sample["faces"].numpy()).ensure_normals()
    first = corrupt_same_topology_mesh(
        mesh, noise_std=0.02, smoothing_iters=2, smoothing_strength=0.1, seed=7
    )
    second = corrupt_same_topology_mesh(
        mesh, noise_std=0.02, smoothing_iters=2, smoothing_strength=0.1, seed=7
    )
    np.testing.assert_array_equal(first.faces, mesh.faces)
    np.testing.assert_allclose(first.vertices, second.vertices)
    assert not np.allclose(first.vertices, mesh.vertices)


def test_sparse_uniform_reconstruction_path_writes_error_arrays(tmp_path):
    sample = tiny_sample()
    metrics = reconstruct_and_evaluate(
        sample,
        sample["laplacian_target"],
        tmp_path,
        {
            "operator_type": "uniform",
            "lambda_lap": 1.0,
            "lambda_anchor": 0.01,
            "num_iters": 3,
            "learning_rate": 0.01,
            "dense_vertex_limit": 2,
            "chamfer_samples": 0,
        },
    )
    assert metrics["reconstruction"]["predicted_solver"] == "sparse_uniform_oracle_core"
    assert metrics["reconstruction"]["all_finite"]
    assert (tmp_path / "laplacian_error.npy").exists()
    assert (tmp_path / "position_error.npy").exists()


def test_reconstruction_can_disable_placeholder_oracle(tmp_path):
    sample = tiny_sample()
    metrics = reconstruct_and_evaluate(
        sample,
        sample["laplacian_target"],
        tmp_path,
        {
            "operator_type": "uniform",
            "num_iters": 1,
            "dense_vertex_limit": 100,
            "chamfer_samples": 0,
            "evaluate_oracle": False,
        },
    )

    assert "oracle" not in metrics["geometry"]
    assert metrics["reconstruction"]["oracle_evaluated"] is False
    assert metrics["reconstruction"]["oracle_final_loss"] is None
    assert not (tmp_path / "oracle_refined.obj").exists()

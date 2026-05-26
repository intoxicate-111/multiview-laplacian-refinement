import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.coarse_lap_oracle import (  # noqa: E402
    CoarseGraphOracleConfig,
    apply_uniform_laplacian,
    build_uniform_laplacian_data,
    local_vertex_scales,
    prepare_coarse_graph_targets,
    run_coarse_graph_laplacian_oracles,
)
from mlr.data import Mesh  # noqa: E402


class CoarseLapOracleTests(unittest.TestCase):
    def test_targets_are_defined_on_coarse_vertex_set(self):
        coarse = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.1],
                    [1.0, 0.0, 0.1],
                    [0.0, 1.0, 0.1],
                    [1.0, 1.0, 0.1],
                ]
            ),
            faces=np.array([[0, 1, 2], [1, 3, 2]]),
        )
        gt = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]]),
        )
        targets = prepare_coarse_graph_targets(
            coarse,
            gt,
            CoarseGraphOracleConfig(num_iters=2, reg_iters=2),
        )
        self.assertEqual(targets.projected_vertices.shape, coarse.vertices.shape)
        self.assertEqual(targets.delta_target.shape, coarse.vertices.shape)
        self.assertEqual(targets.h.shape[0], coarse.num_vertices)
        self.assertEqual(targets.laplacian_matrix.shape, (coarse.num_vertices, coarse.num_vertices))
        np.testing.assert_allclose(targets.delta_target, targets.laplacian_matrix @ targets.projected_vertices)
        self.assertNotEqual(targets.projected_vertices.shape[0], gt.num_vertices)

    def test_normalize_recover_is_identity(self):
        coarse = Mesh(
            vertices=np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2], [0.0, 1.0, 0.2]]),
            faces=np.array([[0, 1, 2]]),
        )
        gt = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        config = CoarseGraphOracleConfig(num_iters=2, normalized_eps=1e-8, reg_iters=2)
        targets = prepare_coarse_graph_targets(coarse, gt, config)
        recovered = targets.delta_hat_target * (targets.h[:, None] ** 2 + config.normalized_eps)
        np.testing.assert_allclose(recovered, targets.delta_target)

    def test_uniform_apply_matches_dense_matrix(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        data = build_uniform_laplacian_data(faces, len(vertices))
        dense = np.array(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, -0.5],
                [-0.5, -0.5, 1.0],
            ]
        )
        np.testing.assert_allclose(apply_uniform_laplacian(vertices, data), dense @ vertices)
        h = local_vertex_scales(vertices, data)
        self.assertEqual(h.shape, (3,))
        self.assertTrue(np.all(h > 0.0))

    def test_full_run_writes_expected_debug_outputs(self):
        coarse = Mesh(
            vertices=np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2], [0.0, 1.0, 0.2]]),
            faces=np.array([[0, 1, 2]]),
        )
        gt = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_coarse_graph_laplacian_oracles(
                coarse,
                gt,
                tmp,
                CoarseGraphOracleConfig(num_iters=3, log_every=1, chamfer_samples=0, reg_iters=2),
            )
            root = Path(tmp)
            self.assertTrue((root / "projected_gt_on_coarse.obj").exists())
            self.assertTrue((root / "registered_gt_on_coarse.obj").exists())
            self.assertTrue((root / "delta_target.npy").exists())
            self.assertTrue((root / "h.npy").exists())
            self.assertTrue((root / "delta_hat_target.npy").exists())
            self.assertTrue((root / "log_raw.json").exists())
            self.assertTrue((root / "log_normalized.json").exists())

    def test_full_run_can_use_cuda_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        coarse = Mesh(
            vertices=np.array([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2], [0.0, 1.0, 0.2]]),
            faces=np.array([[0, 1, 2]]),
        )
        gt = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_coarse_graph_laplacian_oracles(
                coarse,
                gt,
                tmp,
                CoarseGraphOracleConfig(
                    device="cuda",
                    num_iters=2,
                    log_every=1,
                    chamfer_samples=0,
                    reg_iters=2,
                ),
            )
        self.assertEqual(summary["comparison"]["raw_oracle_refined"]["device"], "cuda")


if __name__ == "__main__":
    unittest.main()

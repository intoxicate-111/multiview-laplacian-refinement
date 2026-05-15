import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.gt_laplacian import (
    GTLaplacianTargetConfig,
    interpolate_gt_laplacian_to_coarse,
    refine_coarse_mesh_with_gt_laplacian,
)
from mlr.refinement import RefinementConfig


class GTLaplacianTests(unittest.TestCase):
    def test_interpolates_gt_values_with_barycentric_weights(self):
        gt_mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        coarse_mesh = Mesh(
            vertices=np.array(
                [
                    [0.25, 0.25, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        gt_values = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )

        target = interpolate_gt_laplacian_to_coarse(
            coarse_mesh,
            gt_mesh,
            gt_laplacian_values=gt_values,
        )

        np.testing.assert_allclose(target.barycentric[0], [0.5, 0.25, 0.25], atol=1e-12)
        np.testing.assert_allclose(target.delta_target[0], [0.5, 0.5, 0.75], atol=1e-12)
        self.assertEqual(target.delta_target.shape, coarse_mesh.vertices.shape)

    def test_refinement_accepts_interpolated_gt_laplacian_target(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        coarse_mesh = Mesh(vertices.copy(), faces.copy())
        gt_mesh = Mesh(vertices.copy(), faces.copy())

        result = refine_coarse_mesh_with_gt_laplacian(
            coarse_mesh,
            gt_mesh,
            target_config=GTLaplacianTargetConfig(operator_type="uniform"),
            refinement_config=RefinementConfig(
                operator_type="uniform",
                num_iters=5,
                learning_rate=1e-3,
                lambda_anchor=0.0,
            ),
        )

        self.assertEqual(result.vertices.shape, coarse_mesh.vertices.shape)
        self.assertTrue(result.history)
        np.testing.assert_allclose(result.vertices, coarse_mesh.vertices, atol=1e-8)

    def test_refinement_allows_coarse_and_gt_vertex_count_mismatch(self):
        gt_mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        coarse_mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.2],
                    [1.0, 0.0, 0.2],
                    [0.0, 1.0, 0.2],
                    [0.3, 0.3, 0.6],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        )

        result = refine_coarse_mesh_with_gt_laplacian(
            coarse_mesh,
            gt_mesh,
            target_config=GTLaplacianTargetConfig(operator_type="uniform"),
            refinement_config=RefinementConfig(
                operator_type="uniform",
                num_iters=3,
                learning_rate=1e-3,
            ),
        )

        self.assertEqual(gt_mesh.num_vertices, 3)
        self.assertEqual(coarse_mesh.num_vertices, 4)
        self.assertEqual(result.target.delta_target.shape, coarse_mesh.vertices.shape)
        self.assertEqual(result.vertices.shape, coarse_mesh.vertices.shape)


if __name__ == "__main__":
    unittest.main()

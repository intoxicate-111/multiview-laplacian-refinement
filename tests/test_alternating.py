import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.alternating import AlternatingRefinementConfig, alternating_refinement_loop
from mlr.data import Camera, Mesh
from mlr.refinement import RefinementConfig


class AlternatingTests(unittest.TestCase):
    def test_loop_runs_with_identity_pseudo_surface(self):
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        camera = Camera(
            intrinsics=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_size=(100, 100),
        )
        config = AlternatingRefinementConfig(
            num_outer_iters=2,
            inner=RefinementConfig(num_iters=5, learning_rate=1e-3),
        )
        result = alternating_refinement_loop(mesh, images=None, cameras=[camera], config=config)
        self.assertEqual(result.mesh.vertices.shape, mesh.vertices.shape)
        self.assertEqual(len(result.outer_history), 2)


if __name__ == "__main__":
    unittest.main()

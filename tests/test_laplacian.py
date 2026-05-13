import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.laplacian import build_laplacian, compute_laplacian_coordinates


class LaplacianTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.faces = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ]
        )

    def test_uniform_laplacian_rows_sum_to_zero(self):
        operator = build_laplacian(self.vertices, self.faces, "uniform")
        np.testing.assert_allclose(operator.matrix.sum(axis=1), 0.0, atol=1e-12)

    def test_constant_positions_have_zero_laplacian(self):
        positions = np.ones_like(self.vertices)
        delta = compute_laplacian_coordinates(positions, self.faces, "uniform")
        np.testing.assert_allclose(delta, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

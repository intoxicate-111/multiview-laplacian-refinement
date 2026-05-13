import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.metrics import correspondence_metrics
from mlr.oracle import OracleBaselineConfig, run_oracle_baselines


class OracleTests(unittest.TestCase):
    def test_position_baseline_reduces_correspondence_error(self):
        gt_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        init_vertices = gt_vertices + np.array([0.1, -0.05, 0.02])
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        init_mesh = Mesh(init_vertices, faces)
        config = OracleBaselineConfig(num_iters=50, learning_rate=1e-2, lambda_anchor=0.1)
        results = run_oracle_baselines(init_mesh, gt_vertices, config)
        init_rmse = correspondence_metrics(init_vertices, gt_vertices)["rmse"]
        final_rmse = correspondence_metrics(results["position_only"].vertices, gt_vertices)["rmse"]
        self.assertLess(final_rmse, init_rmse)


if __name__ == "__main__":
    unittest.main()

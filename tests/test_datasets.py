import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.datasets import load_reconstruction_input
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset


class DatasetLoaderTests(unittest.TestCase):
    def test_loads_synthetic_dataset(self):
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset = generate_synthetic_dataset(
                mesh,
                tmp,
                SyntheticRenderConfig(num_views=2, width=32, height=32),
            )
            loaded = load_reconstruction_input(dataset.dataset_path)
        self.assertEqual(len(loaded.image_paths), 2)
        self.assertEqual(len(loaded.cameras), 2)
        self.assertEqual(len(loaded.mask_paths), 2)
        self.assertIsNotNone(loaded.gt_mesh_path)


if __name__ == "__main__":
    unittest.main()

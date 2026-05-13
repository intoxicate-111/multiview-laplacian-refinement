import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh


class IoTests(unittest.TestCase):
    def test_obj_roundtrip(self):
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.obj"
            save_mesh(mesh, path)
            loaded = load_mesh(path)
        np.testing.assert_allclose(loaded.vertices, mesh.vertices)
        np.testing.assert_array_equal(loaded.faces, mesh.faces)
        self.assertEqual(loaded.normals.shape, mesh.vertices.shape)


if __name__ == "__main__":
    unittest.main()

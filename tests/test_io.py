import sys
import struct
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

    def test_binary_little_endian_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.ply"
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex 3\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "element face 1\n"
                "property list uchar uint vertex_indices\n"
                "end_header\n"
            ).encode("ascii")
            body = b"".join(
                [
                    struct.pack("<fff", 0.0, 0.0, 0.0),
                    struct.pack("<fff", 1.0, 0.0, 0.0),
                    struct.pack("<fff", 0.0, 1.0, 0.0),
                    struct.pack("<BIII", 3, 0, 1, 2),
                ]
            )
            path.write_bytes(header + body)
            loaded = load_mesh(path)
        np.testing.assert_allclose(loaded.vertices, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        np.testing.assert_array_equal(loaded.faces, np.array([[0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()

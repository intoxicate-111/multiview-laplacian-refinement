import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.coarse import OpenMVSCommandMeshGenerator, write_colmap_text_model, write_openmvg_sfm_data
from mlr.data import Camera, Mesh
from mlr.io import save_mesh


class OpenMVSAdapterTests(unittest.TestCase):
    def test_writes_openmvg_sfm_data(self):
        camera = Camera(
            intrinsics=np.array([[120.0, 0.0, 32.0], [0.0, 120.0, 24.0], [0.0, 0.0, 1.0]]),
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_size=(64, 48),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "000.png"
            sfm_path = root / "sfm_data.json"
            write_openmvg_sfm_data(sfm_path, [image], [camera], scene_dir=root)
            payload = json.loads(sfm_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["sfm_data_version"], "0.3")
        self.assertEqual(len(payload["views"]), 1)
        self.assertEqual(payload["views"][0]["value"]["ptr_wrapper"]["data"]["filename"], "images/000.png")

    def test_writes_colmap_text_model(self):
        camera = Camera(
            intrinsics=np.array([[120.0, 0.0, 32.0], [0.0, 90.0, 24.0], [0.0, 0.0, 1.0]]),
            rotation=np.eye(3),
            translation=np.array([0.1, 0.2, 0.3]),
            image_size=(64, 48),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source.png"
            image.write_bytes(b"fake image bytes")
            colmap_root = root / "colmap"
            write_colmap_text_model(colmap_root, [image], [camera])
            cameras_txt = (colmap_root / "sparse" / "cameras.txt").read_text(encoding="utf-8")
            images_txt = (colmap_root / "sparse" / "images.txt").read_text(encoding="utf-8")
            self.assertTrue((colmap_root / "images" / "00000001.png").exists())

        self.assertIn("1 PINHOLE 64 48 120", cameras_txt)
        self.assertIn("1 1 0 0 0", images_txt)
        self.assertIn("1 00000001.png", images_txt)

    def test_openmvs_adapter_can_use_command_template(self):
        camera = Camera(
            intrinsics=np.eye(3),
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_size=(8, 8),
        )
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "coarse.obj"
            save_mesh(mesh, output)
            generator = OpenMVSCommandMeshGenerator(
                scene_dir=root / "scene",
                output_mesh_path=output,
                command_template='python -c "print(\\"mock openmvs\\")"',
                interface_format="openmvg",
                compute_visibility=False,
            )
            loaded = generator.generate([root / "images" / "000.png"], [camera])
            self.assertEqual(loaded.num_faces, 1)
            self.assertTrue((root / "scene" / "sfm_data.json").exists())

    def test_openmvs_adapter_can_prepare_colmap_input(self):
        camera = Camera(
            intrinsics=np.eye(3),
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_size=(8, 8),
        )
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "000.png"
            image.parent.mkdir()
            image.write_bytes(b"fake image bytes")
            output = root / "coarse.obj"
            save_mesh(mesh, output)
            generator = OpenMVSCommandMeshGenerator(
                scene_dir=root / "scene",
                output_mesh_path=output,
                command_template='python -c "print(\\"mock openmvs colmap\\")"',
                interface_format="colmap",
                compute_visibility=False,
            )
            loaded = generator.generate([image], [camera])
            self.assertEqual(loaded.num_faces, 1)
            self.assertTrue((root / "scene" / "colmap" / "sparse" / "cameras.txt").exists())
            self.assertTrue((root / "scene" / "colmap" / "images" / "00000001.png").exists())


if __name__ == "__main__":
    unittest.main()

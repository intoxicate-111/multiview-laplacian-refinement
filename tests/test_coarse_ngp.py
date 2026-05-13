import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.coarse import camera_to_nerf_transform, write_instant_ngp_transforms
from mlr.coarse import NvidiaInstantNGPMeshGenerator
from mlr.data import Camera
from mlr.io import save_mesh
from mlr.data import Mesh


class InstantNGPAdapterTests(unittest.TestCase):
    def test_camera_to_nerf_transform_uses_camera_center(self):
        camera = Camera(
            intrinsics=np.eye(3),
            rotation=np.eye(3),
            translation=np.array([1.0, 2.0, 3.0]),
        )
        transform = camera_to_nerf_transform(camera, convert_cv_to_gl=False)
        np.testing.assert_allclose(transform[:3, 3], np.array([-1.0, -2.0, -3.0]))

    def test_writes_transforms_json(self):
        camera = Camera(
            intrinsics=np.array([[100.0, 0.0, 32.0], [0.0, 90.0, 24.0], [0.0, 0.0, 1.0]]),
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_size=(64, 48),
        )
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp)
            image = scene_dir / "images" / "000.png"
            path = scene_dir / "transforms.json"
            write_instant_ngp_transforms(path, [image], [camera], scene_dir)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["fl_x"], 100.0)
        self.assertEqual(payload["w"], 64)
        self.assertEqual(payload["frames"][0]["file_path"], "images/000.png")
        self.assertEqual(len(payload["frames"][0]["transform_matrix"]), 4)

    def test_ngp_adapter_can_use_command_template(self):
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
            generator = NvidiaInstantNGPMeshGenerator(
                scene_dir=root / "scene",
                output_mesh_path=output,
                command_template='python -c "print(\\"mock ngp\\")"',
                compute_visibility=False,
            )
            loaded = generator.generate([root / "images" / "000.png"], [camera])
            self.assertEqual(loaded.num_faces, 1)
            self.assertTrue((root / "scene" / "transforms.json").exists())


if __name__ == "__main__":
    unittest.main()

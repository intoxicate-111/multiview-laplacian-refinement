import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.synthetic import (
    SyntheticRenderConfig,
    create_synthetic_cameras,
    generate_synthetic_dataset,
    render_mesh_view,
)
from mlr.synthetic import generate_synthetic_datasets_from_mesh_dir


class SyntheticDatasetTests(unittest.TestCase):
    def test_renders_and_writes_dataset(self):
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
        config = SyntheticRenderConfig(num_views=3, width=64, height=64, render_mode="normal")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = generate_synthetic_dataset(mesh, tmp, config)
            payload = json.loads(dataset.dataset_path.read_text(encoding="utf-8"))
            rgb, mask, depth = render_mesh_view(mesh, dataset.cameras[0], config)
            self.assertEqual(len(payload["image_paths"]), 3)
            self.assertTrue(dataset.image_paths[0].exists())
            self.assertTrue(dataset.mask_paths[0].exists())
            self.assertTrue(dataset.depth_paths[0].exists())
            self.assertEqual(rgb.shape, (64, 64, 3))
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(depth.shape, (64, 64))

    def test_generates_dataset_for_each_mesh_in_directory(self):
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
        config = SyntheticRenderConfig(num_views=2, width=32, height=32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir()
            save_mesh(mesh, mesh_dir / "a.obj")
            save_mesh(mesh, mesh_dir / "b.obj")
            datasets = generate_synthetic_datasets_from_mesh_dir(mesh_dir, root / "inputs", config)
            self.assertEqual(len(datasets), 2)
            self.assertTrue((root / "inputs" / "a" / "dataset.json").exists())
            self.assertTrue((root / "inputs" / "b" / "images" / "0001.png").exists())

    def test_unknown_backend_fails_clearly(self):
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        config = SyntheticRenderConfig(num_views=1, width=16, height=16)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = generate_synthetic_dataset(mesh, tmp, config)
            with self.assertRaises(ValueError):
                render_mesh_view(mesh, dataset.cameras[0], SyntheticRenderConfig(backend="bogus"))

    def test_sphere_trajectory_uses_multiple_elevations(self):
        mesh = Mesh(
            vertices=np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            faces=np.array([[0, 1, 2]]),
        )
        cameras = create_synthetic_cameras(
            mesh,
            num_views=8,
            image_size=(32, 32),
            trajectory="sphere",
            min_elevation_degrees=-60,
            max_elevation_degrees=60,
        )
        centers_y = np.array([camera.center[1] for camera in cameras])
        self.assertGreater(float(centers_y.max() - centers_y.min()), 0.5)


if __name__ == "__main__":
    unittest.main()

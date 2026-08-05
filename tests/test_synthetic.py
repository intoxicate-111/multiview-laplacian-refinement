import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.synthetic import (
    CUBE_SURFACE_VIEW_NAMES,
    SyntheticRenderConfig,
    create_synthetic_cameras,
    generate_synthetic_dataset,
    render_mesh_view,
    render_mesh_views_opengl,
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

    def test_cuda_backend_renders_or_fails_clearly(self):
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        config = SyntheticRenderConfig(num_views=1, width=16, height=16)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = generate_synthetic_dataset(mesh, tmp, config)
            cuda_config = SyntheticRenderConfig(backend="cuda", width=16, height=16)
            try:
                rgb, mask, depth = render_mesh_view(mesh, dataset.cameras[0], cuda_config)
            except RuntimeError as exc:
                self.assertIn("CUDA backend requires", str(exc))
            else:
                self.assertEqual(rgb.shape, (16, 16, 3))
                self.assertEqual(mask.shape, (16, 16))
                self.assertEqual(depth.shape, (16, 16))

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

    def test_cube_surface_uses_exact_face_and_corner_centers(self):
        mesh = Mesh(
            vertices=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        cameras = create_synthetic_cameras(
            mesh,
            num_views=14,
            image_size=(32, 32),
            trajectory="cube_surface",
            cube_half_extent=1.5,
            fov_degrees=90.0,
        )
        centers = np.stack([camera.center for camera in cameras])
        expected = np.asarray([
            (1.5, 0, 0), (-1.5, 0, 0), (0, 1.5, 0), (0, -1.5, 0),
            (0, 0, 1.5), (0, 0, -1.5),
            (-1.5, -1.5, -1.5), (-1.5, -1.5, 1.5),
            (-1.5, 1.5, -1.5), (-1.5, 1.5, 1.5),
            (1.5, -1.5, -1.5), (1.5, -1.5, 1.5),
            (1.5, 1.5, -1.5), (1.5, 1.5, 1.5),
        ])
        np.testing.assert_allclose(centers, expected, atol=1e-10, rtol=0.0)
        self.assertEqual(tuple(camera.name for camera in cameras), CUBE_SURFACE_VIEW_NAMES)
        self.assertEqual(np.stack([camera.intrinsics for camera in cameras]).shape, (14, 3, 3))
        extrinsics = np.stack([
            np.block([[camera.rotation, camera.translation[:, None]], [np.array([[0.0, 0.0, 0.0, 1.0]])]])
            for camera in cameras
        ])
        self.assertEqual(extrinsics.shape, (14, 4, 4))
        np.testing.assert_allclose(np.linalg.norm(centers[:6], axis=1), 1.5)
        np.testing.assert_allclose(np.linalg.norm(centers[6:], axis=1), np.sqrt(3.0) * 1.5)
        forwards = np.stack([camera.rotation[2] for camera in cameras])
        np.testing.assert_allclose(
            forwards, -centers / np.linalg.norm(centers, axis=1, keepdims=True), atol=1e-10
        )

    def test_opengl_egl_cube_render_is_deterministic_when_available(self):
        vertices = np.array([
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ], dtype=np.float64)
        faces = np.array([
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
        ], dtype=np.int64)
        mesh = Mesh(vertices, faces)
        config = SyntheticRenderConfig(
            num_views=14, width=64, height=64, trajectory="cube_surface",
            fov_degrees=90.0, backend="opengl", normalize_mesh=False,
            opengl_context_backend="egl", cube_half_extent=1.5, antialiasing="msaa4",
        )
        cameras = create_synthetic_cameras(
            mesh, 14, (64, 64), trajectory="cube_surface",
            cube_half_extent=1.5, fov_degrees=90.0,
        )
        try:
            first = render_mesh_views_opengl(mesh, cameras, config)
            second = render_mesh_views_opengl(mesh, cameras, config)
        except Exception as exc:  # pragma: no cover - depends on CI EGL support
            self.skipTest(f"EGL unavailable: {exc}")
        for (rgb_a, mask_a, depth_a), (rgb_b, mask_b, depth_b) in zip(first, second, strict=True):
            np.testing.assert_array_equal(rgb_a, rgb_b)
            np.testing.assert_array_equal(mask_a, mask_b)
            np.testing.assert_allclose(depth_a, depth_b, atol=1e-7, rtol=0.0)
            self.assertTrue(np.any(mask_a))
            self.assertTrue(np.isfinite(depth_a[mask_a]).all())
            self.assertTrue(np.all(depth_a[mask_a] > 0))

    def test_cube_surface_face_views_frame_unit_sphere_proxy(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh = Mesh(np.asarray(sphere.vertices), np.asarray(sphere.faces))
        cameras = create_synthetic_cameras(
            mesh, 14, (128, 128), trajectory="cube_surface",
            cube_half_extent=1.5, fov_degrees=90.0,
        )
        config = SyntheticRenderConfig(width=128, height=128, backend="cpu", normalize_mesh=False)
        for camera in cameras[:6]:
            _, mask, depth = render_mesh_view(mesh, camera, config)
            self.assertTrue(np.any(mask))
            self.assertFalse(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
            ys, xs = np.nonzero(mask)
            occupancy = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / 128.0
            self.assertGreaterEqual(occupancy, 0.70)
            self.assertLessEqual(occupancy, 0.95)
            self.assertTrue(np.isfinite(depth[mask]).all())

    def test_cube_dataset_json_files_store_actual_camera_matrices(self):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        mesh = Mesh(np.asarray(sphere.vertices), np.asarray(sphere.faces))
        config = SyntheticRenderConfig(
            num_views=14,
            width=24,
            height=20,
            trajectory="cube_surface",
            fov_degrees=90.0,
            backend="cpu",
            normalize_mesh=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            dataset = generate_synthetic_dataset(mesh, tmp, config)
            camera_payload = json.loads((Path(tmp) / "cameras.json").read_text(encoding="utf-8"))
            dataset_payload = json.loads(dataset.dataset_path.read_text(encoding="utf-8"))

        for payload in (camera_payload, dataset_payload["cameras"]):
            self.assertEqual(tuple(item["name"] for item in payload), CUBE_SURFACE_VIEW_NAMES)
            self.assertEqual(np.asarray([item["intrinsics"] for item in payload]).shape, (14, 3, 3))
            self.assertEqual(np.asarray([item["extrinsics"] for item in payload]).shape, (14, 4, 4))


if __name__ == "__main__":
    unittest.main()

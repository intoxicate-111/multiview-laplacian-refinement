import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlr.coarse import NvidiaNvdiffrecMeshGenerator
from mlr.data import Mesh
from mlr.datasets import load_reconstruction_input
from mlr.io import save_mesh
from mlr.nvdiffrec import NvdiffrecRunConfig, export_nvdiffrec_nerf_dataset, write_nvdiffrec_config
from mlr.synthetic import SyntheticRenderConfig, generate_synthetic_dataset


class NvdiffrecAdapterTests(unittest.TestCase):
    def test_exports_rgba_nerf_dataset_and_config(self):
        mesh = _tetra_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = generate_synthetic_dataset(
                mesh,
                root / "inputs",
                SyntheticRenderConfig(num_views=2, width=32, height=32),
            )
            data = load_reconstruction_input(dataset.dataset_path)
            nerf_dir = export_nvdiffrec_nerf_dataset(data, root / "nerf")
            cfg_path = write_nvdiffrec_config(
                root / "config.json",
                nerf_dir,
                root / "out",
                NvdiffrecRunConfig(iterations=3, train_res=(32, 32), texture_res=(32, 32)),
            )
            transforms = json.loads((nerf_dir / "transforms_train.json").read_text(encoding="utf-8"))
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(len(transforms["frames"]), 2)
        self.assertIn("camera_angle_x", transforms)
        self.assertEqual(transforms["frames"][0]["file_path"], "images/0000")
        self.assertEqual(cfg["iter"], 3)
        self.assertEqual(cfg["ref_mesh"], str(nerf_dir))

    def test_nvdiffrec_generator_reads_mock_result_mesh(self):
        mesh = _tetra_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = generate_synthetic_dataset(
                mesh,
                root / "inputs",
                SyntheticRenderConfig(num_views=1, width=32, height=32),
            )
            data = load_reconstruction_input(dataset.dataset_path)
            result_mesh = root / "mock_result.obj"
            output_mesh = root / "coarse.obj"
            save_mesh(mesh, result_mesh)
            generator = NvidiaNvdiffrecMeshGenerator(
                run_dir=root / "run",
                output_mesh_path=output_mesh,
                nvdiffrec_root=root,
                command_template='python -c "print(\\"mock nvdiffrec\\")"',
                result_mesh_path=result_mesh,
                run_config=NvdiffrecRunConfig(iterations=1, train_res=(32, 32), texture_res=(32, 32)),
                compute_visibility=False,
            )
            loaded = generator.generate(data.image_paths, data.cameras)
            self.assertTrue(output_mesh.exists())
        self.assertEqual(loaded.num_faces, mesh.num_faces)


def _tetra_mesh() -> Mesh:
    return Mesh(
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mlr.baselines.future2000 import (
    export_nds_scene,
    export_nerf_scene,
    export_nvdiffrec_scene,
    export_openmvs_scene,
)
from mlr.io import load_mesh


class Poison:
    def __array__(self):
        raise AssertionError("GT/evaluation field was accessed")


def _sample(root: Path) -> dict:
    image_paths = []
    for index in range(2):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2:6, 2:6] = 64 + index
        path = root / f"source_{index}.png"
        Image.fromarray(rgb).save(path)
        image_paths.append(path.name)
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    extrinsics = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    extrinsics[1, 0, 3] = 0.25
    return {
        "sample_id": "held_out__v00",
        "vertices": vertices,
        "faces": faces,
        "image_paths": image_paths,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "_dataset_root": str(root),
        # Poisoned labels prove the exporters' explicit field boundary.
        "target_positions": Poison(),
        "gt_vertices": Poison(),
        "gt_faces": Poison(),
        "raw_laplacian_target": Poison(),
    }


def test_nds_export_consumes_no_gt_and_preserves_current_mesh(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    result = export_nds_scene(sample, tmp_path / "nds")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    exported = load_mesh(result.initial_obj)

    assert result.view_count == 2
    assert np.allclose(exported.vertices, sample["vertices"].numpy())
    assert set(metadata["consumed_sample_fields"]).isdisjoint(
        {"target_positions", "gt_vertices", "gt_faces", "raw_laplacian_target"}
    )
    assert (result.scene_dir / "views/0001_k.txt").is_file()
    assert Image.open(result.scene_dir / "views/0001.png").mode == "RGBA"


def test_nerf_exports_are_camera_complete_and_gt_free(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    result = export_nerf_scene(sample, tmp_path / "nerf2mesh", method="nerf2mesh")
    transforms = json.loads(
        (result.scene_dir / "transforms_train.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(transforms["frames"]) == 2
    assert transforms["fl_x"] == 8.0
    assert metadata["forbidden_fields_consumed"] == []
    assert (result.scene_dir / "mask/0001.png").is_file()


def test_exmesh_export_uses_exactly_one_colmap_copy_of_each_view(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    result = export_nerf_scene(sample, tmp_path / "exmesh", method="exmesh")
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert metadata["view_count"] == 2
    assert metadata["camera_format"] == "COLMAP text model in sparse/0"
    assert metadata["forbidden_fields_consumed"] == []
    assert not (result.scene_dir / "transforms_train.json").exists()
    assert (result.scene_dir / "sparse/0/cameras.txt").is_file()
    image_lines = [
        line
        for line in (result.scene_dir / "sparse/0/images.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert len(image_lines) == 2
    assert (result.scene_dir / "images/00000001.png").is_file()
    assert (result.scene_dir / "mesh.ply").is_file()
    assert (result.scene_dir / "train_mask/00000001_gtmask.png").is_file()


def test_openmvs_export_uses_current_mesh_and_same_cameras(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    result = export_openmvs_scene(sample, tmp_path / "openmvs")
    sparse = result.scene_dir / "colmap/sparse"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.view_count == 2
    assert (sparse / "cameras.txt").is_file()
    assert (sparse / "images.txt").is_file()
    assert (result.scene_dir / "colmap/images/00000001.png").is_file()
    assert "<colmap_dir>/sparse" in metadata["interface_command_contract"]
    assert "initial_current.ply" in metadata["interface_command_contract"]
    assert metadata["forbidden_fields_consumed"] == []


def test_nvdiffrec_export_uses_exact_cameras_and_current_mesh(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    result = export_nvdiffrec_scene(sample, tmp_path / "nvdiffrec")
    transforms = json.loads(
        (result.scene_dir / "transforms_train.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    exported = load_mesh(result.initial_obj)
    obj_text = result.initial_obj.read_text(encoding="utf-8")

    assert np.allclose(exported.vertices, sample["vertices"].numpy())
    assert np.array_equal(exported.faces, sample["faces"].numpy())
    assert "mtllib initial_current.mtl" in obj_text
    assert "usemtl defaultMat" in obj_text
    assert (result.scene_dir / "initial_current.mtl").is_file()
    assert len(transforms["frames"]) == 2
    assert transforms["frames"][0]["intrinsics"] == sample["intrinsics"][0].tolist()
    assert transforms["frames"][0]["resolution_wh"] == [8, 8]
    assert metadata["geometry_path"].startswith("official DLMesh")
    assert metadata["forbidden_fields_consumed"] == []
    assert Image.open(result.scene_dir / "images/0000.png").mode == "RGBA"

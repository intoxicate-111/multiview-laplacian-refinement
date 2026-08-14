import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
import numpy as np

from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample, validate_sample
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset

from .helpers import tiny_sample


def test_prepared_sample_can_be_saved_and_loaded(tmp_path):
    path = tmp_path / "sample.pt"
    save_prepared_sample(tiny_sample(), path)
    loaded = load_prepared_sample(path)
    assert loaded["sample_id"] == "tiny"
    assert loaded["images"].shape == (1, 3, 16, 16)
    assert loaded["laplacian_target"].shape == (4, 3)
    assert loaded["raw_laplacian_target"].shape == (4, 3)
    assert loaded["normalized_laplacian_target"].shape == (4, 3)
    assert loaded["local_edge_length"].shape == (4,)
    assert loaded["valid_scale_mask"].shape == (4,)
    assert loaded["valid_scale_mask"].all()
    torch.testing.assert_close(loaded["local_edge_scale"], loaded["local_edge_length"].square())


def test_legacy_sample_is_upgraded_without_changing_raw_target(tmp_path):
    legacy = tiny_sample()
    original_target = legacy["laplacian_target"].clone()
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)

    loaded = load_prepared_sample(path)

    torch.testing.assert_close(loaded["laplacian_target"], original_target)
    torch.testing.assert_close(loaded["raw_laplacian_target"], original_target)
    assert loaded["metadata"]["edge_scale_definition"] == "square_of_mean_incident_edge_length"
    assert loaded["metadata"]["edge_scale_source"] == "input_prediction_mesh"
    assert loaded["metadata"]["edge_scale_epsilon"] == 1e-12


def test_invalid_shape_has_useful_field_name():
    sample = tiny_sample()
    sample["intrinsics"] = torch.eye(4).unsqueeze(0)
    with pytest.raises(ValueError, match="intrinsics must have shape"):
        validate_sample(sample)


def test_lazy_image_paths_save_without_images_and_load_compatibly(tmp_path):
    root = tmp_path / "dataset"
    prepared_dir = root / "prepared"
    image_dir = root / "models" / "1" / "views" / "images"
    prepared_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image_paths = []
    for index in range(14):
        path = image_dir / f"{index:04d}.png"
        Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8)).save(path)
        image_paths.append(path.relative_to(root).as_posix())

    sample = tiny_sample()
    sample.pop("images")
    sample["image_paths"] = image_paths
    sample["source_image_size"] = [8, 8]
    sample["prepared_image_size"] = 16
    sample["prepared_storage_format"] = "lazy_image_paths_v1"
    sample["intrinsics"] = sample["intrinsics"].repeat(14, 1, 1)
    sample["extrinsics"] = sample["extrinsics"].repeat(14, 1, 1)
    sample["visibility"] = sample["visibility"].repeat(14, 1)
    path = prepared_dir / "sample.pt"
    save_prepared_sample(sample, path)

    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert "images" not in raw
    assert raw["image_paths"] == image_paths
    loaded = load_prepared_sample(path)
    assert loaded["images"].shape == (14, 3, 16, 16)
    assert loaded["images"].dtype == torch.float32
    assert torch.all((loaded["images"] >= 0) & (loaded["images"] <= 1))


def test_lazy_image_loader_can_preserve_uint8_until_device_transfer(tmp_path):
    from mlr.learned_laplacian.sample_io import load_and_resize_images

    image_path = tmp_path / "view.png"
    pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
    Image.fromarray(pixels).save(image_path)

    images, _ = load_and_resize_images(
        [image_path],
        8,
        dtype=torch.uint8,
    )

    assert images.dtype == torch.uint8
    assert images.shape == (1, 3, 8, 8)
    assert torch.all(images == 127)


def test_lazy_manifest_paths_resolve_from_manifest_root_across_working_directories(tmp_path):
    root = tmp_path / "dataset"
    prepared_dir = root / "artifacts" / "prepared"
    image_dir = root / "images"
    prepared_dir.mkdir(parents=True)
    image_dir.mkdir()
    image_path = image_dir / "view.png"
    Image.fromarray(np.full((6, 10, 3), 127, dtype=np.uint8)).save(image_path)

    sample = tiny_sample()
    sample.pop("images")
    sample["image_paths"] = ["images/view.png"]
    sample["source_image_size"] = [10, 6]
    sample["prepared_image_size"] = 12
    sample["prepared_storage_format"] = "lazy_image_paths_v1"
    sample_path = prepared_dir / "sample.pt"
    save_prepared_sample(sample, sample_path)
    manifest_path = root / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"samples": [{"path": "artifacts/prepared/sample.pt", "split": "train", "sample_id": "tiny"}]}
        ),
        encoding="utf-8",
    )

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    previous_cwd = Path.cwd()
    try:
        os.chdir(other_cwd)
        dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
        loaded = dataset[0]
    finally:
        os.chdir(previous_cwd)

    assert loaded["images"].shape == (1, 3, 12, 12)
    assert loaded["_dataset_root"] == str(root.resolve())


def test_lazy_images_can_be_remapped_to_node_local_storage(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    prepared_dir = root / "prepared"
    source_images = root / "shared-rgb"
    local_images = tmp_path / "node-local-rgb"
    prepared_dir.mkdir(parents=True)
    source_images.mkdir()
    local_images.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        source_images / "view.png"
    )
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(
        local_images / "view.png"
    )

    sample = tiny_sample()
    sample.pop("images")
    sample["image_paths"] = ["shared-rgb/view.png"]
    sample["source_image_size"] = [8, 8]
    sample["prepared_image_size"] = 8
    sample["prepared_storage_format"] = "lazy_image_paths_v1"
    path = prepared_dir / "sample.pt"
    save_prepared_sample(sample, path)

    monkeypatch.setenv("MLR_IMAGE_PATH_REMAP_FROM", str(source_images))
    monkeypatch.setenv("MLR_IMAGE_PATH_REMAP_TO", str(local_images))
    loaded = load_prepared_sample(path)

    assert torch.all(loaded["images"] == 1)

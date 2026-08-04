import json
import math

import numpy as np
import pytest
import torch

from mlr.laplacian import compute_laplacian_coordinates
from mlr.learned_laplacian.dataset import save_prepared_sample
from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian.multi_trainer import train_multi_object

from .helpers import tiny_sample


def _triangle_sample(sample_id: str) -> dict:
    sample = tiny_sample()
    vertices = sample["vertices"][:3].clone()
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target_positions = vertices.clone()
    target_positions[:, 2] += torch.tensor([0.03, -0.02, 0.01])
    sample.update(
        {
            "sample_id": sample_id,
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": torch.nn.functional.normalize(
                vertices - vertices.mean(dim=0), dim=-1
            ),
            "initial_laplacian": torch.from_numpy(
                compute_laplacian_coordinates(vertices.numpy(), faces.numpy(), "uniform")
            ).float(),
            "laplacian_target": torch.from_numpy(
                compute_laplacian_coordinates(
                    target_positions.numpy(), faces.numpy(), "uniform"
                )
            ).float(),
            "target_confidence": torch.ones(3),
            "visibility": torch.ones((1, 3), dtype=torch.bool),
            "target_positions": target_positions,
            "gt_vertices": target_positions,
            "gt_faces": faces,
        }
    )
    return sample


def _multi_config() -> dict:
    return {
        "seed": 11,
        "device": "cpu",
        "input_mode": "coarse_only",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {
            "method": "square_of_mean_incident_edge_length",
            "epsilon": 1e-12,
            "clip_max_norm": None,
        },
        "image_encoder": {"feature_dim": 8},
        "model": {"hidden_dim": 16, "num_graph_layers": 1, "dropout": 0.0},
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "multi_object_training": {
            "epochs": 4,
            "gradient_accumulation_meshes": 2,
            "shuffle": True,
            "validation_every_epochs": 2,
            "checkpoint_every_epochs": 0,
        },
    }


def test_manifest_lazily_loads_variable_topology_splits(tmp_path):
    train = tiny_sample()
    train["sample_id"] = "tetra_train"
    validation = _triangle_sample("triangle_validation")
    save_prepared_sample(train, tmp_path / "train.pt")
    save_prepared_sample(validation, tmp_path / "validation.pt")
    manifest = {
        "samples": [
            {"sample_id": "tetra_train", "path": "train.pt", "split": "train"},
            {
                "sample_id": "triangle_validation",
                "path": "validation.pt",
                "split": "validation",
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")

    assert train_dataset.sample_ids == ("tetra_train",)
    assert validation_dataset.sample_ids == ("triangle_validation",)
    assert train_dataset[0]["vertices"].shape[0] == 4
    assert validation_dataset[0]["vertices"].shape[0] == 3
    validate_disjoint_splits(train_dataset, validation_dataset)


def test_manifest_rejects_train_validation_path_leakage(tmp_path):
    sample = tiny_sample()
    save_prepared_sample(sample, tmp_path / "shared.pt")
    manifest = {
        "samples": [
            {"path": "shared.pt", "split": "train"},
            {"path": "shared.pt", "split": "validation"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")

    with pytest.raises(ValueError, match="appears in both"):
        validate_disjoint_splits(train_dataset, validation_dataset)


def test_shared_model_trains_across_different_mesh_topologies(tmp_path):
    tetra = tiny_sample()
    tetra["sample_id"] = "tetra_train"
    triangle = _triangle_sample("triangle_train")
    validation = _triangle_sample("triangle_validation")

    result = train_multi_object(
        [tetra, triangle],
        [validation],
        _multi_config(),
        output_dir=tmp_path,
        progress=False,
    )

    assert result.optimizer_steps == 4
    assert result.target_mode == "edge_scale_normalized_laplacian"
    assert 1 <= result.best_epoch <= 4
    assert math.isfinite(result.best_selection_loss)
    assert math.isfinite(result.final_train_loss)
    assert math.isfinite(result.final_validation_loss)
    assert result.per_object_metrics["train"]["tetra_train"]["vertex_count"] == 4
    assert result.per_object_metrics["train"]["triangle_train"]["vertex_count"] == 3
    assert result.per_object_metrics["validation"]["triangle_validation"]["face_count"] == 1
    for split_metrics in result.per_object_metrics.values():
        for metrics in split_metrics.values():
            assert all(
                math.isfinite(value)
                for value in metrics["recovered_raw_space"].values()
            )
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "predictions" / "train" / "tetra_train_raw_delta.npy").is_file()
    triangle_prediction = np.load(
        tmp_path / "predictions" / "train" / "triangle_train_raw_delta.npy"
    )
    assert triangle_prediction.shape == (3, 3)

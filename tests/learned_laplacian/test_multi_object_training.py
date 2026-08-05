import json
import math

import numpy as np
import pytest
import torch

from mlr.laplacian import compute_laplacian_coordinates
from mlr.learned_laplacian import dataset as dataset_module
from mlr.learned_laplacian.dataset import save_prepared_sample
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian import multi_dataset as multi_dataset_module
from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian import multi_trainer
from mlr.learned_laplacian.multi_trainer import (
    _build_lr_scheduler,
    _prepare_object_static,
    train_multi_object,
)
from mlr.learned_laplacian.target_scaling import normalize_laplacian_by_edge_scale

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


def test_manifest_dataset_loads_each_file_only_once(tmp_path, monkeypatch):
    train = tiny_sample()
    train["sample_id"] = "train"
    validation = _triangle_sample("validation")
    save_prepared_sample(train, tmp_path / "train.pt")
    save_prepared_sample(validation, tmp_path / "validation.pt")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"path": "train.pt", "split": "train"},
                    {"path": "validation.pt", "split": "validation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    original_load = multi_dataset_module.load_prepared_sample
    loaded_paths = []

    def counted_load(path, **kwargs):
        loaded_paths.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(multi_dataset_module, "load_prepared_sample", counted_load)
    validate_disjoint_splits(train_dataset, validation_dataset)
    train_samples = tuple(train_dataset)
    validation_samples = tuple(validation_dataset)

    assert len(loaded_paths) == 2
    assert train_samples[0] is train_dataset[0]
    assert validation_samples[0] is validation_dataset[0]


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


def test_static_preparation_runs_once_per_sample_across_epochs(monkeypatch):
    samples = [tiny_sample(), _triangle_sample("triangle_train")]
    samples[0]["sample_id"] = "tetra_train"
    validation = [_triangle_sample("triangle_validation")]
    calls = {"validate": 0, "edges": 0, "normalize": 0}
    original_validate = multi_trainer.validate_sample
    original_edges = dataset_module.faces_to_edge_index
    original_normalize = dataset_module.normalize_laplacian_by_edge_scale

    def counted_validate(sample):
        calls["validate"] += 1
        return original_validate(sample)

    def counted_edges(faces, num_vertices=None):
        calls["edges"] += 1
        return original_edges(faces, num_vertices)

    def counted_normalize(*args, **kwargs):
        calls["normalize"] += 1
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(multi_trainer, "validate_sample", counted_validate)
    monkeypatch.setattr(dataset_module, "faces_to_edge_index", counted_edges)
    monkeypatch.setattr(dataset_module, "normalize_laplacian_by_edge_scale", counted_normalize)

    train_multi_object(samples, validation, _multi_config(), progress=False)

    assert calls == {"validate": 3, "edges": 3, "normalize": 3}


def test_static_prepared_fields_match_reference_calculation():
    config = _multi_config()
    config["target_scaling"]["clip_max_norm"] = 0.05
    sample = _triangle_sample("prepared_triangle")
    prepared = _prepare_object_static(sample, config)
    expected_edges = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
    expected_degree = sample["vertices"].new_zeros((3, 1))
    expected_degree.index_add_(
        0,
        expected_edges[1],
        torch.ones((expected_edges.shape[1], 1)),
    )
    expected_target = normalize_laplacian_by_edge_scale(
        prepared.sample["raw_laplacian_target"],
        prepared.sample["local_edge_length"],
        eps=config["target_scaling"]["epsilon"],
        valid_scale_mask=prepared.sample["valid_scale_mask"],
    )
    magnitudes = torch.linalg.vector_norm(expected_target, dim=-1)
    expected_clipped = magnitudes > 0.05
    expected_target = expected_target * (
        0.05 / magnitudes.clamp_min(1e-12)
    ).clamp_max(1.0).unsqueeze(-1)

    assert torch.equal(prepared.sample["edge_index"], expected_edges)
    assert torch.equal(prepared.sample["vertex_degree"], expected_degree)
    assert torch.equal(prepared.sample["valid_scale_mask"], torch.ones(3, dtype=torch.bool))
    assert torch.allclose(prepared.training_target, expected_target)
    assert prepared.clipped_target_vertices == int(expected_clipped.sum())


def test_device_cache_switch_preserves_cpu_training_results():
    def run(cache_on_device: bool):
        config = _multi_config()
        config["multi_object_training"]["cache_prepared_samples_on_device"] = cache_on_device
        return train_multi_object(
            [_triangle_sample("train")],
            [_triangle_sample("validation")],
            config,
            progress=False,
        )

    cached = run(True)
    streamed = run(False)

    assert [item["train_loss"] for item in cached.history] == pytest.approx(
        [item["train_loss"] for item in streamed.history]
    )
    assert [item["validation_loss"] for item in cached.history] == pytest.approx(
        [item["validation_loss"] for item in streamed.history]
    )
    assert cached.best_epoch == streamed.best_epoch
    assert cached.best_selection_loss == pytest.approx(streamed.best_selection_loss)


def test_validation_schedule_and_timing_metrics(tmp_path):
    config = _multi_config()
    config["multi_object_training"].update(
        {"epochs": 5, "validation_every_epochs": 3}
    )
    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        output_dir=tmp_path,
        progress=False,
    )

    validated_epochs = [
        item["epoch"] for item in result.history if item["validation_loss"] is not None
    ]
    assert validated_epochs == [1, 3, 5]
    assert all(item["train_seconds"] >= 0 for item in result.history)
    assert result.static_preparation_seconds >= 0
    assert result.device_cache_seconds >= 0
    assert result.mean_epoch_train_seconds >= 0
    assert result.mean_validation_seconds >= 0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mean_epoch_train_seconds"] == result.mean_epoch_train_seconds
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=False)
    assert set(checkpoint) == {
        "epoch",
        "train_loss",
        "validation_loss",
        "model_config",
        "model_state_dict",
        "optimizer_state_dict",
        "experiment_config",
        "train_meshes",
        "validation_meshes",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("cache_on_device", [True, False])
def test_cuda_device_cache_paths(cache_on_device):
    config = _multi_config()
    config["device"] = "cuda"
    config["multi_object_training"].update(
        {"epochs": 1, "cache_prepared_samples_on_device": cache_on_device}
    )

    result = train_multi_object(
        [_triangle_sample("train")], (), config, progress=False
    )

    assert result.device == "cuda"
    assert math.isfinite(result.final_train_loss)


def _use_constant_validation(monkeypatch, loss=1.0):
    calls = []

    def constant_validation(*args, **kwargs):
        calls.append(loss)
        return loss, {}

    monkeypatch.setattr(multi_trainer, "_evaluate_dataset", constant_validation)
    return calls


def test_scheduler_disabled_or_omitted_preserves_training(monkeypatch):
    _use_constant_validation(monkeypatch)

    def run(scheduler_config):
        config = _multi_config()
        if scheduler_config is not None:
            config["training"]["lr_scheduler"] = scheduler_config
        return train_multi_object(
            [_triangle_sample("train")],
            [_triangle_sample("validation")],
            config,
            progress=False,
        )

    omitted = run(None)
    disabled = run({"type": "none"})

    assert [item["train_loss"] for item in omitted.history] == pytest.approx(
        [item["train_loss"] for item in disabled.history]
    )
    assert omitted.optimizer_steps == disabled.optimizer_steps
    assert omitted.best_epoch == disabled.best_epoch
    assert omitted.best_selection_loss == disabled.best_selection_loss
    assert {item["learning_rate"] for item in omitted.history} == {0.01}
    assert {item["learning_rate"] for item in disabled.history} == {0.01}
    assert omitted.lr_scheduler_type == disabled.lr_scheduler_type == "none"


def test_reduce_on_plateau_uses_validation_patience(monkeypatch):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience_validations": 1,
                "threshold": 0.0,
                "threshold_mode": "abs",
                "cooldown_validations": 0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 4, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert [item["learning_rate"] for item in result.history] == pytest.approx(
        [0.001, 0.001, 0.0005, 0.0005]
    )
    assert result.lr_reduction_count == 1


def test_scheduler_steps_only_on_validation_epochs(monkeypatch):
    validation_calls = _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience_validations": 0,
                "threshold": 0.0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 6, "validation_every_epochs": 5}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    assert [item["validation_loss"] is not None for item in result.history] == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert [item["learning_rate"] for item in result.history] == pytest.approx(
        [0.001, 0.001, 0.001, 0.001, 0.0005, 0.00025]
    )
    # Three scheduled validations plus the final train and validation evaluations.
    assert len(validation_calls) == 5


def test_scheduler_respects_minimum_learning_rate(monkeypatch):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"].update(
        {
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.1,
                "patience_validations": 0,
                "threshold": 0.0,
                "cooldown_validations": 0,
                "min_lr": 1e-6,
            },
        }
    )
    config["multi_object_training"].update(
        {"epochs": 8, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        progress=False,
    )

    learning_rates = [item["learning_rate"] for item in result.history]
    assert min(learning_rates) == pytest.approx(1e-6)
    assert result.final_learning_rate == pytest.approx(1e-6)
    assert all(rate >= 1e-6 for rate in learning_rates)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"factor": 0.0}, "factor"),
        ({"factor": 1.0}, "factor"),
        ({"patience_validations": -1}, "patience_validations"),
        ({"threshold": -1.0}, "threshold"),
        ({"threshold_mode": "invalid"}, "threshold_mode"),
        ({"cooldown_validations": -1}, "cooldown_validations"),
        ({"min_lr": -1.0}, "min_lr"),
        ({"type": "cosine"}, "Unsupported lr_scheduler type"),
    ],
)
def test_scheduler_rejects_invalid_configuration(override, message):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler_config = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience_validations": 1,
        "threshold": 0.0,
        "threshold_mode": "abs",
        "cooldown_validations": 0,
        "min_lr": 1e-6,
    }
    scheduler_config.update(override)

    with pytest.raises(ValueError, match=message):
        _build_lr_scheduler(optimizer, {"lr_scheduler": scheduler_config})


def test_scheduler_history_metrics_and_reduction_log(tmp_path, monkeypatch, capsys):
    _use_constant_validation(monkeypatch)
    config = _multi_config()
    config["training"]["lr_scheduler"] = {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience_validations": 0,
        "threshold": 1.0,
        "threshold_mode": "abs",
        "cooldown_validations": 0,
        "min_lr": 1e-6,
    }
    config["multi_object_training"].update(
        {"epochs": 3, "validation_every_epochs": 1}
    )

    result = train_multi_object(
        [_triangle_sample("train")],
        [_triangle_sample("validation")],
        config,
        output_dir=tmp_path,
        progress=True,
    )

    output = capsys.readouterr().out
    assert "lr=5.00000000e-03" in output
    assert "learning rate reduced: 1.00000000e-02 -> 5.00000000e-03" in output
    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert all("learning_rate" in item for item in history)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["initial_learning_rate"] == 0.01
    assert metrics["final_learning_rate"] == pytest.approx(0.0025)
    assert metrics["lr_scheduler_type"] == "reduce_on_plateau"
    assert metrics["lr_reduction_count"] == 2
    assert result.lr_reduction_count == 2

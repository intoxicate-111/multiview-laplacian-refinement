from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mlr.learned_laplacian.controlled_displacement import (
    CURRENT_GRAPH_LAPLACIAN,
    DIRECT_VERTEX_DISPLACEMENT,
    displacement_target,
    prediction_semantics,
    recover_direct_displacement,
)
from mlr.learned_laplacian.multi_trainer import (
    _prepare_object_static,
    train_multi_object,
)

from .helpers import tiny_sample


def _config() -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "input_mode": "coarse_only",
        "prediction_semantics": DIRECT_VERTEX_DISPLACEMENT,
        "target_mode": "raw_laplacian",
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
            "prediction_loss_space": "output_representation",
            "gradient_clip_norm": 1.0,
        },
        "multi_object_training": {
            "epochs": 1,
            "max_optimizer_steps": 1,
            "gradient_accumulation_meshes": 1,
            "shuffle": False,
            "validation_every_epochs": 1,
            "checkpoint_every_epochs": 0,
        },
    }


def _sample(sample_id: str) -> dict:
    sample = tiny_sample()
    sample["sample_id"] = sample_id
    sample["target_positions"] = sample["vertices"] + torch.tensor(
        [0.01, -0.02, 0.03]
    )
    sample["gt_vertices"] = sample["target_positions"].clone()
    sample["gt_faces"] = sample["faces"].clone()
    return sample


def test_legacy_configs_default_to_current_graph_laplacian() -> None:
    assert prediction_semantics({}) == CURRENT_GRAPH_LAPLACIAN
    with pytest.raises(ValueError, match="prediction_semantics"):
        prediction_semantics({"prediction_semantics": "unknown"})


def test_displacement_target_and_direct_recovery_are_exact() -> None:
    sample = _sample("exact")
    expected = sample["target_positions"] - sample["vertices"]
    assert torch.equal(displacement_target(sample), expected)
    assert torch.allclose(
        recover_direct_displacement(sample["vertices"], expected),
        sample["target_positions"],
    )
    current_np = sample["vertices"].numpy()
    expected_np = expected.numpy()
    assert np.allclose(
        recover_direct_displacement(current_np, expected_np),
        sample["target_positions"].numpy(),
    )


def test_preparation_uses_displacement_label_without_gt_model_input() -> None:
    sample = _sample("prepared")
    prepared = _prepare_object_static(sample, _config())
    assert torch.allclose(
        prepared.training_target,
        sample["target_positions"] - sample["vertices"],
    )
    assert prepared.raw_target is None
    assert "target_positions" not in prepared.sample
    assert "gt_vertices" not in prepared.sample
    assert "gt_faces" not in prepared.sample


def test_displacement_mode_trains_and_validates_with_same_backbone(tmp_path) -> None:
    result = train_multi_object(
        [_sample("train")],
        [_sample("validation")],
        _config(),
        output_dir=tmp_path,
        progress=False,
    )
    assert result.prediction_semantics == DIRECT_VERTEX_DISPLACEMENT
    assert result.optimizer_steps == 1
    assert math.isfinite(result.final_train_loss)
    assert result.final_validation_loss is not None
    assert math.isfinite(result.final_validation_loss)
    assert (tmp_path / "checkpoint_latest.pt").is_file()

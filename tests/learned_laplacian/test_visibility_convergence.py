import copy
import csv
import json

import numpy as np
import pytest
import torch

from mlr.learned_laplacian.image_ablation import _condition_sample
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.visibility_convergence import (
    latest_resume_checkpoint,
    normalize_checkpoint_steps,
    validate_expanded_sample_ids,
    validate_summary_consistency,
    validate_visibility_shape,
    visibility_group_masks,
)

from .helpers import tiny_sample


def _config(max_steps: int, checkpoint_steps: list[int]) -> dict:
    return {
        "seed": 17,
        "device": "cpu",
        "input_mode": "coarse_only",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {
            "method": "square_of_mean_incident_edge_length",
            "epsilon": 1e-12,
            "clip_max_norm": None,
        },
        "image_encoder": {"feature_dim": 8},
        "model": {"hidden_dim": 8, "num_graph_layers": 1, "dropout": 0.0},
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.1,
            "gradient_clip_norm": 1.0,
            "lr_scheduler": {"type": "none"},
        },
        "multi_object_training": {
            "epochs": 8,
            "max_optimizer_steps": max_steps,
            "checkpoint_optimizer_steps": checkpoint_steps,
            "gradient_accumulation_meshes": 2,
            "shuffle": True,
            "validation_every_epochs": 1,
            "checkpoint_every_epochs": 0,
            "early_stopping": {"enabled": False},
        },
    }


def _samples():
    first = tiny_sample()
    first["sample_id"] = "train_a"
    second = copy.deepcopy(tiny_sample())
    second["sample_id"] = "train_b"
    validation = copy.deepcopy(tiny_sample())
    validation["sample_id"] = "validation"
    return [first, second], [validation]


def test_checkpoint_step_schedule_is_exact():
    assert normalize_checkpoint_steps([2000, 0, 100, 100, 250], 2000) == (
        0,
        100,
        250,
        2000,
    )
    with pytest.raises(ValueError, match="include step 0"):
        normalize_checkpoint_steps([100, 2000], 2000)


def test_visibility_groups_and_views_vertices_shape():
    visibility = torch.zeros((14, 5), dtype=torch.bool)
    validate_visibility_shape(visibility, 5)
    with pytest.raises(ValueError, match=r"\[views, vertices\]"):
        validate_visibility_shape(torch.zeros((5, 14), dtype=torch.bool), 5)
    groups = visibility_group_masks(np.array([0, 1, 2, 3, 7]))
    assert [int(groups[name].sum()) for name in groups] == [1, 1, 1, 1, 1]


def test_rgb_only_ablation_keeps_query_graph_target_and_visibility_fixed():
    base = {
        "images": torch.arange(24).reshape(4, 3, 2, 1),
        "vertices": torch.randn(6, 3),
        "query_positions": torch.randn(6, 3),
        "faces": torch.tensor([[0, 1, 2], [2, 3, 4]]),
        "laplacian_target": torch.randn(6, 3),
        "visibility": torch.randint(0, 2, (4, 6), dtype=torch.bool),
        "intrinsics": torch.randn(4, 3, 3),
        "extrinsics": torch.randn(4, 4, 4),
    }
    permutation = torch.tensor([2, 0, 3, 1])
    changed = _condition_sample(
        base, torch.zeros_like(base["images"]), permutation, "shuffled_images"
    )
    assert torch.equal(changed["images"], base["images"].index_select(0, permutation))
    for key in (
        "vertices",
        "query_positions",
        "faces",
        "laplacian_target",
        "visibility",
        "intrinsics",
        "extrinsics",
    ):
        assert changed[key] is base[key]


def test_expanded_mesh_ids_are_fixed():
    validate_expanded_sample_ids(("a", "b"), ("a", "b"))
    with pytest.raises(ValueError, match="mesh IDs changed"):
        validate_expanded_sample_ids(("b", "a"), ("a", "b"))


def test_summary_csv_and_json_must_have_identical_checkpoint_steps(tmp_path):
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["optimizer_step"])
        writer.writeheader()
        writer.writerows([{"optimizer_step": 0}, {"optimizer_step": 100}])
    json_path.write_text(
        json.dumps({"checkpoints": [{"optimizer_step": 0}, {"optimizer_step": 100}]}),
        encoding="utf-8",
    )
    validate_summary_consistency(csv_path, json_path)
    json_path.write_text(
        json.dumps({"checkpoints": [{"optimizer_step": 0}, {"optimizer_step": 250}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagree"):
        validate_summary_consistency(csv_path, json_path)


def test_optimizer_step_resume_matches_uninterrupted_training(tmp_path):
    train, validation = _samples()
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    train_multi_object(
        train,
        validation,
        _config(4, [0, 2, 4]),
        output_dir=continuous_dir,
        progress=False,
    )
    train_multi_object(
        train,
        validation,
        _config(2, [0, 2]),
        output_dir=resumed_dir,
        progress=False,
    )
    resume_checkpoint = latest_resume_checkpoint(
        resumed_dir / "checkpoints", [0, 2, 4], 4
    )
    assert resume_checkpoint is not None
    train_multi_object(
        train,
        validation,
        _config(4, [0, 2, 4]),
        output_dir=resumed_dir,
        progress=False,
        resume_checkpoint=resume_checkpoint,
    )
    continuous = torch.load(
        continuous_dir / "checkpoints" / "checkpoint_step_000004.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed = torch.load(
        resumed_dir / "checkpoints" / "checkpoint_step_000004.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert continuous["optimizer_steps"] == resumed["optimizer_steps"] == 4
    for key, value in continuous["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][key], rtol=0, atol=0)

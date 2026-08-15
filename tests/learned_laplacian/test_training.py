import math

import torch

from mlr.learned_laplacian.losses import weighted_robust_laplacian_loss
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.trainer import load_checkpoint, train_single_object

from .helpers import tiny_sample


def _config():
    return {
        "seed": 7,
        "device": "cpu",
        "input_mode": "coarse_only",
        "image_encoder": {"feature_dim": 8},
        "model": {"hidden_dim": 32, "num_graph_layers": 2, "dropout": 0.0},
        "training": {
            "steps": 60,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.1,
            "gradient_clip_norm": 1.0,
            "log_every": 10,
            "checkpoint_every": 0,
        },
    }


def test_cpu_training_decreases_loss_and_checkpoint_round_trips(tmp_path):
    result = train_single_object(tiny_sample(), _config(), tmp_path, progress=False)
    assert result.best_loss < result.initial_loss * 0.5
    assert result.prediction_metrics["mse"] >= 0.0
    checkpoint = torch.load(tmp_path / "best.pt", weights_only=False)
    architecture = checkpoint["model_config"]
    restored = LearnedLaplacianModel(**architecture)
    loaded = load_checkpoint(tmp_path / "best.pt", restored)
    assert loaded["step"] == result.best_step
    for expected, actual in zip(result.model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_edge_scale_normalized_target_mode_trains_and_reports_mode(tmp_path):
    config = _config()
    config["training"]["steps"] = 5
    config["target_mode"] = "edge_scale_normalized_laplacian"
    config["target_scaling"] = {"epsilon": 1e-12, "clip_max_norm": None}

    result = train_single_object(tiny_sample(), config, tmp_path, progress=False)

    assert result.target_mode == "edge_scale_normalized_laplacian"
    assert result.target_scaling_epsilon == 1e-12
    assert result.clipped_target_vertices == 0
    assert math.isfinite(result.best_loss)
    assert all(math.isfinite(value) for value in result.prediction_metrics.values())


def test_target_magnitude_weighting_emphasizes_large_target_errors():
    target = torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    prediction = target.clone()
    prediction[1] = 0.0
    confidence = torch.ones(2)
    base = weighted_robust_laplacian_loss(
        prediction, target, confidence, huber_delta=0.01
    )
    weighted = weighted_robust_laplacian_loss(
        prediction,
        target,
        confidence,
        huber_delta=0.01,
        target_magnitude_weight_lambda=4.0,
    )
    assert weighted > base


def test_target_magnitude_weighting_rejects_negative_lambda():
    value = torch.zeros((1, 3))
    try:
        weighted_robust_laplacian_loss(
            value,
            value,
            torch.ones(1),
            target_magnitude_weight_lambda=-1.0,
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative target magnitude weight must fail")


def test_raw_mse_loss_is_mean_squared_component_error():
    prediction = torch.tensor([[1.0, 2.0, 3.0], [3.0, 0.0, -1.0]])
    target = torch.zeros_like(prediction)
    confidence = torch.tensor([1.0, 0.5])

    actual = weighted_robust_laplacian_loss(
        prediction,
        target,
        confidence,
        loss_type="mse",
    )
    per_vertex = (prediction - target).square().mean(dim=-1)
    expected = (confidence * per_vertex).sum() / confidence.sum()

    torch.testing.assert_close(actual, expected)

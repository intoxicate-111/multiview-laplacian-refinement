import torch

from mlr.learned_laplacian.losses import (
    confidence_calibration_metrics,
    confidence_reliability_loss,
)
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.sample_io import prepare_gt_query_sample_from_prepared

from .helpers import tiny_sample


def test_optional_confidence_head_is_bounded_and_keeps_three_vector_output():
    sample = tiny_sample()
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        predict_confidence=True,
    )
    output = model(sample)
    assert output.predicted_laplacian.shape == (4, 3)
    assert output.delta_hat_prediction is output.predicted_laplacian
    assert output.confidence_prediction is not None
    assert output.confidence_prediction.shape == (4,)
    assert torch.all((output.confidence_prediction >= 0) & (output.confidence_prediction <= 1))


def test_confidence_loss_penalizes_trivial_zero_confidence():
    prediction = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.zeros_like(prediction)
    weight = torch.ones(2)
    almost_zero = confidence_reliability_loss(
        torch.full((2,), 1e-4), prediction, target, weight, regularizer=0.01
    )
    moderate = confidence_reliability_loss(
        torch.tensor([0.9, 0.1]), prediction, target, weight, regularizer=0.01
    )
    assert moderate < almost_zero


def test_confidence_calibration_correlation_detects_inverse_error_order():
    target = torch.zeros((5, 3))
    prediction = torch.tensor([[5.0, 0, 0], [4.0, 0, 0], [3.0, 0, 0], [2.0, 0, 0], [1.0, 0, 0]])
    metrics = confidence_calibration_metrics(
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), prediction, target
    )
    assert metrics["correlation_with_negative_error"] > 0.99
    assert len(metrics["bins"]) == 5


def test_multi_mesh_training_logs_confidence_without_reweighting_primary_loss(tmp_path):
    train = prepare_gt_query_sample_from_prepared(tiny_sample())
    train["sample_id"] = "confidence_train"
    validation = prepare_gt_query_sample_from_prepared(tiny_sample())
    validation["sample_id"] = "confidence_validation"
    config = {
        "seed": 3,
        "device": "cpu",
        "input_mode": "coarse_plus_multiview",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {"epsilon": 1e-12},
        "query_training": {
            "enabled": True,
            "exact_fraction": 0.25,
            "normal_std_h": 0.0003,
            "tangent_std_h": 0.0003,
            "max_offset_h": 0.001,
            "apply_to_validation": True,
            "zero_initial_laplacian": True,
        },
        "image_encoder": {"feature_dim": 8},
        "model": {
            "hidden_dim": 16,
            "num_graph_layers": 1,
            "geometry_mode": "query_fourier",
            "position_encoding": {"num_frequencies": 2, "include_input": True},
        },
        "confidence": {
            "enabled": True,
            "loss_weight": 1.0,
            "regularizer": 0.01,
            "minimum_confidence": 1e-4,
            "quantile_bins": 2,
        },
        "training": {
            "learning_rate": 0.001,
            "loss": "huber",
            "huber_delta": 0.01,
        },
        "multi_object_training": {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "validation_every_epochs": 1,
            "shuffle": False,
        },
    }
    result = train_multi_object(
        [train], [validation], config, output_dir=tmp_path, progress=False
    )
    record = result.history[0]
    assert record["train_normalized_laplacian_loss"] == record["train_loss"]
    assert record["train_objective"] != record["train_loss"]
    assert record["train_confidence_loss"] is not None
    assert record["validation_mean_confidence"] is not None
    assert (tmp_path / "checkpoint_latest.pt").is_file()
    assert (tmp_path / "checkpoint_best.pt").is_file()

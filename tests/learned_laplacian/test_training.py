import torch

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

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .dataset import move_sample_to_device, validate_sample
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .model import LearnedLaplacianModel


@dataclass
class TrainingResult:
    model: LearnedLaplacianModel
    history: list[dict[str, float]]
    initial_loss: float
    final_loss: float
    best_loss: float
    best_step: int
    prediction_metrics: dict[str, float]
    device: str


def train_single_object(
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    device_override: str | None = None,
    input_mode_override: str | None = None,
    zero_images: bool = False,
    progress: bool = True,
) -> TrainingResult:
    sample = validate_sample(sample)
    seed = int(config.get("seed", 7))
    _seed_everything(seed)
    requested_device = device_override or str(config.get("device", "cpu"))
    device = _resolve_device(requested_device)
    device_sample = move_sample_to_device(sample, device)

    image_config = config.get("image_encoder", {})
    model_config = config.get("model", {})
    input_mode = input_mode_override or str(config.get("input_mode", "coarse_plus_multiview"))
    model = LearnedLaplacianModel(
        image_feature_dim=int(image_config.get("feature_dim", 32)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_graph_layers=int(model_config.get("num_graph_layers", 3)),
        dropout=float(model_config.get("dropout", 0.0)),
        input_mode=input_mode,
        zero_images=zero_images,
    ).to(device)
    training = config.get("training", {})
    steps = int(training.get("steps", 5000))
    if steps < 1:
        raise ValueError("training.steps must be positive.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    log_every = max(int(training.get("log_every", 50)), 1)
    checkpoint_every = max(int(training.get("checkpoint_every", 0)), 0)
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    loss_type = str(training.get("loss", "huber"))
    huber_delta = float(training.get("huber_delta", 0.01))
    charbonnier_epsilon = float(training.get("charbonnier_epsilon", 1e-3))
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    model.train()
    with torch.no_grad():
        initial_prediction = model(device_sample).predicted_laplacian
        initial_tensor = weighted_robust_laplacian_loss(
            initial_prediction,
            device_sample["laplacian_target"],
            device_sample["target_confidence"],
            loss_type=loss_type,
            huber_delta=huber_delta,
            charbonnier_epsilon=charbonnier_epsilon,
        )
    initial_loss = float(initial_tensor.item())
    history = [{"step": 0.0, "loss": initial_loss}]
    best_loss = initial_loss
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    if output_path is not None:
        _save_checkpoint(output_path / "best.pt", model, optimizer, 0, best_loss, config)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(device_sample).predicted_laplacian
        loss = weighted_robust_laplacian_loss(
            prediction,
            device_sample["laplacian_target"],
            device_sample["target_confidence"],
            loss_type=loss_type,
            huber_delta=huber_delta,
            charbonnier_epsilon=charbonnier_epsilon,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training produced a non-finite loss at step {step}.")
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        value = float(loss.detach().item())
        if value < best_loss:
            best_loss = value
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            if output_path is not None:
                _save_checkpoint(output_path / "best.pt", model, optimizer, step, best_loss, config)
        if step == 1 or step == steps or step % log_every == 0:
            history.append({"step": float(step), "loss": value})
            if progress:
                print(f"step={step:05d} loss={value:.8f} best={best_loss:.8f}", flush=True)
        if output_path is not None and checkpoint_every > 0 and step % checkpoint_every == 0:
            _save_checkpoint(
                output_path / f"checkpoint_step_{step:06d}.pt",
                model,
                optimizer,
                step,
                value,
                config,
            )

    final_loss = value
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_prediction = model(device_sample).predicted_laplacian
        metrics = laplacian_prediction_metrics(best_prediction, device_sample["laplacian_target"])
    if output_path is not None:
        (output_path / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    return TrainingResult(
        model=model,
        history=history,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        best_step=best_step,
        prediction_metrics=metrics,
        device=str(device),
    )


def load_checkpoint(
    path: str | Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _save_checkpoint(
    path: Path,
    model: LearnedLaplacianModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "step": int(step),
            "loss": float(loss),
            "model_config": model.architecture_config(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment_config": dict(config),
        },
        path,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

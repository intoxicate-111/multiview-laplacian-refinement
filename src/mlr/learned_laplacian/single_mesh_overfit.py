from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .losses import weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .query_training import apply_query_augmentation, query_augmentation_settings
from .target_scaling import denormalize_laplacian_by_edge_scale


LOSS_VARIANTS = (
    "base_fixed_0.01",
    "base_adaptive",
    "weighted_adaptive",
    "magnitude_direction",
)


def single_mesh_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    variant: str,
    adaptive_delta: float,
    magnitude_weight_lambda: float = 4.0,
    direction_lambda: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Loss alternatives used by the controlled single-mesh experiment."""

    weights = confidence.float().clamp_min(0)
    target_magnitude = torch.linalg.vector_norm(target.float(), dim=-1)
    if variant in {"base_fixed_0.01", "base_adaptive"}:
        delta = 0.01 if variant == "base_fixed_0.01" else adaptive_delta
        per_vertex = F.huber_loss(
            prediction.float(), target.float(), delta=delta, reduction="none"
        ).mean(dim=-1)
    elif variant == "weighted_adaptive":
        mean_magnitude = (weights * target_magnitude).sum() / weights.sum().clamp_min(epsilon)
        magnitude_weights = 1.0 + magnitude_weight_lambda * target_magnitude / (
            mean_magnitude + epsilon
        )
        weights = weights * magnitude_weights
        per_vertex = F.huber_loss(
            prediction.float(), target.float(), delta=adaptive_delta, reduction="none"
        ).mean(dim=-1)
    elif variant == "magnitude_direction":
        prediction_magnitude = torch.linalg.vector_norm(prediction.float(), dim=-1)
        magnitude_loss = F.huber_loss(
            prediction_magnitude,
            target_magnitude,
            delta=adaptive_delta,
            reduction="none",
        )
        stable = target_magnitude > epsilon
        cosine_loss = 1.0 - F.cosine_similarity(
            prediction.float(), target.float(), dim=-1, eps=epsilon
        )
        cosine_loss = torch.where(stable, cosine_loss, torch.zeros_like(cosine_loss))
        per_vertex = magnitude_loss + direction_lambda * cosine_loss
    else:
        raise ValueError(f"Unknown loss variant: {variant}")
    return (weights * per_vertex).sum() / weights.sum().clamp_min(epsilon)


def run_single_mesh_overfit(
    manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    sample_id: str | None = None,
    steps: int = 1000,
    log_every: int = 25,
    device: str = "cuda",
    seed: int = 17,
    learning_rate: float = 1e-3,
    magnitude_weight_lambda: float = 4.0,
    direction_lambda: float = 1.0,
    overwrite: bool = False,
    skip_reconstruction: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Overfit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if steps < 1 or log_every < 1:
        raise ValueError("steps and log_every must be positive.")
    config = _read_json(Path(config_path))
    dataset = PreparedMeshDataset.from_manifest(Path(manifest), split)
    index = _find_sample(dataset, sample_id)
    static = dataset.load_static(index)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    np.random.seed(seed)

    prepared = _prepare_item_for_use(
        _prepare_object_static(static, config),
        config,
        resolved_device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )
    base_sample = dict(prepared.sample)
    target = prepared.training_target.float()
    confidence = base_sample["target_confidence"].float()
    valid = base_sample["valid_scale_mask"].bool() & (confidence > 0)
    target_magnitude = torch.linalg.vector_norm(target[valid], dim=-1)
    adaptive_delta = float(torch.median(target_magnitude).item())
    target_distribution = {
        "magnitude_min": float(target_magnitude.min().item()),
        "magnitude_median": adaptive_delta,
        "magnitude_mean": float(target_magnitude.mean().item()),
        "magnitude_p90": float(torch.quantile(target_magnitude, 0.90).item()),
        "magnitude_p95": float(torch.quantile(target_magnitude, 0.95).item()),
        "magnitude_p99": float(torch.quantile(target_magnitude, 0.99).item()),
        "magnitude_max": float(target_magnitude.max().item()),
        "adaptive_huber_delta": adaptive_delta,
        "adaptive_huber_delta_rule": "median valid normalized target vector magnitude",
    }
    _write_json(output_dir / "target_distribution.json", target_distribution)
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)
    query_settings = query_augmentation_settings(config)
    if not query_settings.enabled:
        raise ValueError("Single-mesh overfit expects query_training.enabled=true.")

    variant_results: dict[str, Any] = {}
    for variant in LOSS_VARIANTS:
        print(f"Training single mesh with {variant}...", flush=True)
        torch.manual_seed(seed)
        model = _build_model(config, None, False).to(resolved_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Adaptive-target losses are O(10-100), unlike the historical
        # Huber(0.01) objective.  The default 65536 scale can overflow their
        # first backward pass before GradScaler has adapted, making the
        # controlled variants appear frozen.
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled, init_scale=128.0, growth_interval=2000
        )
        history: list[dict[str, Any]] = []
        start = time.perf_counter()
        model.train()
        for step in range(1, steps + 1):
            sample = apply_query_augmentation(
                base_sample, query_settings, base_seed=seed, epoch=step
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved_device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                prediction = model(sample).predicted_laplacian
            objective = single_mesh_loss(
                prediction,
                target,
                confidence,
                variant=variant,
                adaptive_delta=adaptive_delta,
                magnitude_weight_lambda=magnitude_weight_lambda,
                direction_lambda=direction_lambda,
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite {variant} loss at step {step}.")
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % log_every == 0 or step == steps:
                evaluation = _evaluate_model(
                    model,
                    base_sample,
                    target,
                    confidence,
                    valid,
                    query_settings,
                    resolved_device,
                    amp_enabled,
                    amp_dtype,
                    seed,
                )
                row = {
                    "step": step,
                    "training_objective": float(objective.detach().item()),
                    **evaluation,
                }
                history.append(row)
                print(
                    f"  step={step:04d} objective={row['training_objective']:.6g} "
                    f"exact_huber={row['exact_huber_0.01']:.6g} "
                    f"ratio={row['exact_magnitude_ratio']:.4f} "
                    f"high_cos={row['exact_high_cosine']:.4f}",
                    flush=True,
                )
                model.train()
        runtime = time.perf_counter() - start
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = variant_dir / "final.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "variant": variant,
                "steps": steps,
                "sample_id": str(base_sample["sample_id"]),
                "adaptive_huber_delta": adaptive_delta,
                "config": config,
            },
            checkpoint_path,
        )
        _write_history(variant_dir / "history.csv", history)
        _write_json(variant_dir / "history.json", {"history": history})
        _plot_history(variant_dir / "curves.png", history, variant)
        final_prediction = _predict_exact(
            model, base_sample, resolved_device, amp_enabled, amp_dtype
        )
        np.save(variant_dir / "exact_prediction.npy", final_prediction)
        if skip_reconstruction:
            reconstruction = {"skipped": True}
        else:
            normalized_prediction = torch.from_numpy(final_prediction)
            raw_prediction = denormalize_laplacian_by_edge_scale(
                normalized_prediction, static["local_edge_length"]
            )
            reconstruction = reconstruct_and_evaluate(
                static,
                raw_prediction,
                variant_dir / "reconstruction",
                _reconstruction_config(config),
                normalized_prediction=normalized_prediction,
                edge_scale_epsilon=float(
                    config.get("target_scaling", {}).get("epsilon", 1e-12)
                ),
            )
            reconstruction = _sanitize(reconstruction)
            _write_json(variant_dir / "reconstruction" / "metrics.json", reconstruction)
        variant_results[variant] = {
            "runtime_seconds": runtime,
            "checkpoint": str(checkpoint_path),
            "initial": history[0],
            "final": history[-1],
            "reconstruction": reconstruction,
        }
        del model, optimizer, scaler
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "manifest": str(Path(manifest).resolve()),
        "config": str(Path(config_path).resolve()),
        "sample_id": str(base_sample["sample_id"]),
        "split": split,
        "steps": steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "target_distribution": target_distribution,
        "loss_parameters": {
            "magnitude_weight_lambda": magnitude_weight_lambda,
            "direction_lambda": direction_lambda,
        },
        "variants": variant_results,
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


@torch.no_grad()
def _evaluate_model(
    model: torch.nn.Module,
    base_sample: Mapping[str, Any],
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    query_settings: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    seed: int,
) -> dict[str, float]:
    model.eval()
    exact = dict(base_sample)
    exact["query_positions"] = exact["vertices"]
    exact["query_is_exact"] = torch.ones(
        exact["vertices"].shape[0], dtype=torch.bool, device=device
    )
    perturbed = apply_query_augmentation(
        base_sample, query_settings, base_seed=seed, epoch=1_000_003
    )
    predictions = []
    for sample in (exact, perturbed):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions.append(model(sample).predicted_laplacian.float())
    result: dict[str, float] = {}
    for name, prediction in zip(("exact", "perturbed"), predictions):
        loss = weighted_robust_laplacian_loss(
            prediction, target, confidence, loss_type="huber", huber_delta=0.01
        )
        target_mag = torch.linalg.vector_norm(target[valid], dim=-1)
        pred_mag = torch.linalg.vector_norm(prediction[valid], dim=-1)
        high = target_mag >= torch.quantile(target_mag, 0.90)
        cosine = F.cosine_similarity(
            prediction[valid][high], target[valid][high], dim=-1, eps=1e-8
        )
        result[f"{name}_huber_0.01"] = float(loss.item())
        result[f"{name}_magnitude_ratio"] = float(
            pred_mag.mean().div(target_mag.mean().clamp_min(1e-12)).item()
        )
        result[f"{name}_high_cosine"] = float(cosine.mean().item())
    return result


@torch.no_grad()
def _predict_exact(
    model: torch.nn.Module,
    base_sample: Mapping[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    exact = dict(base_sample)
    exact["query_positions"] = exact["vertices"]
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        prediction = model(exact).predicted_laplacian.float()
    return prediction.detach().cpu().numpy()


def _find_sample(dataset: PreparedMeshDataset, sample_id: str | None) -> int:
    if sample_id is None:
        return min(range(len(dataset)), key=lambda index: dataset.load_static(index)["vertices"].shape[0])
    for index, current in enumerate(dataset.sample_ids):
        if current == sample_id:
            return index
    raise ValueError(f"Sample {sample_id!r} is not present in split {dataset.records[0].split!r}.")


def _reconstruction_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("reconstruction", {})) or {
        "operator_type": "uniform",
        "lambda_lap": 1.0,
        "lambda_anchor": 0.01,
        "lambda_edge": 0.0,
        "num_iters": 500,
        "learning_rate": 0.01,
        "robust_loss": "huber",
        "huber_delta": 0.01,
        "dense_vertex_limit": 5000,
        "chamfer_samples": 1000,
        "metric_seed": 7,
    }


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _plot_history(path: Path, history: list[dict[str, Any]], variant: str) -> None:
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, [row["exact_huber_0.01"] for row in history], label="exact")
    axes[0].plot(steps, [row["perturbed_huber_0.01"] for row in history], label="perturbed")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Huber(0.01)")
    axes[0].legend()
    axes[1].plot(steps, [row["exact_magnitude_ratio"] for row in history])
    axes[1].set_ylabel("mean |pred| / mean |GT|")
    axes[2].plot(steps, [row["exact_high_cosine"] for row in history])
    axes[2].set_ylabel("top-10% cosine")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
    figure.suptitle(variant)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Single-mesh controlled overfit",
        "",
        f"Sample: `{summary['sample_id']}`; steps per variant: {summary['steps']}.",
        f"Adaptive Huber delta: {summary['target_distribution']['adaptive_huber_delta']:.8g} "
        "(median normalized target magnitude).",
        "",
        "| loss | initial exact Huber | final exact Huber | final |pred|/|GT| | final high cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, values in summary["variants"].items():
        initial = values["initial"]
        final = values["final"]
        lines.append(
            f"| {variant} | {initial['exact_huber_0.01']:.8g} | "
            f"{final['exact_huber_0.01']:.8g} | {final['exact_magnitude_ratio']:.4f} | "
            f"{final['exact_high_cosine']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value

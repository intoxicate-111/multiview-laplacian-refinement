from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.dataset import load_prepared_sample, move_sample_to_device
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.losses import laplacian_prediction_metrics
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.trainer import TrainingResult, load_checkpoint, train_single_object
from mlr.learned_laplacian.target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    denormalize_laplacian_by_edge_scale,
    edge_scale_statistics,
    graph_structure_statistics,
    normalize_laplacian_by_edge_scale,
    vector_magnitude_statistics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit one learned per-vertex Laplacian sample.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-mode",
        choices=["coarse_only", "multiview_only", "coarse_plus_multiview"],
    )
    parser.add_argument("--zero-images", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, help="Optional debugging override.")
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help="Skip training and resume evaluation from a saved best.pt checkpoint.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Write pre-training numerical diagnostics and exit.",
    )
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument(
        "--diagnostic-thresholds",
        type=float,
        nargs="+",
        default=[100.0, 10000.0, 100000000.0],
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.steps is not None:
        config.setdefault("training", {})["steps"] = args.steps
    sample = load_prepared_sample(args.sample)
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    pre_training_diagnostics = _write_pre_training_diagnostics(
        sample,
        config,
        args.output_dir,
        epsilon,
        output_path=args.diagnostics_output,
        thresholds=args.diagnostic_thresholds,
    )
    if args.diagnostics_only:
        return 0
    recovered_from_checkpoint = args.evaluate_checkpoint is not None
    if recovered_from_checkpoint:
        result = _recover_training_result(
            args.evaluate_checkpoint, sample, config, args.output_dir, args.device
        )
    else:
        result = train_single_object(
            sample,
            config,
            output_dir=args.output_dir,
            device_override=args.device,
            input_mode_override=args.input_mode,
            zero_images=args.zero_images,
        )
    device_sample = move_sample_to_device(sample, torch.device(result.device))
    with torch.no_grad():
        model_output = result.model(device_sample)
        target_space_prediction = model_output.predicted_laplacian
        local_edge_length = device_sample["local_edge_length"]
        if result.target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
            normalized_prediction = target_space_prediction
            raw_prediction = denormalize_laplacian_by_edge_scale(
                normalized_prediction, local_edge_length
            )
        else:
            raw_prediction = target_space_prediction
            normalized_prediction = normalize_laplacian_by_edge_scale(
                raw_prediction,
                local_edge_length,
                eps=result.target_scaling_epsilon,
                valid_scale_mask=device_sample["valid_scale_mask"],
            )
        prediction = raw_prediction.detach().cpu()
        normalized_prediction_cpu = normalized_prediction.detach().cpu()
    if tuple(prediction.shape) != tuple(sample["raw_laplacian_target"].shape):
        raise RuntimeError("Model output shape does not match [N, 3] target shape.")
    evaluation = reconstruct_and_evaluate(
        sample,
        prediction,
        output_dir=args.output_dir,
        reconstruction_config=config.get("reconstruction", {}),
        normalized_prediction=normalized_prediction_cpu,
        edge_scale_epsilon=result.target_scaling_epsilon,
    )
    raw_target = sample["raw_laplacian_target"]
    normalized_target = normalize_laplacian_by_edge_scale(
        raw_target,
        sample["local_edge_length"],
        eps=result.target_scaling_epsilon,
        valid_scale_mask=sample["valid_scale_mask"],
    )
    metrics = {
        "sample_id": sample["sample_id"],
        "device": result.device,
        "input_mode": args.input_mode or config.get("input_mode", "coarse_plus_multiview"),
        "zero_images": bool(args.zero_images),
        "target_mode": result.target_mode,
        "pre_training_diagnostics": pre_training_diagnostics,
        "sample": {
            "num_vertices": int(sample["vertices"].shape[0]),
            "num_faces": int(sample["faces"].shape[0]),
            "num_views": int(sample["images"].shape[0]),
            "image_height": int(sample["images"].shape[-2]),
            "image_width": int(sample["images"].shape[-1]),
            "zero_valid_view_vertices": int((model_output.valid_view_ratio == 0).sum().item()),
            "zero_valid_view_fraction": float(
                (model_output.valid_view_ratio == 0).float().mean().item()
            ),
            "mean_valid_views": float(model_output.valid_views.sum(dim=0).float().mean().item()),
            "metadata": sample.get("metadata", {}),
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": result.device,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        },
        "training": {
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "best_loss": result.best_loss,
            "best_step": result.best_step,
            "loss_reduction_ratio": result.best_loss / max(result.initial_loss, 1e-12),
            "runtime_seconds": result.runtime_seconds,
            "steps": int(config.get("training", {}).get("steps", 0)),
            "target_space_prediction": result.prediction_metrics,
            "clipped_target_vertices": result.clipped_target_vertices,
            "recovered_from_checkpoint": recovered_from_checkpoint,
            "runtime_source": (
                "checkpoint_creation_to_history_creation_estimate"
                if recovered_from_checkpoint
                else "measured_train_single_object_wall_time"
            ),
        },
        "target_distribution": _tensor_distribution(raw_target),
        "target_distributions": {
            "raw_laplacian_magnitude": vector_magnitude_statistics(raw_target),
            "normalized_laplacian_magnitude": vector_magnitude_statistics(normalized_target),
        },
        "edge_scale": {
            **edge_scale_statistics(sample["local_edge_length"]),
            "epsilon": result.target_scaling_epsilon,
            "raw_magnitude_correlation_with_h2": _correlation_with_scale(
                raw_target, sample["local_edge_scale"]
            ),
            "normalized_magnitude_correlation_with_h2": _correlation_with_scale(
                normalized_target, sample["local_edge_scale"]
            ),
        },
        **evaluation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_optional_loss_curve(result.history, args.output_dir / "loss_curve.png")
    print(json.dumps(metrics, indent=2))
    return 0


def _write_pre_training_diagnostics(
    sample: dict,
    config: dict,
    output_dir: Path,
    epsilon: float,
    output_path: Path | None = None,
    thresholds: list[float] | None = None,
) -> dict:
    edge_index = faces_to_edge_index(sample["faces"], sample["vertices"].shape[0])
    raw_target = sample["raw_laplacian_target"]
    normalized_target = normalize_laplacian_by_edge_scale(
        raw_target,
        sample["local_edge_length"],
        eps=epsilon,
        valid_scale_mask=sample["valid_scale_mask"],
    )
    degree = torch.bincount(edge_index[1], minlength=sample["vertices"].shape[0])
    normalized_magnitude = torch.linalg.vector_norm(normalized_target.double(), dim=-1)
    top_indices = torch.argsort(normalized_magnitude, descending=True)[:20]
    thresholds = thresholds or [100.0, 10000.0, 100000000.0]
    top_vertices = []
    for index_t in top_indices:
        index = int(index_t.item())
        top_vertices.append(
            {
                "vertex_index": index,
                "position": sample["vertices"][index].double().tolist(),
                "degree": int(degree[index].item()),
                "h": float(sample["local_edge_length"][index].item()),
                "h2": float(sample["local_edge_scale"][index].item()),
                "raw_target": raw_target[index].double().tolist(),
                "normalized_target": normalized_target[index].double().tolist(),
                "normalized_target_magnitude": float(normalized_magnitude[index].item()),
                "valid_scale": bool(sample["valid_scale_mask"][index].item()),
            }
        )
    diagnostics = {
        "target_mode": str(config.get("target_mode", "raw_laplacian")),
        "vertex_count": int(sample["vertices"].shape[0]),
        "face_count": int(sample["faces"].shape[0]),
        "graph": graph_structure_statistics(edge_index, sample["vertices"].shape[0]),
        "local_edge_scale": edge_scale_statistics(sample["local_edge_length"]),
        "raw_target_magnitude": vector_magnitude_statistics(raw_target),
        "normalized_target_magnitude": vector_magnitude_statistics(normalized_target),
        "finite_values": {
            "raw_target_nan_count": int(torch.isnan(raw_target).sum().item()),
            "raw_target_infinite_count": int(torch.isinf(raw_target).sum().item()),
            "normalized_target_nan_count": int(torch.isnan(normalized_target).sum().item()),
            "normalized_target_infinite_count": int(torch.isinf(normalized_target).sum().item()),
        },
        "normalized_magnitude_above_threshold": {
            str(float(threshold)): int((normalized_magnitude > threshold).sum().item())
            for threshold in thresholds
        },
        "top_20_normalized_target_vertices": top_vertices,
        "correlations_with_h2": {
            "raw_target_magnitude": _correlation_with_scale(
                raw_target, sample["local_edge_scale"]
            ),
            "normalized_target_magnitude": _correlation_with_scale(
                normalized_target, sample["local_edge_scale"]
            ),
        },
        "epsilon": epsilon,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (output_dir / "pre_training_diagnostics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print("pre_training_diagnostics=" + json.dumps(diagnostics, sort_keys=True), flush=True)
    return diagnostics


def _recover_training_result(
    checkpoint_path: Path,
    sample: dict,
    config: dict,
    output_dir: Path,
    device_override: str | None,
) -> TrainingResult:
    device = torch.device(device_override or config.get("device", "cpu"))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LearnedLaplacianModel(**payload["model_config"]).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    device_sample = move_sample_to_device(sample, device)
    target_mode = str(config.get("target_mode", "raw_laplacian"))
    scaling = config.get("target_scaling", {})
    epsilon = float(scaling.get("epsilon", 1e-12))
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        training_target = normalize_laplacian_by_edge_scale(
            device_sample["raw_laplacian_target"],
            device_sample["local_edge_length"],
            eps=epsilon,
            valid_scale_mask=device_sample["valid_scale_mask"],
        )
    else:
        training_target = device_sample["raw_laplacian_target"]
    clipped_target_vertices = 0
    clip_max_norm = scaling.get("clip_max_norm")
    if clip_max_norm is not None:
        magnitudes = torch.linalg.vector_norm(training_target, dim=-1)
        clipped = magnitudes > float(clip_max_norm)
        clipped_target_vertices = int(clipped.sum().item())
        factors = (float(clip_max_norm) / magnitudes.clamp_min(1e-12)).clamp_max(1.0)
        training_target = training_target * factors.unsqueeze(-1)
    with torch.no_grad():
        prediction = model(device_sample).predicted_laplacian
        prediction_metrics = laplacian_prediction_metrics(
            prediction, training_target, valid_mask=device_sample["valid_scale_mask"]
        )
    history_path = output_dir / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    # Windows preserves creation time when best.pt is overwritten. The delta to
    # the post-training history file recovers the elapsed training time after an
    # evaluation-only failure; it is explicitly labelled as an estimate.
    runtime_seconds = max(history_path.stat().st_ctime - checkpoint_path.stat().st_ctime, 0.0)
    return TrainingResult(
        model=model,
        history=history,
        initial_loss=float(history[0]["loss"]),
        final_loss=float(history[-1]["loss"]),
        best_loss=float(payload["loss"]),
        best_step=int(payload["step"]),
        prediction_metrics=prediction_metrics,
        device=str(device),
        runtime_seconds=runtime_seconds,
        peak_gpu_memory_mb=None,
        target_mode=target_mode,
        target_scaling_epsilon=epsilon,
        clipped_target_vertices=clipped_target_vertices,
    )


def _write_optional_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = np.asarray([item["step"] for item in history])
    losses = np.asarray([item["loss"] for item in history])
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(steps, losses)
    axes.set_xlabel("Training step")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _tensor_distribution(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().float().reshape(-1)
    magnitudes = torch.linalg.vector_norm(values.detach().float(), dim=-1)
    return {
        "component_mean": float(flattened.mean().item()),
        "component_std": float(flattened.std().item()),
        "component_abs_mean": float(flattened.abs().mean().item()),
        "vector_magnitude_mean": float(magnitudes.mean().item()),
        "vector_magnitude_median": float(magnitudes.median().item()),
        "vector_magnitude_max": float(magnitudes.max().item()),
    }


def _correlation_with_scale(vectors: torch.Tensor, scale: torch.Tensor) -> float:
    magnitudes = torch.linalg.vector_norm(vectors.detach().double(), dim=-1)
    scale = scale.detach().double()
    if float(magnitudes.std().item()) <= 1e-15 or float(scale.std().item()) <= 1e-15:
        return 0.0
    matrix = torch.stack((magnitudes, scale))
    return float(torch.corrcoef(matrix)[0, 1].item())


if __name__ == "__main__":
    raise SystemExit(main())

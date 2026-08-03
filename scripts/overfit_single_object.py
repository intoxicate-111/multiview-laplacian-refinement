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
from mlr.learned_laplacian.trainer import train_single_object


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
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.steps is not None:
        config.setdefault("training", {})["steps"] = args.steps
    sample = load_prepared_sample(args.sample)
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
        prediction = model_output.predicted_laplacian.detach().cpu()
    if tuple(prediction.shape) != tuple(sample["laplacian_target"].shape):
        raise RuntimeError("Model output shape does not match [N, 3] target shape.")
    evaluation = reconstruct_and_evaluate(
        sample,
        prediction,
        output_dir=args.output_dir,
        reconstruction_config=config.get("reconstruction", {}),
    )
    metrics = {
        "sample_id": sample["sample_id"],
        "device": result.device,
        "input_mode": args.input_mode or config.get("input_mode", "coarse_plus_multiview"),
        "zero_images": bool(args.zero_images),
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
        },
        "target_distribution": _tensor_distribution(sample["laplacian_target"]),
        **evaluation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_optional_loss_curve(result.history, args.output_dir / "loss_curve.png")
    print(json.dumps(metrics, indent=2))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())

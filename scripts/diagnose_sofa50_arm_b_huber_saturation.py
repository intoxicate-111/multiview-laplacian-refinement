#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.huber_saturation_diagnostic import (
    summarize_huber_saturation,
    write_huber_saturation_outputs,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one
from mlr.learned_laplacian.trainer import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose component-wise Huber saturation in Sofa50 Arm B."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Local execution device; defaults to CPU and never submits a scheduler job.",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    run_dir = args.arm_b_run.resolve()
    checkpoint = run_dir / "checkpoint_latest.pt"
    run_config_path = run_dir / "run_config.json"
    if not checkpoint.is_file() or not run_config_path.is_file():
        raise FileNotFoundError(f"Incomplete Arm B run: {run_dir}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    config = run_config.get("experiment_config", run_config)
    metrics_path = run_dir / "metrics.json"
    native_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    if config.get("target_mode") != "raw_laplacian":
        raise ValueError("This diagnostic requires Arm B raw_laplacian output.")
    if config.get("training", {}).get("prediction_loss_space") != "output_representation":
        raise ValueError("This diagnostic requires output_representation loss space.")
    if config.get("training", {}).get("loss") != "huber":
        raise ValueError("This diagnostic requires Huber prediction loss.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    spec = {
        "config": config,
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }
    dataset = PreparedMeshDataset.from_manifest(manifest, args.split)

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        inferred = _infer_one(
            dataset,
            index,
            spec,
            device,
            current_faces=static["faces"],
        )
        valid = torch.as_tensor(inferred["valid"]).bool().numpy()
        predictions.append(torch.as_tensor(inferred["prediction_raw"])[valid].numpy())
        targets.append(torch.as_tensor(inferred["target_raw"])[valid].numpy())
        weights.append(torch.as_tensor(inferred["target_confidence"])[valid].numpy())
        sample_indices.append(np.full(int(valid.sum()), index, dtype=np.int64))
        print(
            f"{args.split} {index + 1}/{len(dataset)} "
            f"{static['sample_id']} valid_vertices={int(valid.sum())}",
            flush=True,
        )

    delta = float(config.get("training", {}).get("huber_delta", 0.01))
    summary = summarize_huber_saturation(
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(weights),
        np.concatenate(sample_indices),
        huber_delta=delta,
    )
    metadata = {
        "experiment": "Sofa50 Arm B raw-Laplacian Huber saturation diagnostic",
        "split": args.split,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer_steps": int(payload.get("optimizer_steps", -1)),
        "target_mode": config.get("target_mode"),
        "prediction_loss_space": config.get("training", {}).get(
            "prediction_loss_space"
        ),
        "local_inference_precision": (
            "float32" if device.type == "cpu" else str(amp_dtype)
        ),
        "recorded_final_validation_loss": native_metrics.get(
            "final_validation_loss"
        ),
    }
    recorded_loss = metadata["recorded_final_validation_loss"]
    recomputed_loss = summary["overall"]["weighted_huber_loss_total"]
    metadata["recomputed_validation_loss"] = recomputed_loss
    metadata["relative_loss_difference_from_recorded"] = (
        (recomputed_loss - float(recorded_loss)) / float(recorded_loss)
        if recorded_loss not in (None, 0)
        else None
    )
    write_huber_saturation_outputs(args.output_dir.resolve(), summary, metadata=metadata)
    print(f"report\t{args.output_dir.resolve() / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

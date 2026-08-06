from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .image_ablation import _condition_sample
from .losses import weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .query_training import apply_query_augmentation, query_augmentation_settings


SINGLE_IMAGE_CONDITIONS = (
    "original_rgb",
    "zero_rgb",
    "shuffled_images",
    "cross_object_rgb",
)


def run_single_checkpoint_image_ablation(
    checkpoint: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    *,
    split: str = "validation",
    sample_id: str,
    device: str = "cuda",
    seed: int = 17,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = _device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    config = payload["config"]
    model = _build_model(config, None, False).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    dataset = PreparedMeshDataset.from_manifest(manifest, split)
    index = dataset.sample_ids.index(sample_id)
    donor_index = (index + 1) % len(dataset)
    eval_config = _exact_query_config(config)
    prepared = _prepare(dataset.load_static(index), eval_config, resolved_device)
    donor = _prepare(dataset.load_static(donor_index), eval_config, resolved_device)
    sample = dict(prepared.sample)
    sample["query_positions"] = sample["vertices"]
    target = prepared.training_target.float()
    confidence = sample["target_confidence"].float()
    valid = sample["valid_scale_mask"].bool() & (confidence > 0)
    permutation = torch.randperm(
        int(sample["images"].shape[0]), generator=torch.Generator().manual_seed(seed)
    ).to(resolved_device)
    predictions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for condition in SINGLE_IMAGE_CONDITIONS:
            conditioned = _condition_sample(
                sample, donor.sample["images"], permutation, condition
            )
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=resolved_device.type == "cuda",
            ):
                predictions[condition] = model(conditioned).predicted_laplacian.float()
    zero_loss = _fixed_huber(torch.zeros_like(target), target, confidence)
    conditions = {
        name: _prediction_metrics(prediction, target, confidence, valid, zero_loss)
        for name, prediction in predictions.items()
    }
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target.detach().cpu().numpy(),
        **{name: value.detach().cpu().numpy() for name, value in predictions.items()},
    )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_variant": payload.get("variant"),
        "checkpoint_steps": payload.get("steps"),
        "sample_id": sample_id,
        "donor_sample_id": str(donor.sample["sample_id"]),
        "query": "exact GT-query on the memorized mesh",
        "conditions": conditions,
    }
    _write_json(output_dir / "metrics.json", result)
    _write_conditions_csv(output_dir / "metrics.csv", conditions)
    (output_dir / "REPORT.md").write_text(_single_report(result), encoding="utf-8")
    return result


def run_mesh_count_scaling(
    gt_manifest: str | Path,
    expanded_manifest: str | Path | None,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    mesh_counts: Sequence[int] = (1, 2, 4, 8, 16),
    exposures_per_mesh: int = 500,
    accumulation_meshes: int = 4,
    device: str = "cuda",
    seed: int = 7,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    training = config.setdefault("training", {})
    training["loss"] = "huber"
    training["huber_delta"] = 0.01
    training["target_magnitude_weight_lambda"] = 0.0
    resolved_device = _device(device)
    gt_dataset = PreparedMeshDataset.from_manifest(gt_manifest, "train")
    expanded_dataset = (
        PreparedMeshDataset.from_manifest(expanded_manifest, "train")
        if expanded_manifest is not None
        else None
    )
    maximum = max(int(value) for value in mesh_counts)
    selected_ids = list(gt_dataset.sample_ids[:maximum])
    expanded_by_id = (
        {sample_id: index for index, sample_id in enumerate(expanded_dataset.sample_ids)}
        if expanded_dataset is not None
        else {}
    )
    if expanded_dataset is not None:
        missing = [sample_id for sample_id in selected_ids if sample_id not in expanded_by_id]
        if missing:
            raise ValueError(f"Expanded manifest is missing selected meshes: {missing}")

    print(f"Caching {maximum} GT-query samples on {resolved_device}...", flush=True)
    gt_cache = [
        _prepare(gt_dataset.load_static(index), config, resolved_device)
        for index in range(maximum)
    ]
    results: dict[str, Any] = {}
    for mesh_count_value in mesh_counts:
        mesh_count = int(mesh_count_value)
        run_dir = output_dir / f"mesh_count_{mesh_count:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "mesh_count": mesh_count,
            "selected_sample_ids": selected_ids[:mesh_count],
            "exposures_per_mesh": exposures_per_mesh,
            "gradient_accumulation_meshes": accumulation_meshes,
            "seed": seed,
            "training": training,
            "budget_definition": (
                "Every selected mesh contributes one forward/backward pass per exposure cycle."
            ),
        }
        _write_json(run_dir / "config.json", run_config)
        result = _train_one_scale(
            gt_cache[:mesh_count],
            gt_dataset,
            expanded_dataset,
            expanded_by_id,
            config,
            run_dir,
            exposures_per_mesh,
            accumulation_meshes,
            resolved_device,
            seed,
        )
        results[str(mesh_count)] = result

    summary = {
        "mesh_counts": [int(value) for value in mesh_counts],
        "exposures_per_mesh": exposures_per_mesh,
        "selected_sample_ids_nested_order": selected_ids,
        "loss": {"type": "huber", "delta": 0.01, "target_magnitude_weight_lambda": 0.0},
        "results": results,
        "collapse_rule": (
            "amplitude collapse if mean |pred|/mean |GT| < 0.10; near-zero behavior if "
            "relative improvement over zero predictor is also < 0.01"
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_scaling_csv(output_dir / "summary.csv", results)
    _write_per_mesh_csv(output_dir / "per_mesh_metrics.csv", results)
    (output_dir / "REPORT.md").write_text(_scaling_report(summary), encoding="utf-8")
    return summary


def _train_one_scale(
    prepared_samples: Sequence[Any],
    gt_dataset: PreparedMeshDataset,
    expanded_dataset: PreparedMeshDataset | None,
    expanded_by_id: Mapping[str, int],
    config: Mapping[str, Any],
    run_dir: Path,
    exposures_per_mesh: int,
    accumulation_meshes: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(config, None, False).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("training", {}).get("learning_rate", 1e-3)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 0.0)),
    )
    amp_enabled = device.type == "cuda" and bool(
        config.get("training", {}).get("amp", {}).get("enabled", True)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    settings = query_augmentation_settings(config)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    start = time.perf_counter()
    model.train()
    for exposure in range(1, exposures_per_mesh + 1):
        order = torch.randperm(len(prepared_samples), generator=generator).tolist()
        for offset in range(0, len(order), accumulation_meshes):
            indices = order[offset : offset + accumulation_meshes]
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            for index in indices:
                prepared = prepared_samples[index]
                sample = apply_query_augmentation(
                    prepared.sample, settings, base_seed=seed, epoch=exposure
                )
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    prediction = model(sample).predicted_laplacian
                loss = weighted_robust_laplacian_loss(
                    prediction.float(),
                    prepared.training_target.float(),
                    sample["target_confidence"].float(),
                    loss_type="huber",
                    huber_delta=0.01,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at mesh_count={len(prepared_samples)}, exposure={exposure}."
                    )
                group_losses.append(loss)
            group_loss = torch.stack(group_losses).mean()
            scaler.scale(group_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
        if exposure == 1 or exposure % 25 == 0 or exposure == exposures_per_mesh:
            gt_metrics = _evaluate_prepared(model, prepared_samples, device)
            history.append(
                {
                    "exposure_per_mesh": exposure,
                    "optimizer_steps": optimizer_steps,
                    "gt_query_validation_loss": gt_metrics["validation_loss"],
                    "gt_query_magnitude_ratio": gt_metrics["mean_prediction_to_target_magnitude_ratio"],
                    "gt_query_high_10_cosine": gt_metrics["high_10_cosine"],
                }
            )
            print(
                f"mesh_count={len(prepared_samples):02d} exposure={exposure:04d} "
                f"loss={gt_metrics['validation_loss']:.6g} "
                f"ratio={gt_metrics['mean_prediction_to_target_magnitude_ratio']:.4f} "
                f"high_cos={gt_metrics['high_10_cosine']:.4f}",
                flush=True,
            )
            model.train()

    gt_metrics = _evaluate_prepared(model, prepared_samples, device)
    if expanded_dataset is None:
        expanded_metrics: dict[str, Any] = {
            "available": False,
            "reason": (
                "No coarse/expanded-query manifest was supplied; raw GT meshes must not be "
                "presented as a substitute for real expanded-query geometry."
            ),
        }
    else:
        expanded_prepared = []
        exact_config = _exact_query_config(config)
        for prepared in prepared_samples:
            sample_id = str(prepared.sample["sample_id"])
            expanded_prepared.append(
                _prepare(
                    expanded_dataset.load_static(expanded_by_id[sample_id]),
                    exact_config,
                    device,
                )
            )
        expanded_metrics = _evaluate_prepared(model, expanded_prepared, device)
    runtime = time.perf_counter() - start
    checkpoint = run_dir / "final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "mesh_count": len(prepared_samples),
            "sample_ids": [str(item.sample["sample_id"]) for item in prepared_samples],
            "exposures_per_mesh": exposures_per_mesh,
            "optimizer_steps": optimizer_steps,
        },
        checkpoint,
    )
    _write_json(run_dir / "history.json", {"history": history})
    _write_history_csv(run_dir / "history.csv", history)
    metrics = {
        "mesh_count": len(prepared_samples),
        "sample_ids": [str(item.sample["sample_id"]) for item in prepared_samples],
        "exposures_per_mesh": exposures_per_mesh,
        "optimizer_steps": optimizer_steps,
        "runtime_seconds": runtime,
        "checkpoint": str(checkpoint),
        "gt_query": gt_metrics,
        "expanded_query": expanded_metrics,
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


@torch.no_grad()
def _evaluate_prepared(
    model: torch.nn.Module, samples: Sequence[Any], device: torch.device
) -> dict[str, Any]:
    model.eval()
    records = []
    all_target = []
    all_prediction = []
    losses = []
    zero_losses = []
    for prepared in samples:
        sample = dict(prepared.sample)
        sample["query_positions"] = sample["vertices"]
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction = model(sample).predicted_laplacian.float()
        target = prepared.training_target.float()
        confidence = sample["target_confidence"].float()
        valid = sample["valid_scale_mask"].bool() & (confidence > 0)
        zero_loss = _fixed_huber(torch.zeros_like(target), target, confidence)
        metrics = _prediction_metrics(prediction, target, confidence, valid, zero_loss)
        metrics["sample_id"] = str(sample["sample_id"])
        records.append(metrics)
        losses.append(metrics["validation_loss"])
        zero_losses.append(zero_loss)
        all_target.append(target[valid].detach().cpu())
        all_prediction.append(prediction[valid].detach().cpu())
    target_all = torch.cat(all_target)
    prediction_all = torch.cat(all_prediction)
    target_magnitude = torch.linalg.vector_norm(target_all, dim=-1)
    prediction_magnitude = torch.linalg.vector_norm(prediction_all, dim=-1)
    high = target_magnitude >= torch.quantile(target_magnitude, 0.90)
    loss = float(np.mean(losses))
    zero_loss = float(np.mean(zero_losses))
    return {
        "validation_loss": loss,
        "zero_predictor_loss": zero_loss,
        "relative_improvement_vs_zero_predictor": _relative_improvement(zero_loss, loss),
        "mean_prediction_to_target_magnitude_ratio": float(
            prediction_magnitude.mean().div(target_magnitude.mean().clamp_min(1e-12)).item()
        ),
        "high_10_cosine": float(
            F.cosine_similarity(
                prediction_all[high], target_all[high], dim=-1, eps=1e-8
            ).mean().item()
        ),
        "per_mesh": records,
    }


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    zero_loss: float,
) -> dict[str, float]:
    loss = _fixed_huber(prediction, target, confidence)
    target_magnitude = torch.linalg.vector_norm(target[valid], dim=-1)
    prediction_magnitude = torch.linalg.vector_norm(prediction[valid], dim=-1)
    high = target_magnitude >= torch.quantile(target_magnitude, 0.90)
    return {
        "validation_loss": loss,
        "zero_predictor_loss": zero_loss,
        "relative_improvement_vs_zero_predictor": _relative_improvement(zero_loss, loss),
        "mean_prediction_to_target_magnitude_ratio": float(
            prediction_magnitude.mean().div(target_magnitude.mean().clamp_min(1e-12)).item()
        ),
        "high_10_cosine": float(
            F.cosine_similarity(
                prediction[valid][high], target[valid][high], dim=-1, eps=1e-8
            ).mean().item()
        ),
    }


def _prepare(sample: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> Any:
    return _prepare_item_for_use(
        _prepare_object_static(sample, config),
        config,
        device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )


def _exact_query_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.setdefault("query_training", {})["enabled"] = False
    return result


def _fixed_huber(
    prediction: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor
) -> float:
    return float(
        weighted_robust_laplacian_loss(
            prediction, target, confidence, loss_type="huber", huber_delta=0.01
        ).item()
    )


def _relative_improvement(baseline: float, value: float) -> float:
    return (baseline - value) / baseline if baseline > 0 else 0.0


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_conditions_csv(path: Path, conditions: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ("condition", "validation_loss", "zero_predictor_loss", "relative_improvement_vs_zero_predictor", "mean_prediction_to_target_magnitude_ratio", "high_10_cosine")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metrics in conditions.items():
            writer.writerow({"condition": name, **metrics})


def _write_scaling_csv(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ("mesh_count", "query", "validation_loss", "zero_predictor_loss", "relative_improvement_vs_zero_predictor", "mean_prediction_to_target_magnitude_ratio", "high_10_cosine", "optimizer_steps", "runtime_seconds")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for count, result in results.items():
            for query in _available_queries(result):
                writer.writerow({"mesh_count": count, "query": query, **{key: result[query][key] for key in fields if key in result[query]}, "optimizer_steps": result["optimizer_steps"], "runtime_seconds": result["runtime_seconds"]})


def _write_per_mesh_csv(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ("mesh_count", "query", "sample_id", "validation_loss", "zero_predictor_loss", "relative_improvement_vs_zero_predictor", "mean_prediction_to_target_magnitude_ratio", "high_10_cosine")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for count, result in results.items():
            for query in _available_queries(result):
                for metrics in result[query]["per_mesh"]:
                    writer.writerow({"mesh_count": count, "query": query, **metrics})


def _write_history_csv(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _single_report(result: Mapping[str, Any]) -> str:
    lines = ["# Single-mesh checkpoint image ablation", "", "| condition | loss | vs zero predictor | |pred|/|GT| | high-10% cosine |", "|---|---:|---:|---:|---:|"]
    for name, metrics in result["conditions"].items():
        lines.append(f"| {name} | {metrics['validation_loss']:.8g} | {metrics['relative_improvement_vs_zero_predictor']:.3%} | {metrics['mean_prediction_to_target_magnitude_ratio']:.3%} | {metrics['high_10_cosine']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def _scaling_report(summary: Mapping[str, Any]) -> str:
    lines = ["# Mesh-count scaling diagnostic", "", f"Each mesh received {summary['exposures_per_mesh']} forward/backward exposures.", "", "| meshes | query | loss | vs zero | |pred|/|GT| | high-10% cosine |", "|---:|---|---:|---:|---:|---:|"]
    amplitude_collapse_at = None
    strict_collapse_at = None
    for count, result in summary["results"].items():
        for query in _available_queries(result):
            metrics = result[query]
            lines.append(f"| {count} | {query} | {metrics['validation_loss']:.8g} | {metrics['relative_improvement_vs_zero_predictor']:.3%} | {metrics['mean_prediction_to_target_magnitude_ratio']:.3%} | {metrics['high_10_cosine']:.4f} |")
        gt = result["gt_query"]
        if amplitude_collapse_at is None and gt["mean_prediction_to_target_magnitude_ratio"] < 0.10:
            amplitude_collapse_at = int(count)
        if strict_collapse_at is None and gt["mean_prediction_to_target_magnitude_ratio"] < 0.10 and gt["relative_improvement_vs_zero_predictor"] < 0.01:
            strict_collapse_at = int(count)
    expanded_available = any(
        result.get("expanded_query", {}).get("available", True)
        for result in summary["results"].values()
    )
    expanded_note = (
        "Expanded-query metrics use the corresponding prepared prediction graph."
        if expanded_available
        else (
            "Expanded-query metrics are unavailable because no real coarse/expanded graph "
            "was supplied; GT geometry was not used as a proxy."
        )
    )
    lines.extend(("", f"First GT-query amplitude collapse (<10%): {amplitude_collapse_at if amplitude_collapse_at is not None else 'none'} meshes.", f"First GT-query scale satisfying the stricter amplitude-plus-zero-baseline rule: {strict_collapse_at if strict_collapse_at is not None else 'none'}.", "", "The curve need not be monotonic because equal per-mesh exposures produce more shared optimizer updates at larger mesh counts. Per-mesh values are in `per_mesh_metrics.csv`; GT-query metrics are in-sample exact-query memorization metrics. " + expanded_note, ""))
    return "\n".join(lines)


def _available_queries(result: Mapping[str, Any]) -> tuple[str, ...]:
    if result.get("expanded_query", {}).get("available", True):
        return ("gt_query", "expanded_query")
    return ("gt_query",)

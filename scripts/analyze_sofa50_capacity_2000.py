#!/usr/bin/env python3
from __future__ import annotations

"""Sofa50 C0/C1/C2 capacity ablation analysis.

VERSION: 2026-08-08-v2-source-manifest-root-fix

Important path rule:
- Fresh RGB evaluation NEVER uses <arm>/dataset_manifest.json as the dataset root.
- By default the original training manifest is recovered from
  <arm>/run_config.json -> source_manifest.
- --manifest may explicitly override that source.
"""

import argparse
import copy
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.diagnostics import _loss_kwargs
from mlr.learned_laplacian.geometry_aware_sampling import _magnitude_masks
from mlr.learned_laplacian.losses import weighted_robust_laplacian_loss
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from mlr.learned_laplacian.trainer import load_checkpoint


VERSION = "2026-08-08-v2-source-manifest-root-fix"
ARMS = {
    "C0": "C0_16_64",
    "C1": "C1_32_128",
    "C2": "C2_64_256",
}
EXPECTED_WIDTHS = {
    "C0": (16, 64),
    "C1": (32, 128),
    "C2": (64, 256),
}
CONDITIONS = (
    "original_rgb",
    "zero_rgb",
    "shuffled_images",
    "cross_object_rgb",
    "shuffled_view_order",
    "zero_predictor",
)
GROUPS = (
    "all",
    "lowest_10",
    "smooth_bottom_90",
    "high_top_10",
    "high_top_1_to_10",
    "high_top_1",
)
REPORT_GROUPS = ("all", "smooth_bottom_90", "high_top_10", "high_top_1")
PAIRWISE = (("C0", "C1"), ("C0", "C2"), ("C1", "C2"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else 0.0


def resolve_manifest(output_root: Path, cli_manifest: Path | None) -> tuple[Path, str, dict[str, str]]:
    recorded: dict[str, str] = {}
    for arm, dirname in ARMS.items():
        run_config_path = output_root / dirname / "run_config.json"
        if not run_config_path.is_file():
            raise FileNotFoundError(f"Missing run metadata: {run_config_path}")
        run_config = read_json(run_config_path)
        source = run_config.get("source_manifest")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing source_manifest in {run_config_path}")
        recorded[arm] = str(Path(source).expanduser().resolve())

    if cli_manifest is not None:
        manifest = cli_manifest.expanduser().resolve()
        source_kind = "cli --manifest override"
    else:
        unique = sorted(set(recorded.values()))
        if len(unique) != 1:
            raise ValueError("C0/C1/C2 record different source_manifest values:\n" + "\n".join(unique))
        manifest = Path(unique[0])
        source_kind = "run_config.json -> source_manifest"

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Original evaluation manifest does not exist: {manifest}\n"
            "Pass --manifest with the current Sofa50 gt_query_manifest.json path."
        )
    return manifest, source_kind, recorded


def manifest_sample_ids(path: Path, split: str) -> list[str]:
    payload = read_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Manifest has no samples list: {path}")
    result: list[str] = []
    for item in samples:
        if not isinstance(item, Mapping) or item.get("split") != split:
            continue
        sample_id = item.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            result.append(sample_id)
    return result


def audit_validation_sample_sets(output_root: Path, evaluation_manifest: Path) -> dict[str, Any]:
    reference = manifest_sample_ids(evaluation_manifest, "validation")
    if not reference:
        raise ValueError("Evaluation manifest has no validation sample IDs")
    arms: dict[str, Any] = {}
    reference_set = set(reference)
    for arm, dirname in ARMS.items():
        run_manifest = output_root / dirname / "dataset_manifest.json"
        if not run_manifest.is_file():
            raise FileNotFoundError(f"Missing run manifest for audit: {run_manifest}")
        ids = manifest_sample_ids(run_manifest, "validation")
        arms[arm] = {
            "count": len(ids),
            "matches_evaluation_manifest": set(ids) == reference_set,
            "missing": sorted(reference_set - set(ids)),
            "extra": sorted(set(ids) - reference_set),
        }
        if set(ids) != reference_set:
            raise ValueError(f"{arm} validation sample set differs from evaluation manifest")
    return {"evaluation_validation_ids": reference, "arms": arms}


def preflight_lazy_images(dataset: PreparedMeshDataset) -> dict[str, Any]:
    checked = 0
    first_resolved: str | None = None
    roots: set[str] = set()
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        root_value = static.get("_dataset_root")
        if root_value is None:
            raise ValueError(f"Sample {static.get('sample_id')} has no _dataset_root")
        root = Path(str(root_value)).expanduser().resolve()
        roots.add(str(root))
        image_paths = static.get("image_paths")
        if not image_paths:
            if isinstance(static.get("images"), torch.Tensor):
                continue
            raise ValueError(f"Sample {static.get('sample_id')} has neither image_paths nor embedded images")
        for raw in image_paths:
            path = Path(str(raw))
            resolved = path if path.is_absolute() else root / path
            resolved = resolved.resolve()
            if first_resolved is None:
                first_resolved = str(resolved)
            checked += 1
            if not resolved.is_file():
                raise FileNotFoundError(
                    "Lazy RGB preflight failed before checkpoint evaluation.\n"
                    f"sample_id={static.get('sample_id')}\n"
                    f"dataset_root={root}\n"
                    f"image_path={raw}\n"
                    f"resolved={resolved}"
                )
    return {
        "mesh_count": len(dataset),
        "dataset_roots": sorted(roots),
        "checked_image_files": checked,
        "first_resolved_image": first_resolved,
    }


def amp_settings(config: Mapping[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    amp = config.get("training", {}).get("amp", {})
    enabled = bool(amp.get("enabled", False)) and device.type == "cuda"
    dtype_name = str(amp.get("dtype", "float16")).lower()
    return enabled, torch.bfloat16 if dtype_name in ("bfloat16", "bf16") else torch.float16


def condition_sample(
    base: Mapping[str, Any],
    donor_images: torch.Tensor,
    permutation: torch.Tensor,
    condition: str,
) -> dict[str, Any]:
    sample = dict(base)
    if condition == "original_rgb":
        return sample
    if condition == "zero_rgb":
        sample["images"] = torch.zeros_like(base["images"])
        return sample
    if condition == "shuffled_images":
        sample["images"] = base["images"].index_select(0, permutation)
        return sample
    if condition == "cross_object_rgb":
        if tuple(donor_images.shape) != tuple(base["images"].shape):
            raise ValueError("cross_object_rgb requires matching image tensor shapes")
        sample["images"] = donor_images
        return sample
    if condition == "shuffled_view_order":
        for name in ("images", "intrinsics", "extrinsics", "visibility"):
            value = sample.get(name)
            if isinstance(value, torch.Tensor):
                sample[name] = value.index_select(0, permutation)
        return sample
    raise ValueError(f"Unknown condition: {condition}")


def metrics_for_mask(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"vertex_count": 0}
    pred = prediction[mask].astype(np.float64, copy=False)
    gt = target[mask].astype(np.float64, copy=False)
    residual = pred - gt
    endpoint = np.linalg.norm(residual, axis=1)
    pred_mag = np.linalg.norm(pred, axis=1)
    gt_mag = np.linalg.norm(gt, axis=1)
    denom = pred_mag * gt_mag
    stable = denom > 1e-12
    mean_cos = None
    if stable.any():
        mean_cos = float(np.mean(np.sum(pred[stable] * gt[stable], axis=1) / denom[stable]))
    global_denom = float(np.linalg.norm(pred) * np.linalg.norm(gt))
    global_cos = None
    if global_denom > 1e-12:
        global_cos = float(np.dot(pred.reshape(-1), gt.reshape(-1)) / global_denom)
    return {
        "vertex_count": int(mask.sum()),
        "mean_endpoint_error": float(endpoint.mean()),
        "median_endpoint_error": float(np.median(endpoint)),
        "mean_squared_error": float(np.mean(np.square(residual))),
        "mean_cosine": mean_cos,
        "global_cosine": global_cos,
        "mean_gt_magnitude": float(gt_mag.mean()),
        "mean_prediction_magnitude": float(pred_mag.mean()),
        "prediction_to_gt_magnitude_ratio": safe_div(float(pred_mag.mean()), float(gt_mag.mean())),
        "prediction_to_gt_global_norm_ratio": safe_div(float(np.linalg.norm(pred)), float(np.linalg.norm(gt))),
        "group_relative_error": safe_div(float(endpoint.mean()), float(gt_mag.mean())),
    }


def aggregate_metrics(records: Sequence[Mapping[str, Any]], loss_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    target = np.concatenate([np.asarray(r["target"]) for r in records], axis=0)
    confidence = np.concatenate([np.asarray(r["confidence"]) for r in records], axis=0)
    valid = np.concatenate([np.asarray(r["valid_mask"], dtype=bool) for r in records], axis=0)
    group_masks = {
        group: np.concatenate([np.asarray(r["group_masks"][group], dtype=bool) for r in records], axis=0) & valid
        for group in GROUPS
    }
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        if condition == "zero_predictor":
            pred = np.zeros_like(target)
        else:
            pred = np.concatenate([np.asarray(r["predictions"][condition]) for r in records], axis=0)
        cond: dict[str, Any] = {
            group: metrics_for_mask(pred, target, group_masks[group])
            for group in GROUPS
        }
        pred_t = torch.from_numpy(pred[valid]).float()
        target_t = torch.from_numpy(target[valid]).float()
        conf_t = torch.from_numpy(confidence[valid]).float()
        cond["training_loss"] = float(
            weighted_robust_laplacian_loss(pred_t, target_t, conf_t, **loss_kwargs).item()
        )
        conditions[condition] = cond

    original = conditions["original_rgb"]
    zero = conditions["zero_rgb"]
    zero_pred = conditions["zero_predictor"]
    image_gaps: dict[str, Any] = {}
    baseline_improvement: dict[str, Any] = {}
    for group in REPORT_GROUPS:
        image_gaps[group] = {
            "endpoint_zero_minus_original": (
                float(zero[group]["mean_endpoint_error"]) - float(original[group]["mean_endpoint_error"])
            ),
            "endpoint_relative_improvement_with_rgb": safe_div(
                float(zero[group]["mean_endpoint_error"]) - float(original[group]["mean_endpoint_error"]),
                float(zero[group]["mean_endpoint_error"]),
            ),
            "global_cosine_original_minus_zero": (
                float(original[group].get("global_cosine") or 0.0) - float(zero[group].get("global_cosine") or 0.0)
            ),
        }
        baseline_improvement[group] = {
            "endpoint_relative_improvement_vs_zero_predictor": safe_div(
                float(zero_pred[group]["mean_endpoint_error"]) - float(original[group]["mean_endpoint_error"]),
                float(zero_pred[group]["mean_endpoint_error"]),
            )
        }
    return {
        "conditions": conditions,
        "image_dependence": image_gaps,
        "improvement_vs_zero_predictor": baseline_improvement,
    }


@torch.no_grad()
def evaluate_arm(
    arm: str,
    arm_dir: Path,
    dataset: PreparedMeshDataset,
    device: torch.device,
    seed: int,
    prediction_dir: Path,
) -> dict[str, Any]:
    config = read_json(arm_dir / "config.json")
    exact_config = copy.deepcopy(config)
    exact_config.setdefault("query_training", {})["apply_to_validation"] = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(exact_config, None, False).to(device)
    checkpoint_payload = load_checkpoint(arm_dir / "best.pt", model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = amp_settings(exact_config, device)
    loss_kwargs = _loss_kwargs(exact_config)

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    prediction_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index in range(len(dataset)):
        prepared = _prepare_item_for_use(
            _prepare_object_static(dataset.load_static(index), exact_config),
            exact_config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        donor = _prepare_item_for_use(
            _prepare_object_static(dataset.load_static((index + 1) % len(dataset)), exact_config),
            exact_config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        base = dict(prepared.sample)
        base["query_positions"] = base["vertices"]
        base["query_is_exact"] = torch.ones(len(base["vertices"]), dtype=torch.bool, device=device)
        target = prepared.training_target.float()
        confidence = base["target_confidence"].float()
        valid = base["valid_scale_mask"].bool() & (confidence > 0)
        target_np = target.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        masks = _magnitude_masks(np.linalg.norm(target_np, axis=1))
        group_masks = {group: np.asarray(masks[group], dtype=bool) & valid_np for group in GROUPS}

        permutation = torch.randperm(
            int(base["images"].shape[0]),
            generator=torch.Generator().manual_seed(seed + index * 104729),
        ).to(device)
        predictions: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            if condition == "zero_predictor":
                continue
            sample = condition_sample(base, donor.sample["images"], permutation, condition)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = model(sample).predicted_laplacian.float()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"{arm} {base['sample_id']} {condition}: non-finite prediction")
            predictions[condition] = prediction.detach().cpu().numpy()

        sample_id = str(base["sample_id"])
        np.savez_compressed(
            prediction_dir / f"{sample_id}.npz",
            target=target_np,
            confidence=confidence.detach().cpu().numpy(),
            valid_mask=valid_np,
            **predictions,
        )
        records.append({
            "sample_id": sample_id,
            "target": target_np,
            "confidence": confidence.detach().cpu().numpy(),
            "valid_mask": valid_np,
            "group_masks": group_masks,
            "predictions": predictions,
        })
        print(
            f"  {arm} {sample_id}: vertices={len(target_np)} valid={int(valid_np.sum())}",
            flush=True,
        )
        del prepared, donor, base, target, confidence
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_metrics(records, loss_kwargs)
    per_mesh: list[dict[str, Any]] = []
    for record in records:
        one = aggregate_metrics([record], loss_kwargs)
        row: dict[str, Any] = {"arm": arm, "sample_id": record["sample_id"]}
        for group in REPORT_GROUPS:
            metrics = one["conditions"]["original_rgb"][group]
            row[f"{group}_endpoint"] = metrics.get("mean_endpoint_error")
            row[f"{group}_global_cosine"] = metrics.get("global_cosine")
            row[f"{group}_norm_ratio"] = metrics.get("prediction_to_gt_global_norm_ratio")
            row[f"{group}_rgb_gap"] = one["image_dependence"][group]["endpoint_zero_minus_original"]
        per_mesh.append(row)

    return {
        "config": config,
        "checkpoint": str(arm_dir / "best.pt"),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", checkpoint_payload.get("step", -1))),
        "parameter_count": {"total": total_params, "trainable": trainable_params},
        "metrics": aggregate,
        "per_mesh": per_mesh,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, Mapping)]
    if isinstance(data, Mapping):
        for key in ("history", "records", "metrics"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]
    return []


def history_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"records": 0}
    keys = set().union(*(row.keys() for row in rows))
    step_key = next((k for k in ("optimizer_step", "global_step", "step", "epoch") if k in keys), None)
    val_key = next((k for k in ("val_loss", "validation_loss", "val/loss", "validation/loss") if k in keys), None)
    if val_key:
        candidates = [(i, float(r[val_key])) for i, r in enumerate(rows) if is_number(r.get(val_key))]
        best_i = min(candidates, key=lambda x: x[1])[0] if candidates else len(rows) - 1
    else:
        best_i = len(rows) - 1
    best = rows[best_i]
    final = rows[-1]
    return {
        "records": len(rows),
        "best_step": best.get(step_key, best_i) if step_key else best_i,
        "best_validation_loss": best.get(val_key) if val_key else None,
        "final_step": final.get(step_key, len(rows) - 1) if step_key else len(rows) - 1,
        "final_validation_loss": final.get(val_key) if val_key else None,
    }


def console_health(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "issues": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "nan": r"\bnan\b",
        "inf": r"\binf\b",
        "oom": r"out of memory|\boom\b",
        "traceback": r"traceback \(most recent call last\)",
        "killed": r"\bkilled\b",
    }
    issues = [
        {"type": name, "count": len(re.findall(pattern, text, re.IGNORECASE))}
        for name, pattern in patterns.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]
    return {"exists": True, "line_count": len(text.splitlines()), "issues": issues}


def normalized_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.setdefault("image_encoder", {})["feature_dim"] = "CAPACITY_WIDTH"
    value.setdefault("model", {})["hidden_dim"] = "CAPACITY_WIDTH"
    cap = value.get("capacity_ablation")
    if isinstance(cap, dict):
        cap["arm"] = "CAPACITY_ARM"
        cap["image_feature_dim"] = "CAPACITY_WIDTH"
        cap["hidden_dim"] = "CAPACITY_WIDTH"
    return value


def contract_audit(arm_results: Mapping[str, Mapping[str, Any]], recorded_sources: Mapping[str, str]) -> dict[str, Any]:
    configs = {arm: result["config"] for arm, result in arm_results.items()}
    normalized = [normalized_contract(configs[arm]) for arm in ARMS]
    widths: dict[str, Any] = {}
    for arm in ARMS:
        config = configs[arm]
        widths[arm] = {
            "image_feature_dim": int(config.get("image_encoder", {}).get("feature_dim", -1)),
            "hidden_dim": int(config.get("model", {}).get("hidden_dim", -1)),
            "graph_layers": int(config.get("model", {}).get("num_graph_layers", -1)),
            "first_stride": int(config.get("image_encoder", {}).get("first_stride", 1)),
            "second_stride": int(config.get("image_encoder", {}).get("second_stride", 1)),
            "seed": int(config.get("seed", -1)),
            "vertex_sampling": config.get("training", {}).get("vertex_sampling"),
            "max_optimizer_steps": config.get("multi_object_training", {}).get("max_optimizer_steps"),
            "query_validation_aug": bool(config.get("query_training", {}).get("apply_to_validation", True)),
        }
    expected_widths_match = all(
        (widths[arm]["image_feature_dim"], widths[arm]["hidden_dim"]) == EXPECTED_WIDTHS[arm]
        for arm in ARMS
    )
    return {
        "same_contract_except_capacity_fields": normalized[0] == normalized[1] == normalized[2],
        "expected_widths_match": expected_widths_match,
        "same_recorded_source_manifest": len(set(recorded_sources.values())) == 1,
        "widths_and_controls": widths,
        "recorded_source_manifests": dict(recorded_sources),
    }


def pairwise_changes(arm_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in PAIRWISE:
        s = arm_results[source]
        t = arm_results[target]
        s_orig = s["metrics"]["conditions"]["original_rgb"]
        t_orig = t["metrics"]["conditions"]["original_rgb"]
        row: dict[str, Any] = {
            "comparison": f"{target}_vs_{source}",
            "parameter_multiplier": safe_div(float(t["parameter_count"]["total"]), float(s["parameter_count"]["total"])),
        }
        for group in REPORT_GROUPS:
            s_err = float(s_orig[group]["mean_endpoint_error"])
            t_err = float(t_orig[group]["mean_endpoint_error"])
            row[f"{group}_endpoint_improvement"] = safe_div(s_err - t_err, s_err)
            row[f"{group}_global_cosine_change"] = (
                float(t_orig[group].get("global_cosine") or 0.0) - float(s_orig[group].get("global_cosine") or 0.0)
            )
            row[f"{group}_norm_ratio_change"] = (
                float(t_orig[group]["prediction_to_gt_global_norm_ratio"])
                - float(s_orig[group]["prediction_to_gt_global_norm_ratio"])
            )
            s_gap = float(s["metrics"]["image_dependence"][group]["endpoint_zero_minus_original"])
            t_gap = float(t["metrics"]["image_dependence"][group]["endpoint_zero_minus_original"])
            row[f"{group}_rgb_gap_change"] = t_gap - s_gap
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main_comparison_rows(arm_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = (
        "mean_endpoint_error",
        "median_endpoint_error",
        "mean_squared_error",
        "mean_cosine",
        "global_cosine",
        "prediction_to_gt_magnitude_ratio",
        "prediction_to_gt_global_norm_ratio",
        "group_relative_error",
    )
    for condition in CONDITIONS:
        for group in REPORT_GROUPS:
            for field in fields:
                row = {"condition": condition, "group": group, "metric": field}
                for arm in ARMS:
                    row[arm] = arm_results[arm]["metrics"]["conditions"][condition][group].get(field)
                rows.append(row)
    return rows


def report(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    lines = [
        "# Sofa50 model-capacity ablation",
        "",
        f"Analyzer version: `{summary['analyzer_version']}`",
        "",
        f"Evaluation manifest: `{summary['evaluation_manifest']}`",
        "",
        "Fresh evaluation uses exact GT queries on the original Sofa50 validation manifest.",
        "Run-local `dataset_manifest.json` files are audit-only and are never used as the lazy RGB root.",
        "",
        "## Original-RGB checkpoint comparison",
        "",
        "| metric | C0 | C1 | C2 |",
        "|---|---:|---:|---:|",
    ]
    table_metrics = [
        ("all endpoint", "all", "mean_endpoint_error"),
        ("smooth90 endpoint", "smooth_bottom_90", "mean_endpoint_error"),
        ("top10 endpoint", "high_top_10", "mean_endpoint_error"),
        ("top1 endpoint", "high_top_1", "mean_endpoint_error"),
        ("all global cosine", "all", "global_cosine"),
        ("top10 global cosine", "high_top_10", "global_cosine"),
        ("all pred/GT norm", "all", "prediction_to_gt_global_norm_ratio"),
        ("top10 pred/GT norm", "high_top_10", "prediction_to_gt_global_norm_ratio"),
    ]
    for label, group, field in table_metrics:
        vals = []
        for arm in ARMS:
            value = arms[arm]["metrics"]["conditions"]["original_rgb"][group].get(field)
            vals.append("-" if value is None else f"{float(value):.6f}")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines += ["", "## Image-dependence gaps", "", "Positive endpoint gap means correct RGB beats zero RGB.", "", "| group | C0 | C1 | C2 |", "|---|---:|---:|---:|"]
    for group in REPORT_GROUPS:
        vals = [float(arms[arm]["metrics"]["image_dependence"][group]["endpoint_zero_minus_original"]) for arm in ARMS]
        lines.append(f"| {group} | {vals[0]:+.6f} | {vals[1]:+.6f} | {vals[2]:+.6f} |")

    lines += ["", "## Zero-predictor comparison", "", "| group | C0 | C1 | C2 |", "|---|---:|---:|---:|"]
    for group in REPORT_GROUPS:
        vals = [float(arms[arm]["metrics"]["improvement_vs_zero_predictor"][group]["endpoint_relative_improvement_vs_zero_predictor"]) for arm in ARMS]
        lines.append(f"| {group} | {vals[0]:+.2%} | {vals[1]:+.2%} | {vals[2]:+.2%} |")

    lines += ["", "## Pairwise capacity changes", ""]
    for row in summary["pairwise_changes"]:
        lines.append(f"### {row['comparison']}")
        lines.append("")
        lines.append(f"- Parameter multiplier: {float(row['parameter_multiplier']):.3f}x")
        lines.append(f"- Overall endpoint improvement: {float(row['all_endpoint_improvement']):+.2%}")
        lines.append(f"- Smooth90 endpoint improvement: {float(row['smooth_bottom_90_endpoint_improvement']):+.2%}")
        lines.append(f"- Top10 endpoint improvement: {float(row['high_top_10_endpoint_improvement']):+.2%}")
        lines.append(f"- Top1 endpoint improvement: {float(row['high_top_1_endpoint_improvement']):+.2%}")
        lines.append(f"- Overall RGB-gap change: {float(row['all_rgb_gap_change']):+.6f}")
        lines.append("")

    lines += ["## Contract audit", "", "```json", json.dumps(summary["contract_audit"], indent=2), "```", "", "## Run health", ""]
    for arm in ARMS:
        health = arms[arm]["console_health"]
        issues = health.get("issues", [])
        issue_text = "none" if not issues else ", ".join(f"{x['type']}={x['count']}" for x in issues)
        history = arms[arm]["history"]
        lines.append(
            f"- {arm}: params={arms[arm]['parameter_count']['total']:,}, "
            f"best_step={history.get('best_step')}, best_val={history.get('best_validation_loss')}, console={issue_text}"
        )
    lines.append("")
    return "\n".join(lines)


def analyze(output_root: Path, cli_manifest: Path | None, device_name: str, overwrite: bool, seed: int) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    analysis_dir = output_root / "analysis_v2"
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Analysis directory is not empty: {analysis_dir}; use --overwrite")
        shutil.rmtree(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_source, recorded_sources = resolve_manifest(output_root, cli_manifest)
    print(f"Analyzer version: {VERSION}", flush=True)
    print(f"Evaluation manifest source: {manifest_source}", flush=True)
    print(f"Evaluation manifest: {manifest}", flush=True)

    sample_set_audit = audit_validation_sample_sets(output_root, manifest)
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    image_preflight = preflight_lazy_images(dataset)
    print(f"Validation meshes: {len(dataset)}", flush=True)
    print(f"Dataset root(s): {image_preflight['dataset_roots']}", flush=True)
    print(f"Checked lazy RGB files: {image_preflight['checked_image_files']}", flush=True)
    print(f"First resolved validation image: {image_preflight['first_resolved_image']}", flush=True)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    arm_results: dict[str, Any] = {}
    per_mesh_rows: list[dict[str, Any]] = []
    for arm, dirname in ARMS.items():
        arm_dir = output_root / dirname
        for required in ("config.json", "best.pt", "training_history.json"):
            if not (arm_dir / required).is_file():
                raise FileNotFoundError(f"Missing {arm} artifact: {arm_dir / required}")
        print(f"Loading {arm}: {arm_dir}", flush=True)
        print(f"Evaluating {arm} best.pt on exact validation queries...", flush=True)
        result = evaluate_arm(arm, arm_dir, dataset, device, seed, analysis_dir / "predictions" / arm)
        result["history"] = history_summary(load_history(arm_dir / "training_history.json"))
        result["console_health"] = console_health(arm_dir / "console.log")
        arm_results[arm] = result
        per_mesh_rows.extend(result["per_mesh"])

    pairs = pairwise_changes(arm_results)
    audit = contract_audit(arm_results, recorded_sources)
    summary = {
        "analyzer_version": VERSION,
        "experiment": "Sofa50 C0/C1/C2 model-capacity ablation",
        "evaluation_manifest": str(manifest),
        "evaluation_manifest_source": manifest_source,
        "image_preflight": image_preflight,
        "validation_sample_set_audit": sample_set_audit,
        "arms": arm_results,
        "pairwise_changes": pairs,
        "contract_audit": audit,
        "notes": [
            "Fresh image evaluation uses the original source manifest, never an arm-local dataset_manifest.json.",
            "C0/C1/C2 jointly scale image feature width and graph hidden width; causal attribution to either component alone is not supported.",
            "Exact-query validation disables validation query perturbation and sets query_positions=vertices.",
        ],
    }

    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (analysis_dir / "REPORT.md").write_text(report(summary), encoding="utf-8")
    write_csv(analysis_dir / "main_comparison.csv", main_comparison_rows(arm_results))
    write_csv(analysis_dir / "pairwise_changes.csv", pairs)
    write_csv(analysis_dir / "per_mesh_metrics.csv", per_mesh_rows)
    print(f"Wrote: {analysis_dir / 'REPORT.md'}", flush=True)
    print(f"Wrote: {analysis_dir / 'summary.json'}", flush=True)
    print(f"Wrote: {analysis_dir / 'main_comparison.csv'}", flush=True)
    print(f"Wrote: {analysis_dir / 'pairwise_changes.csv'}", flush=True)
    print(f"Wrote: {analysis_dir / 'per_mesh_metrics.csv'}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Sofa50 C0/C1/C2 model-capacity runs.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional explicit original Sofa50 gt_query_manifest.json. By default read run_config.json -> source_manifest.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    analyze(args.output_root, args.manifest, args.device, args.overwrite, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

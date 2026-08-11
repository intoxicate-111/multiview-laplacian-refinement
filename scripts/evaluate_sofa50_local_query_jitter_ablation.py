#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.learned_laplacian.diagnostics import _amp_settings, _loss_kwargs
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _evaluate_dataset,
    _prepare_object_static,
)
from mlr.learned_laplacian.trainer import _seed_everything, load_checkpoint


ARMS = ("A_no_jitter", "B_local_jitter")
SPLITS = ("validation", "test")
CONDITIONS = ("correct_rgb", "zero_rgb")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No best checkpoint in {run_dir}")


def state_hash(config: Mapping[str, Any]) -> str:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.setdefault("local_query_jitter", {})["enabled"] = "CONTROLLED_ARM_VARIABLE"
    result.setdefault("experiment_metadata", {})["arm"] = "CONTROLLED_ARM_LABEL"
    return result


def prepare_split(
    manifest: Path, split: str, config: Mapping[str, Any]
) -> tuple[tuple[Any, ...], dict[str, dict[str, Any]]]:
    dataset = PreparedMeshDataset.from_manifest(manifest, split)
    prepared = []
    metadata = {}
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        sample_id = str(sample["sample_id"])
        prepared.append(
            _prepare_object_static(
                sample, config, keep_image_payload=True, keep_projection=True
            )
        )
        metadata[sample_id] = dict(sample.get("metadata", {}))
    return tuple(prepared), metadata


def evaluate(
    run_dir: Path,
    config: Mapping[str, Any],
    prepared: Sequence[Any],
    device: torch.device,
    *,
    zero_rgb: bool,
) -> tuple[float, dict[str, dict[str, Any]]]:
    model = _build_model(config, None, zero_rgb).to(device)
    load_checkpoint(checkpoint(run_dir), model, map_location=device)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    loss, metrics = _evaluate_dataset(
        model,
        prepared,
        config,
        device,
        _loss_kwargs(config),
        cache_on_device=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        query_settings=None,
        augment_queries=False,
    )
    del model
    torch.cuda.empty_cache()
    return loss, metrics


def metric_row(
    arm: str,
    split: str,
    condition: str,
    sample_id: str,
    value: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    raw = value["recovered_raw_space"]
    target = value["target_space"]
    perturbation = metadata.get("perturbation", {})
    visibility = metadata.get("renderer_visibility", {})
    return {
        "arm": arm,
        "split": split,
        "condition": condition,
        "sample_id": sample_id,
        "object_id": metadata.get("object_id"),
        "variant_index": metadata.get("variant_index"),
        "vertex_count": value["vertex_count"],
        "loss": value["loss"],
        "normalized_endpoint": target["vector_endpoint_error"],
        "normalized_global_cosine": target["global_cosine"],
        "normalized_prediction_to_target_norm_ratio": target["prediction_to_target_norm_ratio"],
        "raw_endpoint": raw["vector_endpoint_error"],
        "raw_top10_endpoint": raw["top_10_percent_vector_endpoint_error"],
        "raw_top1_endpoint": raw["top_1_percent_vector_endpoint_error"],
        "raw_global_cosine": raw["global_cosine"],
        "raw_prediction_to_target_norm_ratio": raw["prediction_to_target_norm_ratio"],
        "raw_top10_cosine": raw["top_10_percent_cosine"],
        "raw_top1_cosine": raw["top_1_percent_cosine"],
        "original_mean_offset_over_h": perturbation.get("mean_offset_over_h"),
        "original_p95_offset_over_h": perturbation.get("p95_offset_over_h"),
        "original_max_offset_over_h": perturbation.get("max_offset_over_h"),
        "zero_visible_vertex_ratio": visibility.get("zero_visible_vertex_ratio"),
    }


def mean_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "loss",
        "normalized_endpoint",
        "normalized_global_cosine",
        "normalized_prediction_to_target_norm_ratio",
        "raw_endpoint",
        "raw_top10_endpoint",
        "raw_top1_endpoint",
        "raw_global_cosine",
        "raw_prediction_to_target_norm_ratio",
        "raw_top10_cosine",
        "raw_top1_cosine",
    )
    return {
        "sample_count": len(rows),
        **{key: float(np.mean([float(row[key]) for row in rows])) for key in keys},
    }


def paired_summary(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and row["condition"] == "correct_rgb"]
    by_key = {(row["arm"], row["sample_id"]): row for row in selected}
    sample_ids = sorted({str(row["sample_id"]) for row in selected})
    metrics = ("raw_endpoint", "raw_top10_endpoint", "raw_top1_endpoint", "raw_global_cosine")
    output: dict[str, Any] = {"paired_samples": len(sample_ids), "B_minus_A": {}}
    for metric in metrics:
        deltas = np.asarray(
            [float(by_key[(ARMS[1], sid)][metric]) - float(by_key[(ARMS[0], sid)][metric]) for sid in sample_ids]
        )
        lower_is_better = "cosine" not in metric
        output["B_minus_A"][metric] = {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "B_better_samples": int(np.sum(deltas < 0 if lower_is_better else deltas > 0)),
        }
    endpoint_improvement = np.asarray(
        [float(by_key[(ARMS[0], sid)]["raw_endpoint"]) - float(by_key[(ARMS[1], sid)]["raw_endpoint"]) for sid in sample_ids]
    )
    correlations = {}
    for field in (
        "original_mean_offset_over_h",
        "original_p95_offset_over_h",
        "zero_visible_vertex_ratio",
        "vertex_count",
    ):
        values = np.asarray([float(by_key[(ARMS[0], sid)][field]) for sid in sample_ids])
        correlations[field] = (
            None
            if len(values) < 2 or values.std() == 0 or endpoint_improvement.std() == 0
            else float(np.corrcoef(values, endpoint_improvement)[0, 1])
        )
    output["correlation_with_A_minus_B_raw_endpoint"] = correlations
    output["largest_B_raw_endpoint"] = sorted(
        (
            {
                "sample_id": sid,
                "A": by_key[(ARMS[0], sid)]["raw_endpoint"],
                "B": by_key[(ARMS[1], sid)]["raw_endpoint"],
                "B_minus_A": float(by_key[(ARMS[1], sid)]["raw_endpoint"]) - float(by_key[(ARMS[0], sid)]["raw_endpoint"]),
            }
            for sid in sample_ids
        ),
        key=lambda row: float(row["B"]),
        reverse=True,
    )[:5]
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--a-run", required=True, type=Path)
    parser.add_argument("--b-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA")
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {ARMS[0]: args.a_run.resolve(), ARMS[1]: args.b_run.resolve()}
    configs = {arm: read_json(run_dirs[arm] / "config.json") for arm in ARMS}
    if controlled_config(configs[ARMS[0]]) != controlled_config(configs[ARMS[1]]):
        raise ValueError("Arm configs differ outside local jitter enablement and arm label")
    initial_hashes = {arm: state_hash(configs[arm]) for arm in ARMS}
    if len(set(initial_hashes.values())) != 1:
        raise ValueError("Seeded initial parameter states differ")

    all_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    for split in SPLITS:
        prepared, metadata = prepare_split(manifest, split, configs[ARMS[1]])
        aggregate[split] = {}
        for arm in ARMS:
            aggregate[split][arm] = {}
            for condition in CONDITIONS:
                loss, metrics = evaluate(
                    run_dirs[arm], configs[arm], prepared, device,
                    zero_rgb=condition == "zero_rgb",
                )
                rows = [
                    metric_row(arm, split, condition, sample_id, value, metadata[sample_id])
                    for sample_id, value in metrics.items()
                ]
                all_rows.extend(rows)
                aggregate[split][arm][condition] = {"dataset_loss": loss, **mean_rows(rows)}

    native = {arm: read_json(run_dirs[arm] / "metrics.json") for arm in ARMS}
    summary = {
        "experiment": "Sofa50 28-view fixed-current local query jitter ablation",
        "manifest": str(manifest),
        "contract": {
            "paired_config_equal_except_jitter_enablement_and_arm_label": True,
            "seeded_initial_parameter_sha256": initial_hashes,
            "training_jitter": "eta_i=h_i*clip_l2(N(0,0.003^2 I),0.009)",
            "validation_test_jitter": False,
            "proxy_target_h_graph_operator_frozen": True,
        },
        "native_training": {
            arm: {
                "best_validation_loss": native[arm]["best_selection_loss"],
                "runtime_seconds": native[arm]["runtime_seconds"],
                "optimizer_steps": native[arm]["optimizer_steps"],
                "best_epoch": native[arm]["best_epoch"],
                "peak_gpu_memory_mb": native[arm].get("peak_gpu_memory_mb"),
            }
            for arm in ARMS
        },
        "deterministic_evaluation": aggregate,
        "paired": {split: paired_summary(all_rows, split) for split in SPLITS},
        "per_sample": all_rows,
    }
    write_json(output_dir / "prediction_summary.json", summary)
    write_csv(output_dir / "prediction_per_sample.csv", all_rows)
    print(json.dumps({"output": str(output_dir), "paired": summary["paired"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

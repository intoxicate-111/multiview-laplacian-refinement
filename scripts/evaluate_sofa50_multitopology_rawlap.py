#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mlr.data import Mesh
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    _infer_one,
    _recover_raw_one,
    _run_config,
)
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.multitopology_rawlap import raw_uniform_laplacian


ARMS = ("old_960_HF", "new_multitopology_rawlap")
ERROR_FIELDS = (
    "raw_epe",
    "raw_rms",
    "raw_max",
    "raw_cosine",
    "recovery_weighted_raw_rms",
    "bottom90_epe",
    "top10_epe",
    "top1_epe",
)
RECOVERY_FIELDS = (
    "reconstruction_chamfer",
    "reconstruction_point_to_surface",
    "reconstruction_normal_consistency",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(run_dir: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = run_dir / "checkpoint_latest.pt"
    metrics = run_dir / "metrics.json"
    if not checkpoint.is_file() or not metrics.is_file():
        raise FileNotFoundError(f"Incomplete run: {run_dir}")
    config = _run_config(run_dir)
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "optimizer_steps": int(payload.get("optimizer_steps", -1)),
        "config": config,
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }


def raw_gt_magnitude_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction[valid].double()
    target = target[valid].double()
    weight = weight[valid].double().clamp_min(0.0)
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target, dim=-1)
    order = torch.argsort(target_magnitude, stable=True)
    count = len(order)
    top10_count = max(1, int(math.ceil(0.10 * count)))
    top1_count = max(1, int(math.ceil(0.01 * count)))
    bottom90 = order[: count - top10_count]
    top10 = order[count - top10_count :]
    top1 = order[count - top1_count :]
    cosine = F.cosine_similarity(
        prediction.reshape(1, -1), target.reshape(1, -1), dim=-1, eps=1e-12
    )
    return {
        "raw_epe": float(error.mean()),
        "raw_rms": float(torch.sqrt(error.square().mean())),
        "raw_max": float(error.max()),
        "raw_cosine": float(cosine.item()),
        "recovery_weighted_raw_rms": float(
            torch.sqrt((weight * error.square()).sum() / weight.sum().clamp_min(1e-12))
        ),
        "bottom90_epe": float(error[bottom90].mean()),
        "top10_epe": float(error[top10].mean()),
        "top1_epe": float(error[top1].mean()),
    }


def sample_contract(static: Mapping[str, Any], dataset_contract: str) -> dict[str, Any]:
    metadata = dict(static.get("metadata", {}))
    if dataset_contract == "legacy_current_proxy":
        positions = static["target_positions"].detach().cpu().double().numpy()
        faces = static["faces"].detach().cpu().numpy()
        recomputed = torch.as_tensor(
            raw_uniform_laplacian(Mesh(positions, faces)), dtype=torch.float32
        )
        saved = static.get("raw_laplacian_target", static["laplacian_target"]).float()
        error = float(torch.max(torch.abs(recomputed - saved)))
        return {
            "sample_id": str(static["sample_id"]),
            "legacy_current_proxy_formula_max_abs_error": error,
            "legacy_current_proxy_formula_pass": error <= 1e-7,
            "target_positions_present": "target_positions" in static,
        }
    clean_vertices = static.get("clean_reference_vertices", static.get("gt_vertices"))
    clean_faces = static.get("clean_reference_faces", static.get("gt_faces"))
    return {
        "sample_id": str(static["sample_id"]),
        "faces_equal": bool(torch.equal(static["faces"], clean_faces)),
        "vertex_count_equal": int(static["vertices"].shape[0]) == int(clean_vertices.shape[0]),
        "target_mode_raw": metadata.get("target_mode") == "raw_laplacian",
        "target_scaling_applied": bool(metadata.get("target_scaling_applied", True)),
        "proxy_used": bool(metadata.get("proxy_used", True)),
        "target_transfer_used": bool(metadata.get("target_transfer_used", True)),
    }


def evaluate(args: argparse.Namespace) -> None:
    arms = (args.old_arm_name, args.new_arm_name)
    if len(set(arms)) != 2:
        raise ValueError("Evaluation arm names must be distinct.")
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    payload = read_json(manifest)
    splits = tuple(
        split
        for split in ("validation", "test")
        if any(str(row.get("split")) == split for row in payload["samples"])
    )
    datasets = {split: PreparedMeshDataset.from_manifest(manifest, split) for split in splits}
    specs = {
        arms[0]: load_spec(args.old_run.resolve(), device),
        arms[1]: load_spec(args.new_run.resolve(), device),
    }
    configs = {arm: spec["config"] for arm, spec in specs.items()}
    checkpoint_steps = {arm: spec["optimizer_steps"] for arm, spec in specs.items()}
    feature_modes = {
        arm: config.get("image_encoder", {}).get("feature_construction", {}).get("mode")
        for arm, config in configs.items()
    }
    preflight = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "checkpoint_steps": checkpoint_steps,
        "feature_modes": feature_modes,
        "target_modes": {arm: config.get("target_mode") for arm, config in configs.items()},
        "same_architecture": all(
            configs[arms[0]].get(key) == configs[arms[1]].get(key)
            for key in ("model", "image_encoder", "input_mode")
        ),
        "same_recovery": configs[arms[0]].get("recovery") == configs[arms[1]].get("recovery"),
        "arms": list(arms),
        "dataset_contract": args.dataset_contract,
    }
    preflight["passed"] = bool(
        all(value == 20_000 for value in checkpoint_steps.values())
        and all(value == "original_plus_high_frequency" for value in feature_modes.values())
        and all(config.get("target_mode") == "raw_laplacian" for config in configs.values())
        and preflight["same_architecture"]
        and preflight["same_recovery"]
    )
    if not preflight["passed"]:
        write_json(output / "preflight_failed.json", preflight)
        raise RuntimeError("Evaluation preflight failed.")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for split, dataset in datasets.items():
        for index in range(len(dataset)):
            if index % args.shard_count != args.shard_index:
                continue
            static = dataset.load_static(index)
            contract_rows.append(sample_contract(static, args.dataset_contract))
            metadata = dict(static.get("metadata", {}))
            for arm, spec in specs.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                values = _infer_one(dataset, index, spec, device, current_faces=static["faces"])
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - start
                metrics = raw_gt_magnitude_metrics(
                    values["prediction_raw"],
                    values["target_raw"],
                    values["recovery_weight"],
                    values["valid"],
                )
                prediction_rows.append(
                    {
                        "split": split,
                        "arm": arm,
                        "sample_id": str(static["sample_id"]),
                        "object_id": metadata.get("object_id"),
                        "variant": metadata.get("variant"),
                        "vertices": int(static["vertices"].shape[0]),
                        "faces": int(static["faces"].shape[0]),
                        "inference_seconds": inference_seconds,
                        "mean_confidence": float(values["confidence"].mean()),
                        **metrics,
                    }
                )
                valid = values["valid"].numpy().astype(bool)
                prefix = f"{split}__{arm}"
                arrays[f"{prefix}__prediction"].append(
                    values["prediction_raw"].numpy()[valid].astype(np.float64)
                )
                arrays[f"{prefix}__target"].append(
                    values["target_raw"].numpy()[valid].astype(np.float64)
                )
                arrays[f"{prefix}__weight"].append(
                    values["recovery_weight"].numpy()[valid].astype(np.float64)
                )
                if split == "test":
                    start = time.perf_counter()
                    recovery, _ = _recover_raw_one(
                        static,
                        values["prediction_raw"],
                        values["prediction_normalized"],
                        values["confidence"],
                        output / "reconstruction" / arm / str(static["sample_id"]),
                        spec["config"],
                    )
                    recovery_rows.append(
                        {
                            "arm": arm,
                            "sample_id": str(static["sample_id"]),
                            "object_id": metadata.get("object_id"),
                            "variant": metadata.get("variant"),
                            "recovery_seconds": time.perf_counter() - start,
                            **recovery,
                        }
                    )
                print(f"{split} {arm} {static['sample_id']} epe={metrics['raw_epe']:.8g}", flush=True)
                del values
                torch.cuda.empty_cache()
    shard = output / "shards"
    write_json(
        shard / f"shard_{args.shard_index:02d}.json",
        {
            "arms": list(arms),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "preflight": preflight,
            "dataset_contract": args.dataset_contract,
            "prediction_rows": prediction_rows,
            "recovery_rows": recovery_rows,
            "contract_rows": contract_rows,
        },
    )
    np.savez_compressed(
        shard / f"arrays_shard_{args.shard_index:02d}.npz",
        **{
            name: np.concatenate(chunks, axis=0)
            for name, chunks in arrays.items()
            if chunks
        },
    )


def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [read_json(output / "shards" / f"shard_{index:02d}.json") for index in range(args.shard_count)]
    arms = tuple(str(value) for value in payloads[0].get("arms", ARMS))
    if len(arms) != 2 or len(set(arms)) != 2:
        raise RuntimeError(f"Invalid evaluation arms: {arms}")
    if any(tuple(str(value) for value in payload.get("arms", ARMS)) != arms for payload in payloads):
        raise RuntimeError("Shard evaluation arms differ.")
    prediction = [row for payload in payloads for row in payload["prediction_rows"]]
    recovery = [row for payload in payloads for row in payload["recovery_rows"]]
    contracts = [row for payload in payloads for row in payload["contract_rows"]]
    dataset_contract = str(payloads[0]["dataset_contract"])
    if any(str(payload["dataset_contract"]) != dataset_contract for payload in payloads):
        raise RuntimeError("Shard dataset contracts differ.")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(args.shard_count):
        with np.load(output / "shards" / f"arrays_shard_{index:02d}.npz") as archive:
            for name in archive.files:
                arrays[name].append(np.asarray(archive[name]))
    aggregate_prediction = []
    for split in sorted({str(row["split"]) for row in prediction}):
        for arm in arms:
            rows = [row for row in prediction if row["split"] == split and row["arm"] == arm]
            prefix = f"{split}__{arm}"
            global_metrics = raw_gt_magnitude_metrics(
                torch.from_numpy(np.concatenate(arrays[f"{prefix}__prediction"])),
                torch.from_numpy(np.concatenate(arrays[f"{prefix}__target"])),
                torch.from_numpy(np.concatenate(arrays[f"{prefix}__weight"])),
                torch.ones(
                    sum(len(chunk) for chunk in arrays[f"{prefix}__target"]),
                    dtype=torch.bool,
                ),
            )
            aggregate_prediction.append(
                {
                    "split": split,
                    "arm": arm,
                    "samples": len(rows),
                    "percentile_contract": "global_by_GT_raw_laplacian_magnitude",
                    **global_metrics,
                    "runtime_seconds": mean(rows, "inference_seconds"),
                }
            )
    aggregate_recovery = []
    for arm in arms:
        rows = [row for row in recovery if row["arm"] == arm]
        aggregate_recovery.append(
            {
                "arm": arm,
                "samples": len(rows),
                **{field: mean(rows, field) for field in RECOVERY_FIELDS},
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
                "improved_over_initial": int(sum(bool(row["improved_over_initial"]) for row in rows)),
                "runtime_seconds": mean(rows, "recovery_seconds"),
            }
        )
    old_prediction = {str(row["sample_id"]): row for row in prediction if row["arm"] == arms[0]}
    new_prediction = {str(row["sample_id"]): row for row in prediction if row["arm"] == arms[1]}
    old_recovery = {str(row["sample_id"]): row for row in recovery if row["arm"] == arms[0]}
    new_recovery = {str(row["sample_id"]): row for row in recovery if row["arm"] == arms[1]}
    paired_rows = []
    for sample_id in sorted(new_prediction):
        old = old_prediction[sample_id]
        new = new_prediction[sample_id]
        row = {"sample_id": sample_id, "split": new["split"], "variant": new.get("variant")}
        for field in ERROR_FIELDS:
            row[f"old_{field}"] = old[field]
            row[f"new_{field}"] = new[field]
            row[f"new_lower_{field}"] = float(new[field]) < float(old[field]) if field != "raw_cosine" else float(new[field]) > float(old[field])
        if sample_id in new_recovery:
            old_r, new_r = old_recovery[sample_id], new_recovery[sample_id]
            for field in RECOVERY_FIELDS:
                row[f"old_{field}"] = old_r[field]
                row[f"new_{field}"] = new_r[field]
        paired_rows.append(row)
    if dataset_contract == "clean_reference":
        contract_audit = {
            "preflight_passed": all(bool(payload["preflight"]["passed"]) for payload in payloads),
            "all_faces_equal": all(bool(row["faces_equal"]) for row in contracts),
            "all_vertex_counts_equal": all(bool(row["vertex_count_equal"]) for row in contracts),
            "all_targets_raw": all(bool(row["target_mode_raw"]) for row in contracts),
            "no_target_scaling": all(not bool(row["target_scaling_applied"]) for row in contracts),
            "no_proxy": all(not bool(row["proxy_used"]) for row in contracts),
            "no_target_transfer": all(not bool(row["target_transfer_used"]) for row in contracts),
        }
    else:
        contract_audit = {
            "preflight_passed": all(bool(payload["preflight"]["passed"]) for payload in payloads),
            "legacy_current_proxy_formula_pass": all(
                bool(row["legacy_current_proxy_formula_pass"]) for row in contracts
            ),
            "target_positions_present": all(bool(row["target_positions_present"]) for row in contracts),
            "maximum_formula_error": max(
                float(row["legacy_current_proxy_formula_max_abs_error"])
                for row in contracts
            ),
        }
    contract_audit["dataset_contract"] = dataset_contract
    contract_audit["passed"] = all(
        bool(value) for key, value in contract_audit.items()
        if key not in {"dataset_contract", "maximum_formula_error"}
    ) and float(contract_audit.get("maximum_formula_error", 0.0)) <= 1e-7
    summary = {
        "arms": list(arms),
        "contract_audit": contract_audit,
        "prediction": aggregate_prediction,
        "recovery": aggregate_recovery,
        "paired_sample_count": len(paired_rows),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "contract_audit.json", contract_audit)
    write_csv(output / "prediction_per_sample.csv", prediction)
    write_csv(output / "recovery_per_sample.csv", recovery)
    write_csv(output / "paired_old_vs_new.csv", paired_rows)
    write_csv(output / "prediction_aggregate.csv", aggregate_prediction)
    write_csv(output / "recovery_aggregate.csv", aggregate_recovery)
    lines = ["# Sofa50 multi-topology raw-Laplacian evaluation", "", f"Contract audit: **{str(contract_audit['passed']).lower()}**.", "", "## Prediction", "", "| Split | Arm | Raw EPE | Raw RMS | Raw max | Cosine | Weighted RMS | Bottom90 | Top10 | Top1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate_prediction:
        lines.append(f"| {row['split']} | {row['arm']} | {row['raw_epe']:.9g} | {row['raw_rms']:.9g} | {row['raw_max']:.9g} | {row['raw_cosine']:.9g} | {row['recovery_weighted_raw_rms']:.9g} | {row['bottom90_epe']:.9g} | {row['top10_epe']:.9g} | {row['top1_epe']:.9g} |")
    lines.extend(["", "## Recovery", "", "| Arm | Chamfer | P2S | Normal | Flips | New degenerates | Improved |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in aggregate_recovery:
        lines.append(f"| {row['arm']} | {row['reconstruction_chamfer']:.9g} | {row['reconstruction_point_to_surface']:.9g} | {row['reconstruction_normal_consistency']:.9g} | {row['introduced_flipped_faces']} | {row['new_degenerate_faces']} | {row['improved_over_initial']}/{row['samples']} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--new-run", type=Path)
    parser.add_argument("--old-arm-name", default=ARMS[0])
    parser.add_argument("--new-arm-name", default=ARMS[1])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--dataset-contract",
        choices=("clean_reference", "legacy_current_proxy"),
        default="clean_reference",
    )
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.old_run is None or args.new_run is None:
            parser.error("--old-run and --new-run are required unless --merge-only is used")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

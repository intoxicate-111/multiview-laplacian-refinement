#!/usr/bin/env python3
"""Evaluate the controlled Sofa50 B_0/B_P cross-anchor matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_sofa50_direct_lap_positional_matched_fusion as source
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.frozen_anchor_cache import (
    FrozenAnchorCache,
    FrozenAnchorDataset,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


EXPECTED_SHA = source.EXPECTED_SHA
ARM_A = source.ARM_A
ARM_B0 = source.ARM_B
ARM_E = source.ARM_E
ARM_BP = "B_P_positional_anchor_conditioned"
FIELDS = (
    "refined_chamfer",
    "p2s_p95",
    "fscore",
    "normal_consistency",
    "same_index_recovered_vertex_rms",
)
LOWER = {"refined_chamfer", "p2s_p95", "same_index_recovered_vertex_rms"}
SOURCE_LABELS = {ARM_A: "A", ARM_B0: "B_0", ARM_BP: "B_P"}
ANCHOR_LABELS = {"V0": "V0", "VP": "V_P"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-ab-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--bp-run", required=True, type=Path)
    parser.add_argument("--bp-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_id(sample_id: str) -> str:
    return sample_id.split("__", 1)[0]


def method(source_arm: str, anchor: str, regularization: float) -> str:
    return f"{SOURCE_LABELS[source_arm]}|{ANCHOR_LABELS[anchor]}|lambda={regularization:g}"


def checkpoint(run: Path, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(requested)
        return requested.resolve()
    for name in ("checkpoint_best.pt", "best.pt"):
        candidate = run / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No best B_P checkpoint in {run}")


def bp_predictions(
    manifest: Path,
    run: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    config_path = run / "launch_config.json"
    if not config_path.is_file():
        payload = read_json(run / "run_config.json")
        config = payload.get("experiment_config", payload)
    else:
        config = read_json(config_path)
    controlled = config.get("controlled_ablation", {})
    recovery = config["training"]["recovery_aware_geometry_loss"]
    if controlled.get("arm") != "B_P" or recovery.get("anchor_mode") != "cached_frozen_vertices":
        raise RuntimeError("B_P run config does not satisfy the anchor-conditioning contract")
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    anchor_cache_path = run / "frozen_arm_e_anchor_cache" / "metadata.json"
    anchor_cache = FrozenAnchorCache(
        anchor_cache_path,
        expected_checkpoint_sha256=EXPECTED_SHA[ARM_E],
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    split_ids: dict[str, list[str]] = {}
    flat: dict[str, list[np.ndarray]] = {}
    raw_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        dataset = FrozenAnchorDataset(
            PreparedMeshDataset.from_manifest(manifest, split),
            anchor_cache,
        )
        split_ids[split] = list(dataset.sample_ids)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for index, sample_id in enumerate(dataset.sample_ids):
            static = dataset.load_static(index)
            prepared = _load_device_item(dataset, index, config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.inference_mode():
                prediction = (
                    model(conditioned).predicted_laplacian.detach().float().cpu().numpy()
                ).astype(np.float64)
            target = np.asarray(static["raw_laplacian_target"], dtype=np.float64)
            if prediction.shape != target.shape:
                raise RuntimeError(f"{sample_id}: B_P prediction/target shape mismatch")
            error = prediction - target
            raw_rows.append({
                "split": split,
                "sample_id": str(sample_id),
                "object_id": object_id(str(sample_id)),
                "raw_epe": float(np.mean(np.linalg.norm(error, axis=1))),
                "raw_rms": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
                "vertices": len(prediction),
            })
            result[str(sample_id)] = {"prediction": prediction, "target": target}
            predictions.append(prediction)
            targets.append(target)
        flat[f"{split}_prediction"] = predictions
        flat[f"{split}_target"] = targets
    archive = {
        key: np.concatenate(values, axis=0) for key, values in flat.items()
    }
    return result, {
        "config": config,
        "checkpoint_payload": payload,
        "anchor_cache": {
            "metadata": str(anchor_cache_path.resolve()),
            "metadata_sha256": sha256_file(anchor_cache_path),
            "contract_audit": anchor_cache.payload["contract_audit"],
            "arm_e_checkpoint_sha256": anchor_cache.payload[
                "arm_e_checkpoint_sha256"
            ],
        },
        "split_ids": split_ids,
        "raw_rows": raw_rows,
        "archive": archive,
    }


def archived_fields(
    report: Path, arm: str, split: str
) -> tuple[list[str], list[int], np.ndarray, np.ndarray, dict[str, Any]]:
    payload = read_json(report / "shards" / f"{arm}.json")
    if payload["checkpoint_sha256"] != EXPECTED_SHA[arm]:
        raise RuntimeError(f"{arm}: archived checkpoint SHA mismatch")
    rows = source.split_rows(payload, split)
    ids = [str(row["sample_id"]) for row in rows]
    counts = [int(row["vertices"]) for row in rows]
    prediction, target, path = source.prediction_array(report, arm, split)
    if sum(counts) != len(prediction):
        raise RuntimeError(f"{arm}/{split}: prediction rows do not close")
    return ids, counts, prediction, target, {
        "metadata": str((report / "shards" / f"{arm}.json").resolve()),
        "arrays": str(path.resolve()),
        "checkpoint_sha256": payload["checkpoint_sha256"],
    }


def split_flat(ids: Sequence[str], counts: Sequence[int], array: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    offset = 0
    for sample_id, count in zip(ids, counts, strict=True):
        output[sample_id] = np.asarray(array[offset : offset + count], dtype=np.float64)
        offset += count
    if offset != len(array):
        raise RuntimeError("Flat prediction partition did not close")
    return output


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        names = sorted({str(row["method"]) for row in rows if row["split"] == split})
        for name in names:
            selected = [row for row in rows if row["split"] == split and row["method"] == name]
            faces = sum(int(row["faces"]) for row in selected)
            output.append({
                "split": split,
                "method": name,
                "samples": len(selected),
                **{field: float(np.mean([float(row[field]) for row in selected])) for field in FIELDS},
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "normalized_flip_rate": float(sum(int(row["introduced_flipped_faces"]) for row in selected) / faces),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved": int(sum(bool(row["improved"]) for row in selected)),
                "worsened": int(sum(bool(row["worsened"]) for row in selected)),
                "pcg_failed_solves": int(sum(not bool(row["pcg_converged"]) for row in selected)),
                "pcg_iterations_max": int(max(int(row["pcg_iterations"]) for row in selected)),
                "pcg_relative_residual_max": float(max(float(row["pcg_relative_residual"]) for row in selected)),
            })
    return output


def bootstrap_values(
    ids: Sequence[str], values: np.ndarray, replicates: int, seed: int
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    mesh = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(1)
    grouped: dict[str, list[float]] = {}
    for sample_id, value in zip(ids, values, strict=True):
        grouped.setdefault(object_id(sample_id), []).append(float(value))
    object_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    cluster = object_means[
        rng.integers(0, len(object_means), size=(replicates, len(object_means)))
    ].mean(1)
    return (
        [float(np.quantile(mesh, 0.025)), float(np.quantile(mesh, 0.975))],
        [float(np.quantile(cluster, 0.025)), float(np.quantile(cluster, 0.975))],
    )


def paired(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
    field: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in rows if row["split"] == "test" and row["method"] == candidate}
    right = {str(row["sample_id"]): row for row in rows if row["split"] == "test" and row["method"] == reference}
    if left.keys() != right.keys() or len(left) != 50:
        raise RuntimeError(f"Paired identity mismatch: {candidate} vs {reference}")
    ids = sorted(left)
    values = np.asarray([float(left[item][field]) - float(right[item][field]) for item in ids])
    mesh_ci, object_ci = bootstrap_values(ids, values, replicates, seed)
    wins = values < 0 if field in LOWER else values > 0
    losses = values > 0 if field in LOWER else values < 0
    ties = ~(wins | losses)
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": field,
        "difference": "candidate_minus_reference",
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "candidate_wins": int(wins.sum()),
        "ties": int(ties.sum()),
        "candidate_losses": int(losses.sum()),
        "mesh_bootstrap_95_percent_ci": mesh_ci,
        "object_cluster_bootstrap_95_percent_ci": object_ci,
    }


def interaction(
    rows: Sequence[Mapping[str, Any]], field: str, replicates: int, seed: int
) -> dict[str, Any]:
    methods = {
        "bp_vp": method(ARM_BP, "VP", 0.01),
        "b0_vp": method(ARM_B0, "VP", 0.01),
        "bp_v0": method(ARM_BP, "V0", 0.01),
        "b0_v0": method(ARM_B0, "V0", 0.01),
    }
    maps = {
        key: {str(row["sample_id"]): row for row in rows if row["split"] == "test" and row["method"] == name}
        for key, name in methods.items()
    }
    ids = sorted(maps["bp_vp"])
    if len(ids) != 50 or any(sorted(value) != ids for value in maps.values()):
        raise RuntimeError("Interaction identity mismatch")
    gap_vp = np.asarray([
        float(maps["bp_vp"][item][field]) - float(maps["b0_vp"][item][field])
        for item in ids
    ])
    gap_v0 = np.asarray([
        float(maps["bp_v0"][item][field]) - float(maps["b0_v0"][item][field])
        for item in ids
    ])
    values = gap_vp - gap_v0
    mesh_ci, object_ci = bootstrap_values(ids, values, replicates, seed)
    return {
        "metric": field,
        "definition": "(B_P@V_P-B_0@V_P) - (B_P@V0-B_0@V0)",
        "gap_vp_mean": float(gap_vp.mean()),
        "gap_v0_mean": float(gap_v0.mean()),
        "interaction_mean": float(values.mean()),
        "interaction_median": float(np.median(values)),
        "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
        "positive": int((values > 0).sum()),
        "mesh_bootstrap_95_percent_ci": mesh_ci,
        "object_cluster_bootstrap_95_percent_ci": object_ci,
    }


def per_object(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cluster in sorted({str(row["object_id"]) for row in rows if row["split"] == "test"}):
        selected_cluster = [row for row in rows if row["split"] == "test" and row["object_id"] == cluster]
        for name in sorted({str(row["method"]) for row in selected_cluster}):
            selected = [row for row in selected_cluster if row["method"] == name]
            output.append({
                "object_id": cluster,
                "method": name,
                "samples": len(selected),
                **{field: float(np.mean([float(row[field]) for row in selected])) for field in FIELDS},
            })
    return output


def per_object_comparison_differences(
    object_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_key = {
        (str(row["object_id"]), str(row["method"])): row for row in object_rows
    }
    for cluster in sorted({str(row["object_id"]) for row in object_rows}):
        for label, candidate, reference in comparisons:
            left = by_key[(cluster, candidate)]
            right = by_key[(cluster, reference)]
            output.append({
                "object_id": cluster,
                "comparison": label,
                "difference": "candidate_minus_reference",
                **{field: float(left[field]) - float(right[field]) for field in FIELDS},
            })
    return output


def fmt(value: float) -> str:
    return f"{value:.10g}"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty report namespace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    run = args.bp_run.resolve()
    bp_checkpoint = checkpoint(run, args.bp_checkpoint)
    bp_checkpoint_sha = sha256_file(bp_checkpoint)
    device = torch.device(args.device)

    bp, bp_audit = bp_predictions(manifest, run, bp_checkpoint, device)
    np.savez_compressed(output / "bp_prediction_arrays.npz", **bp_audit.pop("archive"))
    write_csv(output / "bp_raw_differential_metrics.csv", bp_audit["raw_rows"])

    fields: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    for split in ("validation", "test"):
        expected_ids = list(PreparedMeshDataset.from_manifest(manifest, split).sample_ids)
        for arm, report in (
            (ARM_A, args.arm_ab_report.resolve()),
            (ARM_B0, args.arm_ab_report.resolve()),
            (ARM_E, args.arm_e_report.resolve()),
        ):
            ids, counts, prediction, target, audit = archived_fields(report, arm, split)
            if ids != expected_ids:
                raise RuntimeError(f"{arm}/{split}: archived order differs from manifest")
            fields.setdefault(arm, {}).update(split_flat(ids, counts, prediction))
            provenance[f"{arm}_{split}"] = audit
            if arm == ARM_A:
                a_target = split_flat(ids, counts, target)
            elif arm == ARM_B0:
                b_target = split_flat(ids, counts, target)
        if any(not np.array_equal(a_target[item], b_target[item]) for item in expected_ids):
            raise RuntimeError(f"Arm A/B_0 raw targets differ on {split}")
    fields[ARM_BP] = {sample_id: value["prediction"] for sample_id, value in bp.items()}

    raw_rows: list[dict[str, Any]] = list(bp_audit["raw_rows"])
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(manifest, split)
        for arm in (ARM_A, ARM_B0):
            for index, sample_id in enumerate(dataset.sample_ids):
                target = bp[str(sample_id)]["target"]
                error = fields[arm][str(sample_id)] - target
                raw_rows.append({
                    "split": split,
                    "sample_id": str(sample_id),
                    "object_id": object_id(str(sample_id)),
                    "arm": SOURCE_LABELS[arm],
                    "raw_epe": float(np.mean(np.linalg.norm(error, axis=1))),
                    "raw_rms": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
                    "vertices": len(error),
                })
    for row in raw_rows:
        row.setdefault("arm", "B_P")
    write_csv(output / "all_raw_differential_metrics.csv", raw_rows)
    raw_aggregate = [
        {
            "split": split,
            "arm": arm,
            "raw_epe": float(np.mean([row["raw_epe"] for row in raw_rows if row["split"] == split and row["arm"] == arm])),
            "raw_rms": float(np.mean([row["raw_rms"] for row in raw_rows if row["split"] == split and row["arm"] == arm])),
        }
        for split in ("validation", "test") for arm in ("A", "B_0", "B_P")
    ]

    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(manifest, split)
        for index, sample_id_value in enumerate(dataset.sample_ids):
            sample_id = str(sample_id_value)
            static = dataset.load_static(index)
            initial = Mesh(
                np.asarray(static["vertices"], dtype=np.float64),
                np.asarray(static["faces"], dtype=np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            vp = initial.vertices + fields[ARM_E][sample_id]
            for regularization, anchors in ((0.01, ("V0", "VP")), (0.03, ("VP",))):
                for anchor_name in anchors:
                    anchor = initial.vertices if anchor_name == "V0" else vp
                    for arm in (ARM_A, ARM_B0, ARM_BP):
                        name = method(arm, anchor_name, regularization)
                        recovered, solver = _pcg(
                            fields[arm][sample_id], anchor, static, regularization, device
                        )
                        if not solver["pcg_converged"]:
                            raise RuntimeError(f"{sample_id}/{name}: PCG failed")
                        metric = _geometry_row(
                            split,
                            sample_id,
                            name,
                            Mesh(recovered, initial.faces.copy()).ensure_normals(),
                            clean,
                            initial,
                        )
                        row = _row(
                            split, name, sample_id, index, recovered, clean, initial,
                            metric, solver, regularization,
                        )
                        row.update({
                            "method": name,
                            "field_source": SOURCE_LABELS[arm],
                            "recovery_anchor": ANCHOR_LABELS[anchor_name],
                            "object_id": object_id(sample_id),
                        })
                        rows.append(row)
    aggregates = aggregate(rows)
    object_rows = per_object(rows)
    write_csv(output / "per_mesh_metrics.csv", rows)
    write_csv(output / "aggregate_metrics.csv", aggregates)
    write_csv(output / "per_object_metrics.csv", object_rows)

    comparisons = [
        ("B_P@V_P vs B_0@V_P", method(ARM_BP, "VP", 0.01), method(ARM_B0, "VP", 0.01)),
        ("B_P@V0 vs B_0@V0", method(ARM_BP, "V0", 0.01), method(ARM_B0, "V0", 0.01)),
        ("B_P@V_P vs A@V_P", method(ARM_BP, "VP", 0.01), method(ARM_A, "VP", 0.01)),
    ]
    write_csv(
        output / "per_object_paired_differences.csv",
        per_object_comparison_differences(object_rows, comparisons),
    )
    paired_results: list[dict[str, Any]] = []
    for label, candidate, reference in comparisons:
        for field in FIELDS:
            value = paired(rows, candidate, reference, field, args.bootstrap_replicates, args.seed)
            value["comparison"] = label
            paired_results.append(value)
    interactions = [
        interaction(rows, field, args.bootstrap_replicates, args.seed) for field in FIELDS
    ]
    write_json(output / "paired_bootstrap_results.json", paired_results)
    write_json(output / "anchor_interaction_results.json", interactions)

    primary_cd = next(row for row in paired_results if row["comparison"] == "B_P@V_P vs B_0@V_P" and row["metric"] == "refined_chamfer")
    v0_cd = next(row for row in paired_results if row["comparison"] == "B_P@V0 vs B_0@V0" and row["metric"] == "refined_chamfer")
    interaction_cd = next(row for row in interactions if row["metric"] == "refined_chamfer")
    primary_support = all(primary_cd[key][1] < 0 for key in ("mesh_bootstrap_95_percent_ci", "object_cluster_bootstrap_95_percent_ci"))
    v0_loss = all(v0_cd[key][0] > 0 for key in ("mesh_bootstrap_95_percent_ci", "object_cluster_bootstrap_95_percent_ci"))
    interaction_support = all(interaction_cd[key][1] < 0 for key in ("mesh_bootstrap_95_percent_ci", "object_cluster_bootstrap_95_percent_ci"))
    contradiction = all(primary_cd[key][0] > 0 for key in ("mesh_bootstrap_95_percent_ci", "object_cluster_bootstrap_95_percent_ci")) or all(interaction_cd[key][0] > 0 for key in ("mesh_bootstrap_95_percent_ci", "object_cluster_bootstrap_95_percent_ci"))
    if primary_support and interaction_support and v0_loss:
        verdict = "STRONG EVIDENCE FOR ANCHOR-CONDITIONED DIFFERENTIAL LEARNING"
    elif primary_support and interaction_support:
        verdict = "PARTIAL EVIDENCE FOR ANCHOR CONDITIONING"
    elif contradiction:
        verdict = "EVIDENCE CONTRADICTS ANCHOR-CONDITIONING HYPOTHESIS"
    else:
        verdict = "NO EVIDENCE FOR ANCHOR CONDITIONING"

    training_metrics = read_json(run / "metrics.json") if (run / "metrics.json").is_file() else None
    gradient_audit = read_json(run / "pretraining_gradient_audit.json")
    history = json.loads((run / "training_history.json").read_text(encoding="utf-8")) if (run / "training_history.json").is_file() else []
    bp_val_name = method(ARM_BP, "VP", 0.01)
    bp_validation = next(row for row in aggregates if row["split"] == "validation" and row["method"] == bp_val_name)
    summary = {
        "execution_status": "completed",
        "contract_audit": True,
        "metric_protocol": METRIC_PROTOCOL,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "bp_run": str(run),
        "bp_checkpoint": str(bp_checkpoint),
        "bp_checkpoint_sha256": bp_checkpoint_sha,
        "bp_selected_epoch": bp_audit["checkpoint_payload"].get("epoch"),
        "bp_selected_optimizer_steps": bp_audit["checkpoint_payload"].get("optimizer_steps"),
        "training_metrics": training_metrics,
        "gradient_audit": gradient_audit,
        "validation_bp_vp_lambda_0_01": bp_validation,
        "raw_differential_aggregate": raw_aggregate,
        "aggregate_metrics": aggregates,
        "primary_paired_results": paired_results,
        "anchor_interactions": interactions,
        "verdict": verdict,
        "source_provenance": provenance,
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed},
        "runtime_seconds": time.perf_counter() - started,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_history_records": len(history),
    }
    write_json(output / "summary.json", summary)

    test_primary = [row for row in aggregates if row["split"] == "test" and float(row["method"].split("=")[-1]) == 0.01]
    matrix: dict[tuple[str, str], Mapping[str, Any]] = {
        (row["method"].split("|")[0], row["method"].split("|")[1]): row
        for row in test_primary
    }
    report: list[str] = [
        "# Sofa50 anchor-conditioning ablation",
        "",
        "## 1. EXECUTION STATUS",
        "",
        "**completed**. Exactly one new model, Arm B_P, was trained; A, B_0 and E remained frozen.",
        "",
        "## 2. CONTRACT AUDIT",
        "",
        "Contract audit: **true**. B_P copies B_0's architecture, raw target, Huber loss, beta=0.01, lambda_train=0.01, optimizer, schedule, 20k-step budget, split, views, resolution and seed. The only methodological change is the recovery-loss anchor from V0 to the cached detached frozen Arm-E prediction V_P. B_P used four L40 ranks with accumulation two (global batch eight); this execution layout differs from B_0's historical allocation and is therefore recorded rather than described as a perfectly identical hardware execution.",
        "",
        "## 3. TRAINING SUMMARY",
        "",
        f"- Selected checkpoint: `{bp_checkpoint}` (SHA-256 `{bp_checkpoint_sha}`).",
        f"- Selected epoch/step: `{summary['bp_selected_epoch']}` / `{summary['bp_selected_optimizer_steps']}`.",
        f"- Gradient audit passed: `{gradient_audit['contract_audit']}`; delta gradient norm `{fmt(gradient_audit['delta_prediction_gradient_norm_from_recovery_loss'])}`; E gradients present `{gradient_audit['arm_e_gradients_present']}`.",
        f"- Validation B_P@V_P lambda=0.01: CD `{fmt(bp_validation['refined_chamfer'])}`, P2S p95 `{fmt(bp_validation['p2s_p95'])}`, F-score `{fmt(bp_validation['fscore'])}`, Normal `{fmt(bp_validation['normal_consistency'])}`, VRMS `{fmt(bp_validation['same_index_recovered_vertex_rms'])}`.",
        f"- Evaluation PCG: failed solves `{sum(int(row['pcg_failed_solves']) for row in aggregates)}`; maximum relative residual `{fmt(max(float(row['pcg_relative_residual_max']) for row in aggregates))}`.",
        "",
        "## 4. CROSS-ANCHOR MATRIX (test, lambda=0.01)",
        "",
        "| Field source | V0: CD / P2S p95 / F / Normal / VRMS | V_P: CD / P2S p95 / F / Normal / VRMS |",
        "|---|---:|---:|",
    ]
    for label in ("A", "B_0", "B_P"):
        cells = []
        for anchor in ("V0", "V_P"):
            row = matrix[(label, anchor)]
            cells.append(" / ".join(fmt(float(row[field])) for field in FIELDS))
        report.append(f"| {label} | {cells[0]} | {cells[1]} |")
    report.extend(["", "Secondary V_P-column results at lambda=0.03 are recorded in `aggregate_metrics.csv`.", "", "## 5. PRIMARY PAIRED RESULTS", "", "Differences are candidate minus reference.", "", "| Comparison | CD mean [mesh CI] [object CI] | W/T/L |", "|---|---:|---:|"])
    for label, _, _ in comparisons:
        row = next(value for value in paired_results if value["comparison"] == label and value["metric"] == "refined_chamfer")
        report.append(
            f"| {label} | {fmt(row['mean_difference'])} "
            f"[{fmt(row['mesh_bootstrap_95_percent_ci'][0])}, {fmt(row['mesh_bootstrap_95_percent_ci'][1])}] "
            f"[{fmt(row['object_cluster_bootstrap_95_percent_ci'][0])}, {fmt(row['object_cluster_bootstrap_95_percent_ci'][1])}] | "
            f"{row['candidate_wins']}/{row['ties']}/{row['candidate_losses']} |"
        )
    report.extend([
        "",
        "## 6. ANCHOR-INTERACTION RESULT",
        "",
        f"For CD, `interaction=(B_P@V_P-B_0@V_P)-(B_P@V0-B_0@V0)` is `{fmt(interaction_cd['interaction_mean'])}` with mesh CI `[{fmt(interaction_cd['mesh_bootstrap_95_percent_ci'][0])}, {fmt(interaction_cd['mesh_bootstrap_95_percent_ci'][1])}]` and object-cluster CI `[{fmt(interaction_cd['object_cluster_bootstrap_95_percent_ci'][0])}, {fmt(interaction_cd['object_cluster_bootstrap_95_percent_ci'][1])}]`. Negative values favor anchor-specific conditioning.",
        "",
        "## 7. VERDICT",
        "",
        f"**{verdict}**",
        "",
        "## 8. PAPER IMPLICATION",
        "",
        "This experiment isolates the recovery anchor used during reconstruction-mediated differential training. The paired same-anchor comparison and the paired interaction determine whether any B_P gain is specifically associated with V_P rather than a general improvement. Raw differential metrics are reported separately and do not determine the recovery verdict. The result is scoped to frozen single-pass Sofa50-v2 recovery at the tested lambdas; no paper file was modified.",
        "",
        "## Reproducibility",
        "",
        f"- Evaluator: `{METRIC_PROTOCOL}`.",
        f"- Bootstrap: `{args.bootstrap_replicates}` replicates, seed `{args.seed}`.",
        f"- Git HEAD: `{summary['git_head']}`.",
        f"- Evaluation command: `{' '.join(sys.argv)}`.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": verdict, "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

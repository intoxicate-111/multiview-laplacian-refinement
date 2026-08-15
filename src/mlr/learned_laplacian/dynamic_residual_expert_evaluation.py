from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .canonical_experiment import _exact_query_sample, _load_device_item
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .synthetic_current_h2_ablation import (
    RAW_METRIC_FIELDS,
    _raw_metrics,
    _recover_raw_one,
    _run_config,
    _sha256,
    _target_formula_audit,
    _validate_sample_contract,
)
from .target_scaling import normalize_laplacian_by_edge_scale
from .trainer import _seed_everything, load_checkpoint


ARMS = ("joint_base_branch", "joint_dynamic_final")
SPLITS = ("validation", "test")
GROUPS = ("bottom_90_percent", "top_10_percent", "top_1_percent")
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_dynamic_expert_evaluation(
    manifest_path: str | Path,
    expert_run: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    shard_index: int | None = None,
    shard_count: int = 1,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    run_dir = Path(expert_run).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Dynamic expert evaluation requires an available CUDA device.")
    shard_index = _validated_shard_index(shard_index, shard_count)
    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", *SPLITS)
    }
    validate_disjoint_splits(*datasets.values())
    counts = {split: len(dataset) for split, dataset in datasets.items()}
    if counts != {"train": 200, "validation": 25, "test": 25}:
        raise ValueError(f"Unexpected split counts: {counts}.")

    spec = _load_spec(run_dir, resolved_device)
    audit = _contract_audit(manifest, datasets, spec)
    if not audit["passed"]:
        _write_json(
            output / "shards" / f"contract_audit_shard_{shard_index}.json", audit
        )
        raise RuntimeError("Dynamic expert contract audit failed.")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    gate_sample_rows: list[dict[str, Any]] = []
    formula_checks: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in SPLITS:
        dataset = datasets[split]
        for index in range(len(dataset)):
            if index % shard_count != shard_index:
                continue
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            metadata = dict(static.get("metadata", {}))
            _validate_sample_contract(sample_id, metadata)
            formula_checks.append({"split": split, **_target_formula_audit(static)})
            values = _infer_one(dataset, index, spec, resolved_device, static["faces"])
            valid = values["valid"].numpy().astype(bool)
            target = values["target_raw"].numpy()
            gt_magnitude = np.linalg.norm(target, axis=1)
            base_residual = np.linalg.norm(values["base_raw"].numpy() - target, axis=1)
            final_residual = np.linalg.norm(values["final_raw"].numpy() - target, axis=1)
            expert_norm = np.linalg.norm(values["expert_residual"].numpy(), axis=1)
            correction_norm = (
                values["gate_effective"].numpy() * expert_norm
            )
            for arm, key in zip(ARMS, ("base_raw", "final_raw"), strict=True):
                metrics = _raw_metrics(
                    values[key],
                    values["target_raw"],
                    values["recovery_weight"],
                    values["valid"],
                )
                prediction_rows.append(
                    {
                        "split": split,
                        "arm": arm,
                        "sample_id": sample_id,
                        "object_id": metadata.get("object_id"),
                        "variant_index": metadata.get("variant_index"),
                        "vertex_count": len(target),
                        "valid_vertex_count": int(valid.sum()),
                        **metrics,
                    }
                )
                prefix = f"{split}__{arm}"
                arrays[f"{prefix}__prediction"].append(values[key].numpy()[valid])
                arrays[f"{prefix}__target"].append(target[valid])
                arrays[f"{prefix}__weight"].append(
                    values["recovery_weight"].numpy()[valid]
                )
            arrays[f"{split}__gate_logit"].append(values["gate_logit"].numpy()[valid])
            arrays[f"{split}__gate_signed"].append(values["gate_signed"].numpy()[valid])
            arrays[f"{split}__gate_effective"].append(
                values["gate_effective"].numpy()[valid]
            )
            arrays[f"{split}__gt_magnitude"].append(gt_magnitude[valid])
            arrays[f"{split}__base_residual"].append(base_residual[valid])
            arrays[f"{split}__final_residual"].append(final_residual[valid])
            arrays[f"{split}__expert_norm"].append(expert_norm[valid])
            arrays[f"{split}__correction_norm"].append(correction_norm[valid])
            arrays[f"{split}__object_id"].append(
                np.full(valid.sum(), str(metadata.get("object_id")), dtype="U64")
            )
            active = values["gate_effective"].numpy() > 0
            gate_sample_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "object_id": metadata.get("object_id"),
                    "variant_index": metadata.get("variant_index"),
                    "activation_fraction": float(active[valid].mean()),
                    "mean_effective_gate": float(
                        values["gate_effective"].numpy()[valid].mean()
                    ),
                    "mean_expert_residual_norm": float(expert_norm[valid].mean()),
                    "mean_effective_correction_norm": float(
                        correction_norm[valid].mean()
                    ),
                    "mean_base_raw_error": float(base_residual[valid].mean()),
                    "mean_final_raw_error": float(final_residual[valid].mean()),
                }
            )
            _write_per_vertex(
                output,
                split,
                sample_id,
                values,
                gt_magnitude,
                base_residual,
                final_residual,
            )
            if split == "test":
                for arm, raw_key, norm_key in (
                    (ARMS[0], "base_raw", "base_normalized"),
                    (ARMS[1], "final_raw", "final_normalized"),
                ):
                    recovery, _ = _recover_raw_one(
                        static,
                        values[raw_key],
                        values[norm_key],
                        values["confidence"],
                        output / "reconstruction" / arm / sample_id,
                        spec["config"],
                    )
                    recovery_rows.append(
                        {
                            "arm": arm,
                            "sample_id": sample_id,
                            "object_id": metadata.get("object_id"),
                            "variant_index": metadata.get("variant_index"),
                            **recovery,
                        }
                    )
                    print(
                        f"{arm} {sample_id} chamfer="
                        f"{recovery['reconstruction_chamfer']:.9g} "
                        f"improved={recovery['improved_over_initial']}",
                        flush=True,
                    )
            del values
            torch.cuda.empty_cache()
    return _write_shard(
        manifest,
        output,
        shard_index,
        shard_count,
        audit,
        spec,
        prediction_rows,
        recovery_rows,
        gate_sample_rows,
        formula_checks,
        arrays,
    )


def merge_dynamic_expert_shards(
    manifest_path: str | Path, output_dir: str | Path, *, shard_count: int
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    shard_dir = output / "shards"
    payloads = [
        _read_json(shard_dir / f"shard_{index}.json") for index in range(shard_count)
    ]
    for index, payload in enumerate(payloads):
        if int(payload["shard_index"]) != index:
            raise RuntimeError(f"Invalid shard index {index}.")
        if int(payload["shard_count"]) != shard_count:
            raise RuntimeError(f"Invalid shard count {index}.")
        if payload["manifest_sha256"] != _sha256(manifest):
            raise RuntimeError(f"Manifest mismatch in shard {index}.")
        if payload["contract_audit"] != payloads[0]["contract_audit"]:
            raise RuntimeError("Shard contract audits differ.")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(shard_count):
        with np.load(shard_dir / f"arrays_shard_{index}.npz") as archive:
            for name in archive.files:
                arrays[name].append(archive[name])
    merged = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    return _finalize(
        manifest,
        output,
        payloads[0]["contract_audit"],
        payloads[0]["run_metadata"],
        _concat(payloads, "prediction_rows"),
        _concat(payloads, "recovery_rows"),
        _concat(payloads, "gate_sample_rows"),
        _concat(payloads, "formula_checks"),
        merged,
        shard_count,
    )


def _load_spec(run_dir: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    if not checkpoint.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete expert run: {run_dir}.")
    config = _run_config(run_dir)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "config": config,
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "metrics": _read_json(metrics_path),
        "run_config": _read_json(run_dir / "run_config.json"),
    }


def _infer_one(
    dataset: PreparedMeshDataset,
    index: int,
    spec: Mapping[str, Any],
    device: torch.device,
    current_faces: torch.Tensor | np.ndarray,
) -> dict[str, torch.Tensor]:
    config = spec["config"]
    prepared = _load_device_item(dataset, index, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        output = spec["model"](conditioned)
    required = (
        output.base_laplacian_prediction,
        output.dynamic_expert_residual_prediction,
        output.dynamic_gate_logit,
        output.dynamic_gate_signed,
        output.dynamic_gate_effective,
        output.confidence_prediction,
    )
    if any(value is None for value in required):
        raise RuntimeError("Checkpoint did not produce complete dynamic expert output.")
    h = prepared.sample["local_edge_length"].float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    epsilon = float(config["target_scaling"]["epsilon"])
    base_raw = output.base_laplacian_prediction.float().detach().cpu()
    final_raw = output.predicted_laplacian.float().detach().cpu()
    base_normalized = normalize_laplacian_by_edge_scale(
        base_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    final_normalized = normalize_laplacian_by_edge_scale(
        final_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    confidence = output.confidence_prediction.float().detach().cpu()
    visibility = prepared.sample["visibility"].detach().cpu()
    canonical = canonical_current_graph_recovery_inputs(
        prepared.sample["vertices"].detach().cpu(),
        current_faces,
        final_normalized,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    return {
        "base_raw": base_raw,
        "final_raw": final_raw,
        "base_normalized": base_normalized,
        "final_normalized": final_normalized,
        "expert_residual": output.dynamic_expert_residual_prediction.float().detach().cpu(),
        "gate_logit": output.dynamic_gate_logit.float().detach().cpu(),
        "gate_signed": output.dynamic_gate_signed.float().detach().cpu(),
        "gate_effective": output.dynamic_gate_effective.float().detach().cpu(),
        "target_raw": prepared.raw_target.float().detach().cpu(),
        "confidence": confidence,
        "visibility_count": visibility.to(torch.int64).sum(dim=0),
        "valid": valid,
        "recovery_weight": canonical.weight.detach().cpu(),
    }


def _contract_audit(
    manifest: Path,
    datasets: Mapping[str, PreparedMeshDataset],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    config = spec["config"]
    run_config = spec["run_config"]
    initialization_value = run_config.get("initialization_checkpoint")
    initialization = (
        None if initialization_value is None else Path(str(initialization_value))
    )
    initialization_hash = run_config.get("initialization_checkpoint_sha256")
    _seed_everything(int(config.get("seed", 7)))
    expert_initial = _build_model(config, None, False).cpu()
    baseline_config = json.loads(json.dumps(config))
    baseline_config["model"].pop("dynamic_residual_expert", None)
    _seed_everything(int(config.get("seed", 7)))
    baseline_initial = _build_model(baseline_config, None, False).cpu()
    baseline_initial_state = baseline_initial.state_dict()
    expert_initial_state = expert_initial.state_dict()
    canonical_keys = sorted(baseline_initial_state)
    seeded_canonical_equal = all(
        torch.equal(baseline_initial_state[name], expert_initial_state[name])
        for name in canonical_keys
    )
    dynamic = config.get("model", {}).get("dynamic_residual_expert", {})
    metrics = spec["metrics"]
    split_ids = {split: list(dataset.sample_ids) for split, dataset in datasets.items()}
    run_manifest = _read_json(spec["run_dir"] / "dataset_manifest.json")
    recorded = {split: [] for split in ("train", "validation", "test")}
    for item in run_manifest.get("samples", []):
        if item.get("split") in recorded:
            recorded[item["split"]].append(str(item.get("sample_id")))
    checks = {
        "manifest_exists": manifest.is_file(),
        "split_counts_200_25_25": {k: len(v) for k, v in split_ids.items()}
        == {"train": 200, "validation": 25, "test": 25},
        "run_split_ids_match": recorded == split_ids,
        "no_initialization_checkpoint": initialization is None
        and initialization_hash is None,
        "seeded_canonical_initialization_matches_plain_base": seeded_canonical_equal,
        "raw_mse": config.get("training", {}).get("loss") == "mse",
        "joint_training_all_parameters": config.get("training", {}).get(
            "trainable_parameter_scope"
        )
        == "all",
        "from_scratch_declared": config.get("experiment_metadata", {}).get(
            "initialization"
        )
        == "random_seed_7_no_checkpoint",
        "no_jitter": not config.get("local_query_jitter", {}).get("enabled", False),
        "twenty_thousand_steps": int(metrics.get("optimizer_steps", -1)) == 20_000,
        "four_gpu_world": int(metrics.get("distributed_world_size", -1)) == 4,
        "global_batch_eight": int(metrics.get("global_batch_meshes", -1)) == 8,
        "learned_gate_exact_formula": dynamic.get("gate_formula")
        == "g=max(0,tanh(a))",
        "no_target_routing_declared": config.get("experiment_metadata", {}).get(
            "routing_supervision"
        )
        == "none",
        "oracle_branch_disabled": not config.get("model", {}).get(
            "oracle_residual_expert", {}
        ).get("enabled", False),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": _sha256(manifest),
        "base_initialization_checkpoint": None,
        "base_initialization_sha256": initialization_hash,
        "canonical_base_tensor_count": len(canonical_keys),
        "training_gpu_contract": {
            "gpu_model": "NVIDIA L40",
            "world_size": metrics.get("distributed_world_size"),
            "per_rank_batch_meshes": 1,
            "gradient_accumulation_meshes": config["multi_object_training"][
                "gradient_accumulation_meshes"
            ],
            "global_batch_meshes": metrics.get("global_batch_meshes"),
            "baseline_global_batch_meshes": 6,
            "resource_override": "User changed the run from 2 to 4 L40 after launch.",
        },
    }


def _write_per_vertex(
    output: Path,
    split: str,
    sample_id: str,
    values: Mapping[str, torch.Tensor],
    gt_magnitude: np.ndarray,
    base_residual: np.ndarray,
    final_residual: np.ndarray,
) -> None:
    path = output / "per_vertex" / split / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        base_raw_prediction=values["base_raw"].numpy(),
        expert_residual_raw=values["expert_residual"].numpy(),
        final_raw_prediction=values["final_raw"].numpy(),
        gate_logit_a=values["gate_logit"].numpy(),
        gate_signed_s=values["gate_signed"].numpy(),
        gate_effective_g=values["gate_effective"].numpy(),
        gt_raw_laplacian=values["target_raw"].numpy(),
        base_raw_prediction_residual_magnitude=base_residual,
        final_raw_prediction_residual_magnitude=final_residual,
        gt_raw_laplacian_magnitude=gt_magnitude,
        visibility_count=values["visibility_count"].numpy(),
        confidence=values["confidence"].numpy(),
        recovery_weight=values["recovery_weight"].numpy(),
        valid_scale_mask=values["valid"].numpy(),
    )


def _write_shard(
    manifest: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
    audit: Mapping[str, Any],
    spec: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    gate_sample_rows: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_dir / f"arrays_shard_{shard_index}.npz",
        **{name: np.concatenate(chunks) for name, chunks in arrays.items()},
    )
    payload = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "manifest_sha256": _sha256(manifest),
        "contract_audit": dict(audit),
        "run_metadata": {
            "run_dir": str(spec["run_dir"]),
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "optimizer_steps": spec["metrics"].get("optimizer_steps"),
            "runtime_seconds": spec["metrics"].get("runtime_seconds"),
            "best_validation_loss": spec["metrics"].get("best_selection_loss"),
        },
        "prediction_rows": list(prediction_rows),
        "recovery_rows": list(recovery_rows),
        "gate_sample_rows": list(gate_sample_rows),
        "formula_checks": list(formula_checks),
    }
    _write_json(shard_dir / f"shard_{shard_index}.json", payload)
    return payload


def _finalize(
    manifest: Path,
    output: Path,
    preflight: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    gate_sample_rows: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    shard_count: int,
) -> dict[str, Any]:
    max_formula = max(
        float(row["current_graph_proxy_raw_target_max_abs_error"])
        for row in formula_checks
    )
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "gate_sample_rows": len(gate_sample_rows),
        "formula_checks": len(formula_checks),
        "per_vertex_npz": len(list((output / "per_vertex").glob("*/*.npz"))),
    }
    expected = {
        "prediction_rows": 100,
        "recovery_rows": 50,
        "gate_sample_rows": 50,
        "formula_checks": 50,
        "per_vertex_npz": 50,
    }
    audit = {
        **dict(preflight),
        "counts": counts,
        "counts_match": counts == expected,
        "maximum_target_formula_error": max_formula,
        "passed": bool(preflight["passed"] and counts == expected and max_formula <= 1e-7),
    }
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Final dynamic expert audit failed.")

    prediction_aggregate, group_rows = _prediction_aggregates(arrays)
    gate_summary, gate_group_rows, relationship_rows, object_rows = _gate_analysis(
        arrays
    )
    recovery_aggregate = _recovery_aggregate(recovery_rows)
    paired_rows, paired_summary = _paired(prediction_rows, recovery_rows)
    questions = _answer_questions(
        prediction_aggregate,
        group_rows,
        gate_summary,
        gate_group_rows,
        relationship_rows,
        recovery_aggregate,
        paired_summary,
    )
    summary = {
        "experiment": "Sofa50 frozen raw-MSE base + learned dynamic residual expert",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "evaluation_shards": shard_count,
        "run": dict(run_metadata),
        "contract_audit": audit,
        "prediction_aggregate": prediction_aggregate,
        "gt_raw_laplacian_magnitude_groups": group_rows,
        "gate_summary": gate_summary,
        "gate_by_gt_magnitude_group": gate_group_rows,
        "gate_relationships": relationship_rows,
        "gate_by_object": object_rows,
        "recovery_aggregate": recovery_aggregate,
        "paired_summary": paired_summary,
        "questions": questions,
    }
    _write_json(output / "dynamic_expert_summary.json", summary)
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "prediction_aggregate.csv", prediction_aggregate)
    _write_csv(output / "gt_raw_laplacian_magnitude_groups.csv", group_rows)
    _write_csv(output / "gate_per_sample.csv", gate_sample_rows)
    _write_csv(output / "gate_by_gt_magnitude_group.csv", gate_group_rows)
    _write_csv(output / "gate_relationships.csv", relationship_rows)
    _write_csv(output / "gate_by_object.csv", object_rows)
    _write_csv(output / "recovery_per_sample.csv", recovery_rows)
    _write_csv(output / "recovery_aggregate.csv", recovery_aggregate)
    _write_csv(output / "paired_per_sample.csv", paired_rows)
    _write_json(output / "paired_summary.json", paired_summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _prediction_aggregates(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for split in SPLITS:
        magnitude = arrays[f"{split}__gt_magnitude"]
        masks = _group_masks(magnitude)
        for arm in ARMS:
            prefix = f"{split}__{arm}"
            prediction = torch.from_numpy(arrays[f"{prefix}__prediction"]).double()
            target = torch.from_numpy(arrays[f"{prefix}__target"]).double()
            weight = torch.from_numpy(arrays[f"{prefix}__weight"]).double()
            aggregate.append(
                {
                    "split": split,
                    "arm": arm,
                    "vertex_count": len(prediction),
                    **_raw_metrics(
                        prediction, target, weight, torch.ones(len(prediction), dtype=torch.bool)
                    ),
                }
            )
            error = np.linalg.norm(prediction.numpy() - target.numpy(), axis=1)
            for group, mask in masks.items():
                groups.append(
                    {
                        "split": split,
                        "arm": arm,
                        "group": group,
                        "vertex_count": int(mask.sum()),
                        "mean_raw_error_epe": float(error[mask].mean()),
                    }
                )
    return aggregate, groups


def _gate_analysis(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for split in SPLITS:
        gate = arrays[f"{split}__gate_effective"]
        logit = arrays[f"{split}__gate_logit"]
        signed = arrays[f"{split}__gate_signed"]
        active = gate > 0
        expert_norm = arrays[f"{split}__expert_norm"]
        correction_norm = arrays[f"{split}__correction_norm"]
        magnitude = arrays[f"{split}__gt_magnitude"]
        base_error = arrays[f"{split}__base_residual"]
        final_error = arrays[f"{split}__final_residual"]
        positive = gate[active]
        summaries.append(
            {
                "split": split,
                "vertex_count": len(gate),
                "activation_fraction": float(active.mean()),
                "positive_gate_mean": float(positive.mean()) if len(positive) else 0.0,
                "positive_gate_median": float(np.median(positive)) if len(positive) else 0.0,
                "mean_gate_logit": float(logit.mean()),
                "mean_gate_signed": float(signed.mean()),
                "mean_expert_residual_norm": float(expert_norm.mean()),
                "mean_effective_correction_norm": float(correction_norm.mean()),
                "gate_collapse_off": bool(active.mean() < 0.01),
                "gate_collapse_on": bool(active.mean() > 0.99),
                "residual_near_zero": bool(expert_norm.mean() < 1e-6),
                "active_vertex_base_epe": float(base_error[active].mean())
                if active.any()
                else None,
                "active_vertex_final_epe": float(final_error[active].mean())
                if active.any()
                else None,
                "inactive_vertex_base_epe": float(base_error[~active].mean())
                if (~active).any()
                else None,
                "inactive_vertex_final_epe": float(final_error[~active].mean())
                if (~active).any()
                else None,
            }
        )
        for group, mask in _group_masks(magnitude).items():
            selected_positive = gate[mask & active]
            groups.append(
                {
                    "split": split,
                    "group": group,
                    "vertex_count": int(mask.sum()),
                    "activation_fraction": float(active[mask].mean()),
                    "mean_effective_gate": float(gate[mask].mean()),
                    "positive_gate_mean": float(selected_positive.mean())
                    if len(selected_positive)
                    else 0.0,
                }
            )
        for gate_name, gate_value in (("logit_a", logit), ("effective_g", gate)):
            for value_name, value in (
                ("gt_raw_laplacian_magnitude", magnitude),
                ("base_raw_residual", base_error),
                ("final_raw_residual", final_error),
            ):
                relationships.append(
                    {
                        "split": split,
                        "gate_value": gate_name,
                        "relationship_to": value_name,
                        "pearson": _pearson(gate_value, value),
                        "spearman": _spearman(gate_value, value),
                    }
                )
        object_ids = arrays[f"{split}__object_id"]
        for object_id in sorted(set(object_ids.tolist())):
            mask = object_ids == object_id
            objects.append(
                {
                    "split": split,
                    "object_id": object_id,
                    "vertex_count": int(mask.sum()),
                    "activation_fraction": float(active[mask].mean()),
                    "mean_effective_gate": float(gate[mask].mean()),
                }
            )
        summaries[-1]["object_identity_eta_squared_effective_gate"] = _eta_squared(
            gate, object_ids
        )
    return summaries, groups, relationships, objects


def _recovery_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        result.append(
            {
                "arm": arm,
                "sample_count": len(selected),
                **{
                    field: float(np.mean([float(row[field]) for row in selected]))
                    for field in GEOMETRY_FIELDS
                },
                "introduced_flipped_faces": int(
                    sum(int(row["introduced_flipped_faces"]) for row in selected)
                ),
                "new_degenerate_faces": int(
                    sum(int(row["new_degenerate_faces"]) for row in selected)
                ),
                "improved_over_initial": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
            }
        )
    return result


def _paired(
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = {
        (row["split"], row["sample_id"], row["arm"]): row for row in prediction_rows
    }
    recoveries = {(row["sample_id"], row["arm"]): row for row in recovery_rows}
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        ids = sorted(
            key[1] for key in predictions if key[0] == split and key[2] == ARMS[0]
        )
        for sample_id in ids:
            base = predictions[(split, sample_id, ARMS[0])]
            final = predictions[(split, sample_id, ARMS[1])]
            row: dict[str, Any] = {
                "split": split,
                "sample_id": sample_id,
                "object_id": base.get("object_id"),
                "variant_index": base.get("variant_index"),
            }
            for field in RAW_METRIC_FIELDS:
                row[f"base_{field}"] = base[field]
                row[f"expert_{field}"] = final[field]
                row[f"expert_minus_base_{field}"] = float(final[field]) - float(base[field])
            if split == "test":
                br = recoveries[(sample_id, ARMS[0])]
                er = recoveries[(sample_id, ARMS[1])]
                for field in (
                    "reconstruction_chamfer",
                    "reconstruction_point_to_surface",
                    "reconstruction_normal_consistency",
                    "introduced_flipped_faces",
                ):
                    row[f"base_{field}"] = br[field]
                    row[f"expert_{field}"] = er[field]
                    row[f"expert_minus_base_{field}"] = float(er[field]) - float(br[field])
                row["base_improved"] = bool(br["improved_over_initial"])
                row["expert_improved"] = bool(er["improved_over_initial"])
                row["mse_failure_recovered"] = not row["base_improved"] and row["expert_improved"]
                row["mse_success_lost"] = row["base_improved"] and not row["expert_improved"]
            rows.append(row)
    test = [row for row in rows if row["split"] == "test"]
    summary = {
        "delta_definition": "joint_dynamic_final minus joint_base_branch",
        "test_expert_lower_raw_epe": sum(row["expert_minus_base_raw_epe"] < 0 for row in test),
        "test_expert_lower_chamfer": sum(
            row["expert_minus_base_reconstruction_chamfer"] < 0 for row in test
        ),
        "test_expert_lower_p2s": sum(
            row["expert_minus_base_reconstruction_point_to_surface"] < 0
            for row in test
        ),
        "test_expert_higher_normal_consistency": sum(
            row["expert_minus_base_reconstruction_normal_consistency"] > 0
            for row in test
        ),
        "mse_failures_recovered": sum(row["mse_failure_recovered"] for row in test),
        "mse_successes_lost": sum(row["mse_success_lost"] for row in test),
    }
    return rows, summary


def _answer_questions(
    predictions: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    gate_groups: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    def row(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
        matches = [x for x in rows if all(x.get(k) == v for k, v in criteria.items())]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}.")
        return matches[0]

    base_pred = row(predictions, split="test", arm=ARMS[0])
    expert_pred = row(predictions, split="test", arm=ARMS[1])
    base_rec = row(recovery, arm=ARMS[0])
    expert_rec = row(recovery, arm=ARMS[1])
    test_gate = row(gates, split="test")
    group_delta = {
        group: float(
            row(groups, split="test", arm=ARMS[1], group=group)["mean_raw_error_epe"]
        )
        - float(row(groups, split="test", arm=ARMS[0], group=group)["mean_raw_error_epe"])
        for group in GROUPS
    }
    return {
        "1_gate_learns_when_expert_is_useful": {
            "activation_fraction": test_gate["activation_fraction"],
            "active_vertex_epe_change": (
                None
                if test_gate["active_vertex_final_epe"] is None
                else test_gate["active_vertex_final_epe"]
                - test_gate["active_vertex_base_epe"]
            ),
        },
        "2_where_gate_turns_on": [
            dict(x) for x in gate_groups if x["split"] == "test"
        ],
        "3_geometry_or_object_identity": {
            "object_identity_eta_squared": test_gate[
                "object_identity_eta_squared_effective_gate"
            ],
            "local_relationships": [
                dict(x)
                for x in relationships
                if x["split"] == "test" and x["gate_value"] == "effective_g"
            ],
        },
        "4_tail_error_changes": {
            "top_10_percent_epe_delta": group_delta["top_10_percent"],
            "top_1_percent_epe_delta": group_delta["top_1_percent"],
        },
        "5_bottom90_degradation": {
            "bottom_90_percent_epe_delta": group_delta["bottom_90_percent"],
            "degraded": group_delta["bottom_90_percent"] > 0,
        },
        "6_global_prediction_improvement": {
            field: float(expert_pred[field]) - float(base_pred[field])
            for field in (
                "raw_epe",
                "raw_residual_rms",
                "raw_residual_maximum",
                "raw_global_cosine",
                "recovery_weighted_raw_residual_rms",
            )
        },
        "7_downstream_recovery": {
            "chamfer_delta": expert_rec["reconstruction_chamfer"]
            - base_rec["reconstruction_chamfer"],
            "p2s_delta": expert_rec["reconstruction_point_to_surface"]
            - base_rec["reconstruction_point_to_surface"],
            "normal_consistency_delta": expert_rec["reconstruction_normal_consistency"]
            - base_rec["reconstruction_normal_consistency"],
            "introduced_flips_delta": expert_rec["introduced_flipped_faces"]
            - base_rec["introduced_flipped_faces"],
            "improved_count_delta": expert_rec["improved_over_initial"]
            - base_rec["improved_over_initial"],
            **dict(paired),
        },
        "8_failure_modes": {
            "gate_collapse_off": test_gate["gate_collapse_off"],
            "gate_collapse_on": test_gate["gate_collapse_on"],
            "residual_near_zero": test_gate["residual_near_zero"],
            "validation_test_activation_fraction_delta": test_gate[
                "activation_fraction"
            ]
            - row(gates, split="validation")["activation_fraction"],
        },
        "9_worth_continuing": bool(
            group_delta["top_1_percent"] < 0
            and expert_rec["reconstruction_chamfer"]
            < base_rec["reconstruction_chamfer"]
            and expert_rec["improved_over_initial"] >= base_rec["improved_over_initial"]
        ),
    }


def _group_masks(magnitude: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-magnitude, kind="stable")
    top10_count = max(1, int(math.ceil(0.10 * len(order))))
    top1_count = max(1, int(math.ceil(0.01 * len(order))))
    top10 = np.zeros(len(order), dtype=bool)
    top1 = np.zeros(len(order), dtype=bool)
    top10[order[:top10_count]] = True
    top1[order[:top1_count]] = True
    return {
        "bottom_90_percent": ~top10,
        "top_10_percent": top10,
        "top_1_percent": top1,
    }


def _average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    ranks = np.empty(len(value), dtype=np.float64)
    sorted_value = value[order]
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and sorted_value[stop] == sorted_value[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _eta_squared(value: np.ndarray, group: np.ndarray) -> float:
    total = float(np.square(value - value.mean()).sum())
    if total == 0:
        return 0.0
    between = 0.0
    for name in set(group.tolist()):
        selected = value[group == name]
        between += len(selected) * float((selected.mean() - value.mean()) ** 2)
    return between / total


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 Learned Dynamic Residual Expert",
        "",
        f"Contract audit passed: `{summary['contract_audit']['passed']}`.",
        "",
        "## Raw prediction",
        "",
        "| Split | Arm | EPE | RMS | Max | Cosine | Recovery-weighted RMS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in summary["prediction_aggregate"]:
        lines.append(
            f"| {x['split']} | {x['arm']} | {_f(x['raw_epe'])} | "
            f"{_f(x['raw_residual_rms'])} | {_f(x['raw_residual_maximum'])} | "
            f"{_f(x['raw_global_cosine'])} | {_f(x['recovery_weighted_raw_residual_rms'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| Split | Active | Positive mean | Positive median | Expert norm | Correction norm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for x in summary["gate_summary"]:
        lines.append(
            f"| {x['split']} | {_f(x['activation_fraction'])} | "
            f"{_f(x['positive_gate_mean'])} | {_f(x['positive_gate_median'])} | "
            f"{_f(x['mean_expert_residual_norm'])} | "
            f"{_f(x['mean_effective_correction_norm'])} |"
        )
    lines.extend(
        [
            "",
            "## Test downstream recovery",
            "",
            "| Arm | Chamfer | P2S | Normal | Introduced flips | Improved / 25 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for x in summary["recovery_aggregate"]:
        lines.append(
            f"| {x['arm']} | {_f(x['reconstruction_chamfer'])} | "
            f"{_f(x['reconstruction_point_to_surface'])} | "
            f"{_f(x['reconstruction_normal_consistency'])} | "
            f"{x['introduced_flipped_faces']} | {x['improved_over_initial']} |"
        )
    lines.extend(
        [
            "",
            "## Decision questions",
            "",
            "```json",
            json.dumps(summary["questions"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _validated_shard_index(index: int | None, count: int) -> int:
    if count < 1:
        raise ValueError("shard_count must be positive.")
    if count == 1:
        if index not in (None, 0):
            raise ValueError("Single-shard evaluation requires shard index 0 or None.")
        return 0
    if index is None or not 0 <= index < count:
        raise ValueError("Multi-shard evaluation requires 0 <= shard_index < shard_count.")
    return index


def _concat(payloads: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    return [dict(row) for payload in payloads for row in payload[key]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _f(value: Any) -> str:
    return f"{float(value):.9g}"

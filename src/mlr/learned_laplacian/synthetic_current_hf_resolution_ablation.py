from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .diagnostics import _amp_settings
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .synthetic_current_h2_ablation import (
    _infer_one,
    _raw_metrics,
    _recover_raw_one,
    _run_config,
    _sha256,
    _target_formula_audit,
    _validate_sample_contract,
)
from .synthetic_current_loss_ablation import (
    _concat,
    _f,
    _read_json,
    _validated_shard_index,
    _write_csv,
    _write_json,
)
from .trainer import _seed_everything, load_checkpoint


ARMS = ("HF_960", "HF_1920")
SPLITS = ("validation", "test")
RAW_FIELDS = (
    "raw_epe",
    "raw_residual_rms",
    "raw_residual_maximum",
    "raw_global_cosine",
    "recovery_weighted_raw_residual_rms",
)
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_hf_resolution_ablation(
    manifest_960_path: str | Path,
    manifest_1920_path: str | Path,
    run_960: str | Path,
    run_1920: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    shard_index: int | None = None,
    shard_count: int = 1,
) -> dict[str, Any]:
    manifests = {
        ARMS[0]: Path(manifest_960_path).resolve(),
        ARMS[1]: Path(manifest_1920_path).resolve(),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("HF resolution evaluation requires CUDA")
    shard_index = _validated_shard_index(shard_index, shard_count)

    datasets = {
        arm: {
            split: PreparedMeshDataset.from_manifest(manifest, split)
            for split in ("train", *SPLITS)
        }
        for arm, manifest in manifests.items()
    }
    for arm in ARMS:
        validate_disjoint_splits(*datasets[arm].values())
        counts = {split: len(dataset) for split, dataset in datasets[arm].items()}
        if counts != {"train": 200, "validation": 25, "test": 25}:
            raise ValueError(f"Unexpected split counts for {arm}: {counts}")
    specs = _load_specs(
        {ARMS[0]: Path(run_960).resolve(), ARMS[1]: Path(run_1920).resolve()},
        resolved_device,
    )
    audit = _contract_audit(manifests, datasets, specs)
    if not audit["passed"]:
        failure = output / "shards" / f"contract_audit_shard_{shard_index}.json"
        _write_json(failure, audit)
        raise RuntimeError(f"HF resolution contract audit failed; see {failure}")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    formula_checks: list[dict[str, Any]] = []
    roundtrip_checks: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in SPLITS:
        for index in range(len(datasets[ARMS[0]][split])):
            if index % shard_count != shard_index:
                continue
            expected_id = datasets[ARMS[0]][split].sample_ids[index]
            for arm in ARMS:
                dataset = datasets[arm][split]
                static = dataset.load_static(index)
                sample_id = str(static["sample_id"])
                if sample_id != expected_id:
                    raise RuntimeError(f"Paired sample mismatch: {sample_id} != {expected_id}")
                metadata = dict(static.get("metadata", {}))
                _validate_sample_contract(sample_id, metadata)
                formula_checks.append(
                    {"split": split, "arm": arm, **_target_formula_audit(static)}
                )
                values = _infer_one(
                    dataset,
                    index,
                    specs[arm],
                    resolved_device,
                    current_faces=static["faces"],
                )
                metrics = _raw_metrics(
                    values["prediction_raw"],
                    values["target_raw"],
                    values["recovery_weight"],
                    values["valid"],
                )
                gt_groups = _sample_gt_groups(
                    values["prediction_raw"], values["target_raw"], values["valid"]
                )
                prediction_rows.append(
                    {
                        "split": split,
                        "arm": arm,
                        "sample_id": sample_id,
                        "object_id": metadata.get("object_id"),
                        "variant_index": metadata.get("variant_index"),
                        "vertex_count": int(values["prediction_raw"].shape[0]),
                        "valid_vertex_count": int(values["valid"].sum().item()),
                        **metrics,
                        **gt_groups,
                    }
                )
                roundtrip_checks.append(
                    {
                        "split": split,
                        "arm": arm,
                        "sample_id": sample_id,
                        "max_abs_output_to_raw_roundtrip_error": float(
                            values["roundtrip_error"]
                        ),
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
                    recovery, _ = _recover_raw_one(
                        static,
                        values["prediction_raw"],
                        values["prediction_normalized"],
                        values["confidence"],
                        output / "reconstruction" / arm / sample_id,
                        specs[arm]["config"],
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
                        f"{arm} {sample_id} epe={metrics['raw_epe']:.9g} "
                        f"top1={gt_groups['gt_top1_epe']:.9g} "
                        f"chamfer={recovery['reconstruction_chamfer']:.9g}",
                        flush=True,
                    )
                del values
            torch.cuda.empty_cache()
    return _write_shard(
        manifests,
        output,
        shard_index,
        shard_count,
        specs,
        audit,
        prediction_rows,
        recovery_rows,
        formula_checks,
        roundtrip_checks,
        arrays,
    )


def merge_hf_resolution_ablation_shards(
    manifest_960_path: str | Path,
    manifest_1920_path: str | Path,
    output_dir: str | Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    manifests = {
        ARMS[0]: Path(manifest_960_path).resolve(),
        ARMS[1]: Path(manifest_1920_path).resolve(),
    }
    output = Path(output_dir).resolve()
    payloads = [
        _read_json(output / "shards" / f"shard_{index}.json")
        for index in range(shard_count)
    ]
    hashes = {arm: _sha256(path) for arm, path in manifests.items()}
    for index, payload in enumerate(payloads):
        if payload.get("shard_index") != index or payload.get("shard_count") != shard_count:
            raise RuntimeError(f"Invalid shard metadata: {index}")
        if payload.get("manifest_sha256") != hashes:
            raise RuntimeError(f"Manifest mismatch in shard {index}")
        if payload.get("contract_audit") != payloads[0].get("contract_audit"):
            raise RuntimeError("Shard contract audits differ")
    array_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(shard_count):
        with np.load(output / "shards" / f"arrays_shard_{index}.npz") as archive:
            for name in archive.files:
                array_chunks[name].append(archive[name])
    return _finalize(
        manifests,
        output,
        payloads[0]["arms"],
        payloads[0]["contract_audit"],
        _concat(payloads, "prediction_rows"),
        _concat(payloads, "recovery_rows"),
        _concat(payloads, "formula_checks"),
        _concat(payloads, "roundtrip_checks"),
        {name: np.concatenate(chunks) for name, chunks in array_chunks.items()},
        shard_count,
    )


def _sample_gt_groups(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    prediction = prediction[valid].double()
    target = target[valid].double()
    magnitude = torch.linalg.vector_norm(target, dim=-1).numpy()
    error = torch.linalg.vector_norm(prediction - target, dim=-1).numpy()
    order = np.argsort(-magnitude, kind="stable")
    top10_count = max(1, math.ceil(0.10 * len(order)))
    top1_count = max(1, math.ceil(0.01 * len(order)))
    return {
        "gt_bottom90_epe": float(error[order[top10_count:]].mean()),
        "gt_top10_epe": float(error[order[:top10_count]].mean()),
        "gt_top1_epe": float(error[order[:top1_count]].mean()),
    }


def _load_specs(
    run_dirs: Mapping[str, Path], device: torch.device
) -> dict[str, dict[str, Any]]:
    specs = {}
    for arm in ARMS:
        run_dir = run_dirs[arm]
        checkpoint_path = run_dir / "checkpoint_latest.pt"
        metrics_path = run_dir / "metrics.json"
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete run for {arm}: {run_dir}")
        config = _run_config(run_dir)
        model = _build_model(config, None, False).to(device)
        checkpoint_payload = load_checkpoint(checkpoint_path, model, map_location=device)
        model.eval()
        amp_enabled, amp_dtype = _amp_settings(config, device)
        specs[arm] = {
            "run_dir": run_dir,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "config": config,
            "model": model,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "optimizer_steps": int(checkpoint_payload.get("optimizer_steps", -1)),
            "native_metrics": _read_json(metrics_path),
        }
    return specs


def _controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("method", None)
    result.pop("experiment_metadata", None)
    result["dataset"].pop("name", None)
    result["image_encoder"].pop("view_chunk_size", None)
    result["image_encoder"].pop("gradient_checkpointing", None)
    result.get("data_loading", {}).pop("multiprocessing_sharing_strategy", None)
    multi = result["multi_object_training"]
    for key in (
        "gradient_accumulation_meshes",
        "validation_every_epochs",
        "checkpoint_every_epochs",
        "checkpoint_epochs",
    ):
        multi.pop(key, None)
    return result


def _initial_state_hash(config: Mapping[str, Any]) -> str:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validation_interval(config: Mapping[str, Any], world_size: int) -> int:
    multi = config["multi_object_training"]
    train_count = int(config["dataset"]["expected_split_counts"]["train"])
    return (
        math.ceil(
            math.ceil(train_count / world_size)
            / int(multi["gradient_accumulation_meshes"])
        )
        * int(multi["validation_every_epochs"])
    )


def _contract_audit(
    manifests: Mapping[str, Path],
    datasets: Mapping[str, Mapping[str, PreparedMeshDataset]],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    configs = {arm: specs[arm]["config"] for arm in ARMS}
    split_ids = {
        arm: {split: list(datasets[arm][split].sample_ids) for split in ("train", *SPLITS)}
        for arm in ARMS
    }
    dataset_audit_path = manifests[ARMS[1]].parent / "contract_audit.json"
    dataset_audit = _read_json(dataset_audit_path)
    world_sizes = {
        arm: int(specs[arm]["native_metrics"].get("distributed_world_size", -1))
        for arm in ARMS
    }
    global_batches = {
        arm: int(specs[arm]["native_metrics"].get("global_batch_meshes", -1))
        for arm in ARMS
    }
    intervals = {
        arm: _validation_interval(configs[arm], world_sizes[arm]) for arm in ARMS
    }
    state_hashes = {arm: _initial_state_hash(configs[arm]) for arm in ARMS}
    modes = {
        arm: configs[arm]["image_encoder"]["feature_construction"]["mode"]
        for arm in ARMS
    }
    fixed = all(
        config.get("seed") == 7
        and config.get("target_mode") == "raw_laplacian"
        and config.get("training", {}).get("loss") == "huber"
        and config.get("training", {}).get("huber_delta") == 0.01
        and not config.get("local_query_jitter", {}).get("enabled", False)
        and config.get("experiment_metadata", {}).get("views") == 28
        and config.get("image_encoder", {}).get("feature_dim") == 64
        and config.get("model", {}).get("hidden_dim") == 256
        and config.get("model", {}).get("num_graph_layers") == 3
        and not config.get("model", {}).get("dynamic_residual_expert", {}).get("enabled", False)
        for config in configs.values()
    )
    passed = bool(
        dataset_audit.get("passed") is True
        and split_ids[ARMS[0]] == split_ids[ARMS[1]]
        and _controlled_config(configs[ARMS[0]]) == _controlled_config(configs[ARMS[1]])
        and len(set(state_hashes.values())) == 1
        and modes == {ARMS[0]: "original_plus_high_frequency", ARMS[1]: "original_plus_high_frequency"}
        and world_sizes == {ARMS[0]: 2, ARMS[1]: 4}
        and global_batches == {ARMS[0]: 2, ARMS[1]: 4}
        and set(intervals.values()) == {500}
        and all(specs[arm]["optimizer_steps"] == 20_000 for arm in ARMS)
        and fixed
    )
    return {
        "passed": passed,
        "manifests": {arm: str(path) for arm, path in manifests.items()},
        "manifest_sha256": {arm: _sha256(path) for arm, path in manifests.items()},
        "native_1920_dataset_contract_audit": str(dataset_audit_path),
        "native_1920_dataset_contract_passed": dataset_audit.get("passed"),
        "same_sample_ids_order_and_splits": split_ids[ARMS[0]] == split_ids[ARMS[1]],
        "only_resolution_and_authorized_execution_resource_differences_after_normalization": _controlled_config(configs[ARMS[0]]) == _controlled_config(configs[ARMS[1]]),
        "same_seeded_model_initialization": len(set(state_hashes.values())) == 1,
        "initial_state_hashes": state_hashes,
        "feature_modes": modes,
        "sampled_image_feature_width": {ARMS[0]: 128, ARMS[1]: 128},
        "optimizer_steps": {arm: specs[arm]["optimizer_steps"] for arm in ARMS},
        "validation_optimizer_step_intervals": intervals,
        "distributed_world_sizes": world_sizes,
        "effective_global_batch_meshes": global_batches,
        "strict_single_variable_training_claim": False,
        "fixed_model_target_loss_confidence_visibility_recovery_contract": fixed,
        "all_differences_from_960": [
            "native observation/render resolution: 960x960 -> 1920x1920",
            "intrinsics pixel rows scaled exactly by 2; extrinsics unchanged",
            "DDP world size: 2 -> 4",
            "effective global batch: 2 -> 4 (unavoidable with one mesh per rank)",
            "validation_every_epochs: 5 -> 10 to keep every 500 optimizer steps",
            "checkpoint epoch schedule normalized to the same optimizer-step positions",
            "1920 execution uses view_chunk_size=4 and gradient checkpointing; tested forward/gradient equivalent",
        ],
    }


def _write_shard(
    manifests: Mapping[str, Path],
    output: Path,
    shard_index: int,
    shard_count: int,
    specs: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    roundtrip_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_dir / f"arrays_shard_{shard_index}.npz",
        **{name: np.concatenate(chunks) for name, chunks in arrays.items() if chunks},
    )
    payload = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "manifest_sha256": {arm: _sha256(path) for arm, path in manifests.items()},
        "contract_audit": dict(audit),
        "arms": {
            arm: {
                "run_dir": str(specs[arm]["run_dir"]),
                "checkpoint": str(specs[arm]["checkpoint"]),
                "checkpoint_sha256": specs[arm]["checkpoint_sha256"],
                "optimizer_steps": specs[arm]["optimizer_steps"],
                "native_metrics": specs[arm]["native_metrics"],
            }
            for arm in ARMS
        },
        "prediction_rows": list(prediction_rows),
        "recovery_rows": list(recovery_rows),
        "formula_checks": list(formula_checks),
        "roundtrip_checks": list(roundtrip_checks),
    }
    _write_json(shard_dir / f"shard_{shard_index}.json", payload)
    return payload


def _global_metrics(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    prediction_rows = []
    group_rows = []
    targets_equal = True
    for split in SPLITS:
        baseline_target = arrays[f"{split}__{ARMS[0]}__target"]
        for arm in ARMS:
            prefix = f"{split}__{arm}"
            prediction = torch.from_numpy(arrays[f"{prefix}__prediction"]).double()
            target_np = arrays[f"{prefix}__target"]
            targets_equal &= np.array_equal(target_np, baseline_target)
            target = torch.from_numpy(target_np).double()
            weight = torch.from_numpy(arrays[f"{prefix}__weight"]).double()
            valid = torch.ones(len(prediction), dtype=torch.bool)
            prediction_rows.append(
                {"split": split, "arm": arm, "vertex_count": len(prediction), **_raw_metrics(prediction, target, weight, valid)}
            )
            group_rows.append(
                {"split": split, "arm": arm, "global_vertex_count": len(prediction), **_sample_gt_groups(prediction, target, valid)}
            )
    return prediction_rows, group_rows, targets_equal


def _recovery_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != 25:
            raise RuntimeError(f"Expected 25 recovery rows for {arm}")
        output.append(
            {
                "arm": arm,
                "sample_count": 25,
                **{field: float(np.mean([float(row[field]) for row in selected])) for field in GEOMETRY_FIELDS},
                "introduced_flipped_faces": sum(int(row["introduced_flipped_faces"]) for row in selected),
                "improved_over_initial": sum(bool(row["improved_over_initial"]) for row in selected),
            }
        )
    return output


def _paired(
    prediction_rows: Sequence[Mapping[str, Any]], recovery_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    prediction = {(row["split"], row["sample_id"], row["arm"]): row for row in prediction_rows}
    recovery = {(row["sample_id"], row["arm"]): row for row in recovery_rows}
    rows = []
    fields = (*RAW_FIELDS, "gt_bottom90_epe", "gt_top10_epe", "gt_top1_epe")
    for split in SPLITS:
        sample_ids = sorted(key[1] for key in prediction if key[0] == split and key[2] == ARMS[0])
        for sample_id in sample_ids:
            baseline = prediction[(split, sample_id, ARMS[0])]
            candidate = prediction[(split, sample_id, ARMS[1])]
            row: dict[str, Any] = {"split": split, "sample_id": sample_id}
            for field in fields:
                row[f"hf960_{field}"] = baseline[field]
                row[f"hf1920_{field}"] = candidate[field]
                row[f"hf1920_minus_hf960_{field}"] = float(candidate[field]) - float(baseline[field])
            if split == "test":
                base_geometry = recovery[(sample_id, ARMS[0])]
                candidate_geometry = recovery[(sample_id, ARMS[1])]
                for field in (
                    "reconstruction_chamfer",
                    "reconstruction_point_to_surface",
                    "reconstruction_normal_consistency",
                    "introduced_flipped_faces",
                ):
                    row[f"hf960_{field}"] = base_geometry[field]
                    row[f"hf1920_{field}"] = candidate_geometry[field]
                    row[f"hf1920_minus_hf960_{field}"] = float(candidate_geometry[field]) - float(base_geometry[field])
            rows.append(row)
    test = [row for row in rows if row["split"] == "test"]
    summary = {
        "lower_raw_epe": sum(row["hf1920_minus_hf960_raw_epe"] < 0 for row in test),
        "lower_raw_rms": sum(row["hf1920_minus_hf960_raw_residual_rms"] < 0 for row in test),
        "lower_recovery_weighted_rms": sum(row["hf1920_minus_hf960_recovery_weighted_raw_residual_rms"] < 0 for row in test),
        "lower_gt_top10": sum(row["hf1920_minus_hf960_gt_top10_epe"] < 0 for row in test),
        "lower_gt_top1": sum(row["hf1920_minus_hf960_gt_top1_epe"] < 0 for row in test),
        "lower_chamfer": sum(row["hf1920_minus_hf960_reconstruction_chamfer"] < 0 for row in test),
        "lower_p2s": sum(row["hf1920_minus_hf960_reconstruction_point_to_surface"] < 0 for row in test),
        "higher_normal_consistency": sum(row["hf1920_minus_hf960_reconstruction_normal_consistency"] > 0 for row in test),
        "fewer_or_equal_flips": sum(row["hf1920_minus_hf960_introduced_flipped_faces"] <= 0 for row in test),
    }
    return rows, summary


def _one(rows: Sequence[Mapping[str, Any]], arm: str, split: str | None = None) -> Mapping[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and (split is None or row.get("split") == split)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {arm}/{split}")
    return selected[0]


def _cost(arms: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        metrics = arms[arm]["native_metrics"]
        runtime = float(metrics["runtime_seconds"])
        world = int(metrics["distributed_world_size"])
        output.append(
            {
                "arm": arm,
                "runtime_seconds": runtime,
                "runtime_hours": runtime / 3600.0,
                "distributed_world_size": world,
                "global_batch_meshes": int(metrics["global_batch_meshes"]),
                "peak_gpu_memory_mb_per_rank": float(metrics["peak_gpu_memory_mb"]),
                "mean_optimizer_step_seconds": float(metrics["mean_optimizer_step_seconds"]),
                "total_gpu_hours": runtime * world / 3600.0,
            }
        )
    return output


def _finalize(
    manifests: Mapping[str, Path],
    output: Path,
    arms: Mapping[str, Any],
    preflight: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    roundtrip_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    shard_count: int,
) -> dict[str, Any]:
    expected = {"prediction_rows": 100, "recovery_rows": 50, "formula_checks": 100, "roundtrip_checks": 100}
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "formula_checks": len(formula_checks),
        "roundtrip_checks": len(roundtrip_checks),
    }
    aggregate, groups, targets_equal = _global_metrics(arrays)
    max_formula = max(float(row["current_graph_proxy_raw_target_max_abs_error"]) for row in formula_checks)
    max_roundtrip = max(float(row["max_abs_output_to_raw_roundtrip_error"]) for row in roundtrip_checks)
    audit = {
        **dict(preflight),
        "counts": counts,
        "counts_match": counts == expected,
        "raw_targets_identical_across_960_and_1920": targets_equal,
        "maximum_current_graph_proxy_raw_target_error": max_formula,
        "maximum_output_to_raw_roundtrip_error": max_roundtrip,
    }
    audit["passed"] = bool(preflight["passed"] and counts == expected and targets_equal and max_formula <= 1e-7 and max_roundtrip <= 1e-6)
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Final HF resolution contract audit failed")
    recovery_aggregate = _recovery_metrics(recovery_rows)
    paired_rows, paired_summary = _paired(prediction_rows, recovery_rows)
    costs = _cost(arms)
    base_prediction = _one(aggregate, ARMS[0], "test")
    high_prediction = _one(aggregate, ARMS[1], "test")
    base_groups = _one(groups, ARMS[0], "test")
    high_groups = _one(groups, ARMS[1], "test")
    base_recovery = _one(recovery_aggregate, ARMS[0])
    high_recovery = _one(recovery_aggregate, ARMS[1])
    conclusion = {
        "1920_lowers_test_top10": high_groups["gt_top10_epe"] < base_groups["gt_top10_epe"],
        "1920_lowers_test_top1": high_groups["gt_top1_epe"] < base_groups["gt_top1_epe"],
        "1920_lowers_test_raw_rms": high_prediction["raw_residual_rms"] < base_prediction["raw_residual_rms"],
        "1920_lowers_test_recovery_weighted_rms": high_prediction["recovery_weighted_raw_residual_rms"] < base_prediction["recovery_weighted_raw_residual_rms"],
        "1920_tail_improvement_exceeds_bottom90_improvement": (
            (base_groups["gt_top10_epe"] - high_groups["gt_top10_epe"]) / base_groups["gt_top10_epe"]
            > (base_groups["gt_bottom90_epe"] - high_groups["gt_bottom90_epe"]) / base_groups["gt_bottom90_epe"]
        ),
        "1920_lowers_mean_chamfer": high_recovery["reconstruction_chamfer"] < base_recovery["reconstruction_chamfer"],
        "1920_lowers_mean_p2s": high_recovery["reconstruction_point_to_surface"] < base_recovery["reconstruction_point_to_surface"],
        "1920_improves_mean_normal": high_recovery["reconstruction_normal_consistency"] > base_recovery["reconstruction_normal_consistency"],
        "prediction_improves_but_downstream_does_not": bool(
            high_groups["gt_top10_epe"] < base_groups["gt_top10_epe"]
            and high_groups["gt_top1_epe"] < base_groups["gt_top1_epe"]
            and not (
                high_recovery["reconstruction_chamfer"] < base_recovery["reconstruction_chamfer"]
                and high_recovery["reconstruction_point_to_surface"] < base_recovery["reconstruction_point_to_surface"]
            )
        ),
    }
    summary = {
        "experiment": "Sofa50 direct-raw HF 960x960 vs native 1920x1920",
        "manifests": {arm: str(path) for arm, path in manifests.items()},
        "evaluation_shards": shard_count,
        "arms": dict(arms),
        "contract_audit": audit,
        "prediction_aggregate": aggregate,
        "gt_raw_laplacian_magnitude_groups": groups,
        "recovery_aggregate": recovery_aggregate,
        "paired_summary": paired_summary,
        "runtime_and_gpu_memory": costs,
        "conclusion": conclusion,
    }
    _write_json(output / "hf_resolution_ablation_summary.json", summary)
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "prediction_aggregate.csv", aggregate)
    _write_csv(output / "gt_raw_laplacian_magnitude_groups.csv", groups)
    _write_csv(output / "recovery_per_sample.csv", recovery_rows)
    _write_csv(output / "recovery_aggregate.csv", recovery_aggregate)
    _write_csv(output / "paired_per_sample.csv", paired_rows)
    _write_csv(output / "runtime_and_gpu_memory.csv", costs)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: Mapping[str, Any]) -> str:
    groups = {(row["split"], row["arm"]): row for row in summary["gt_raw_laplacian_magnitude_groups"]}
    lines = [
        "# Sofa50 HF Resolution Ablation: 960 vs 1920",
        "",
        "## Contract audit",
        "",
        f"- Audit passed: `{summary['contract_audit']['passed']}`.",
        "- Same 250 sample IDs/splits, 28 camera poses, current graphs, P_proxy/targets, visibility tensors, direct-raw HF architecture, seed, loss, and recovery solver.",
        "- 1920 RGB observations were natively rerendered with the same CPU reference renderer; they are not resized 960 images.",
        "- This is not a strict single-variable training claim: 4-GPU DDP has effective global batch 4 versus 2 for the 960 baseline. Validation remains every 500 optimizer steps.",
        "- View chunking and gradient checkpointing are execution-only and passed forward/gradient equivalence tests.",
        "",
        "## Validation/test prediction",
        "",
        "| Split | Arm | Raw EPE | Raw RMS | Raw max | Raw cosine | Weighted RMS | Bottom90 | Top10 | Top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["prediction_aggregate"]:
        group = groups[(row["split"], row["arm"])]
        lines.append(
            f"| {row['split']} | {row['arm']} | {_f(row['raw_epe'])} | {_f(row['raw_residual_rms'])} | {_f(row['raw_residual_maximum'])} | {_f(row['raw_global_cosine'])} | {_f(row['recovery_weighted_raw_residual_rms'])} | {_f(group['gt_bottom90_epe'])} | {_f(group['gt_top10_epe'])} | {_f(group['gt_top1_epe'])} |"
        )
    lines.extend(["", "## Test downstream recovery", "", "| Arm | Chamfer | P2S | Normal | Flips | Improved / 25 |", "|---|---:|---:|---:|---:|---:|"])
    for row in summary["recovery_aggregate"]:
        lines.append(
            f"| {row['arm']} | {_f(row['reconstruction_chamfer'])} | {_f(row['reconstruction_point_to_surface'])} | {_f(row['reconstruction_normal_consistency'])} | {row['introduced_flipped_faces']} | {row['improved_over_initial']} |"
        )
    paired = summary["paired_summary"]
    lines.extend(
        [
            "",
            "## Paired 1920 wins / 25",
            "",
            f"- Lower Raw EPE / RMS / recovery-weighted RMS: {paired['lower_raw_epe']} / {paired['lower_raw_rms']} / {paired['lower_recovery_weighted_rms']}.",
            f"- Lower GT-magnitude Top10 / Top1: {paired['lower_gt_top10']} / {paired['lower_gt_top1']}.",
            f"- Lower Chamfer / P2S, higher normal, fewer/equal flips: {paired['lower_chamfer']} / {paired['lower_p2s']} / {paired['higher_normal_consistency']} / {paired['fewer_or_equal_flips']}.",
            "",
            "## Runtime and memory",
            "",
            "| Arm | Runtime h | GPUs | Global batch | Step seconds | Peak memory/rank MB | Total GPU-hours |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["runtime_and_gpu_memory"]:
        lines.append(
            f"| {row['arm']} | {_f(row['runtime_hours'])} | {row['distributed_world_size']} | {row['global_batch_meshes']} | {_f(row['mean_optimizer_step_seconds'])} | {_f(row['peak_gpu_memory_mb_per_rank'])} | {_f(row['total_gpu_hours'])} |"
        )
    result = summary["conclusion"]
    lines.extend(
        [
            "",
            "## Concise conclusion",
            "",
            f"- 1920 lowers Top10 / Top1: `{result['1920_lowers_test_top10']}` / `{result['1920_lowers_test_top1']}`.",
            f"- Raw RMS / recovery-weighted RMS improve together: `{result['1920_lowers_test_raw_rms']}` / `{result['1920_lowers_test_recovery_weighted_rms']}`.",
            f"- Tail-relative gain exceeds Bottom90 gain: `{result['1920_tail_improvement_exceeds_bottom90_improvement']}`.",
            f"- Chamfer / P2S / normal improve: `{result['1920_lowers_mean_chamfer']}` / `{result['1920_lowers_mean_p2s']}` / `{result['1920_improves_mean_normal']}`.",
            f"- Prediction improves but downstream does not: `{result['prediction_improves_but_downstream_does_not']}`. No recovery method was changed or added.",
        ]
    )
    return "\n".join(lines) + "\n"

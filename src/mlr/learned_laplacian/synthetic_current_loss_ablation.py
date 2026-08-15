from __future__ import annotations

import copy
import csv
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
    RAW_METRIC_FIELDS,
    _infer_one,
    _raw_metrics,
    _recover_raw_one,
    _run_config,
    _sha256,
    _target_formula_audit,
    _validate_sample_contract,
)
from .trainer import _seed_everything, load_checkpoint


ARMS = ("Huber_delta_0p01", "raw_MSE")
SPLITS = ("validation", "test")
TAIL_GROUPS = ("bottom_90_percent", "top_10_percent", "top_1_percent")
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_loss_ablation(
    manifest_path: str | Path,
    huber_run: str | Path,
    mse_run: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    shard_index: int | None = None,
    shard_count: int = 1,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The loss ablation evaluator requires an available CUDA device.")
    shard_index = _validated_shard_index(shard_index, shard_count)

    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", *SPLITS)
    }
    validate_disjoint_splits(*datasets.values())
    counts = {name: len(dataset) for name, dataset in datasets.items()}
    if counts != {"train": 200, "validation": 25, "test": 25}:
        raise ValueError(f"Unexpected split counts: {counts}.")

    specs = _load_specs(
        {
            ARMS[0]: Path(huber_run).resolve(),
            ARMS[1]: Path(mse_run).resolve(),
        },
        resolved_device,
    )
    audit = _contract_audit(manifest, datasets, specs)
    if not audit["passed"]:
        failure = output / "shards" / f"contract_audit_shard_{shard_index}.json"
        _write_json(failure, audit)
        raise RuntimeError(f"Loss ablation contract audit failed; see {failure}.")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    formula_checks: list[dict[str, Any]] = []
    roundtrip_checks: list[dict[str, Any]] = []
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
            formula = _target_formula_audit(static)
            formula_checks.append({"split": split, **formula})
            for arm, spec in specs.items():
                values = _infer_one(
                    dataset,
                    index,
                    spec,
                    resolved_device,
                    current_faces=static["faces"],
                )
                metrics = _raw_metrics(
                    values["prediction_raw"],
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
                        "vertex_count": int(values["prediction_raw"].shape[0]),
                        "valid_vertex_count": int(values["valid"].sum().item()),
                        "mean_confidence": float(values["confidence"].mean().item()),
                        "visible_vertex_fraction": float(
                            (values["visibility_count"] > 0).float().mean().item()
                        ),
                        **metrics,
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
                arrays[f"{prefix}__recovery_weight"].append(
                    values["recovery_weight"].numpy()[valid].astype(np.float64)
                )
                if split == "test":
                    recovery_dir = output / "reconstruction" / arm / sample_id
                    recovery, _ = _recover_raw_one(
                        static,
                        values["prediction_raw"],
                        values["prediction_normalized"],
                        values["confidence"],
                        recovery_dir,
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
                        f"{arm} {sample_id} epe={metrics['raw_epe']:.9g} "
                        f"chamfer={recovery['reconstruction_chamfer']:.9g} "
                        f"improved={recovery['improved_over_initial']}",
                        flush=True,
                    )
                del values
            torch.cuda.empty_cache()

    return _write_shard(
        manifest,
        output,
        shard_index=shard_index,
        shard_count=shard_count,
        specs=specs,
        audit=audit,
        prediction_rows=prediction_rows,
        recovery_rows=recovery_rows,
        formula_checks=formula_checks,
        roundtrip_checks=roundtrip_checks,
        arrays=arrays,
    )


def merge_loss_ablation_shards(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive.")
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    shard_dir = output / "shards"
    payloads = [
        _read_json(shard_dir / f"shard_{index}.json") for index in range(shard_count)
    ]
    for index, payload in enumerate(payloads):
        if payload.get("shard_index") != index or payload.get("shard_count") != shard_count:
            raise RuntimeError(f"Invalid metadata in shard {index}.")
        if payload.get("manifest_sha256") != _sha256(manifest):
            raise RuntimeError(f"Manifest mismatch in shard {index}.")
    if any(payload["contract_audit"] != payloads[0]["contract_audit"] for payload in payloads[1:]):
        raise RuntimeError("Shard contract audits do not match.")
    if any(payload["arms"] != payloads[0]["arms"] for payload in payloads[1:]):
        raise RuntimeError("Shard arm metadata do not match.")

    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(shard_count):
        with np.load(shard_dir / f"arrays_shard_{index}.npz") as archive:
            for name in archive.files:
                arrays[name].append(archive[name])
    merged_arrays = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    return _finalize(
        manifest,
        output,
        arms=payloads[0]["arms"],
        preflight=payloads[0]["contract_audit"],
        prediction_rows=_concat(payloads, "prediction_rows"),
        recovery_rows=_concat(payloads, "recovery_rows"),
        formula_checks=_concat(payloads, "formula_checks"),
        roundtrip_checks=_concat(payloads, "roundtrip_checks"),
        arrays=merged_arrays,
        shard_count=shard_count,
    )


def _load_specs(
    run_dirs: Mapping[str, Path], device: torch.device
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dir = run_dirs[arm]
        checkpoint = run_dir / "checkpoint_latest.pt"
        metrics_path = run_dir / "metrics.json"
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete run directory for {arm}: {run_dir}")
        config = _run_config(run_dir)
        model = _build_model(config, None, False).to(device)
        payload = load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        amp_enabled, amp_dtype = _amp_settings(config, device)
        specs[arm] = {
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "config": config,
            "model": model,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "optimizer_steps": int(payload.get("optimizer_steps", -1)),
            "native_metrics": _read_json(metrics_path),
        }
    return specs


def _contract_audit(
    manifest: Path,
    datasets: Mapping[str, PreparedMeshDataset],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    configs = {arm: spec["config"] for arm, spec in specs.items()}
    losses = {arm: str(config["training"]["loss"]) for arm, config in configs.items()}
    initial_hashes = {arm: _initial_state_hash(config) for arm, config in configs.items()}
    controlled = {arm: _controlled_config(config) for arm, config in configs.items()}
    controlled_equal = len({json.dumps(value, sort_keys=True) for value in controlled.values()}) == 1
    split_ids = {split: list(dataset.sample_ids) for split, dataset in datasets.items()}
    run_split_ids = {
        arm: _run_manifest_split_ids(spec["run_dir"] / "run_config.json")
        for arm, spec in specs.items()
    }
    world_sizes = {
        ARMS[0]: 1,
        ARMS[1]: int(
            configs[ARMS[1]].get("experiment_metadata", {}).get(
                "distributed_world_size", 1
            )
        ),
    }
    validation_step_intervals = {
        arm: _validation_step_interval(configs[arm], world_size=world_sizes[arm])
        for arm in ARMS
    }
    global_batch = {
        ARMS[0]: int(configs[ARMS[0]]["multi_object_training"]["gradient_accumulation_meshes"]),
        ARMS[1]: world_sizes[ARMS[1]]
        * int(configs[ARMS[1]]["multi_object_training"]["gradient_accumulation_meshes"]),
    }
    expected_losses = losses == {ARMS[0]: "huber", ARMS[1]: "mse"}
    fixed_semantics = all(
        config.get("target_mode") == "raw_laplacian"
        and config.get("training", {}).get("prediction_loss_space") == "output_representation"
        and not config.get("local_query_jitter", {}).get("enabled", False)
        and config.get("experiment_metadata", {}).get("views") == 28
        for config in configs.values()
    )
    optimizer_steps = {arm: int(spec["optimizer_steps"]) for arm, spec in specs.items()}
    run_splits_match = all(value == split_ids for value in run_split_ids.values())
    passed = bool(
        manifest.is_file()
        and controlled_equal
        and expected_losses
        and fixed_semantics
        and len(set(initial_hashes.values())) == 1
        and all(value == 20_000 for value in optimizer_steps.values())
        and abs(validation_step_intervals[ARMS[0]] - validation_step_intervals[ARMS[1]])
        <= 10
        and run_splits_match
    )
    return {
        "passed": passed,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "losses": losses,
        "only_controlled_loss_diff_after_schedule_normalization": controlled_equal,
        "initial_state_hashes": initial_hashes,
        "initial_states_equal": len(set(initial_hashes.values())) == 1,
        "optimizer_steps": optimizer_steps,
        "validation_optimizer_step_intervals": validation_step_intervals,
        "run_manifest_sample_ids_match": run_splits_match,
        "fixed_model_target_visibility_confidence_recovery_contract": fixed_semantics,
        "authorized_batch_difference": {
            "baseline_global_batch_meshes": global_batch[ARMS[0]],
            "mse_global_batch_meshes": global_batch[ARMS[1]],
            "strict_single_variable_training_claim": False,
            "reason": "User authorized use of currently free GPUs while retaining two meshes accumulated per rank.",
        },
    }


def _controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("method", None)
    result.pop("experiment_metadata", None)
    result["training"].pop("loss", None)
    result.get("data_loading", {}).pop("multiprocessing_sharing_strategy", None)
    multi = result["multi_object_training"]
    for key in (
        "epochs",
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


def _validation_step_interval(config: Mapping[str, Any], *, world_size: int) -> int:
    multi = config["multi_object_training"]
    train_count = int(config["dataset"]["expected_split_counts"]["train"])
    samples_per_rank = math.ceil(train_count / world_size)
    steps_per_epoch = math.ceil(
        samples_per_rank / int(multi["gradient_accumulation_meshes"])
    )
    return steps_per_epoch * int(multi["validation_every_epochs"])


def _run_manifest_split_ids(path: Path) -> dict[str, list[str]]:
    source = _read_json(path).get("source_manifest", {})
    if isinstance(source, str):
        value = _read_json(Path(source))
    elif isinstance(source, Mapping):
        value = source
    else:
        raise ValueError(f"Invalid source_manifest in {path}.")
    samples = value.get("samples", [])
    result = {split: [] for split in ("train", "validation", "test")}
    for item in samples:
        split = str(item.get("split"))
        if split in result:
            result[split].append(str(item.get("sample_id")))
    return result


def _write_shard(
    manifest: Path,
    output: Path,
    *,
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
    packed = {name: np.concatenate(chunks) for name, chunks in arrays.items() if chunks}
    np.savez_compressed(shard_dir / f"arrays_shard_{shard_index}.npz", **packed)
    payload = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "manifest_sha256": _sha256(manifest),
        "contract_audit": dict(audit),
        "arms": _arm_metadata(specs),
        "prediction_rows": list(prediction_rows),
        "recovery_rows": list(recovery_rows),
        "formula_checks": list(formula_checks),
        "roundtrip_checks": list(roundtrip_checks),
    }
    _write_json(shard_dir / f"shard_{shard_index}.json", payload)
    return payload


def _finalize(
    manifest: Path,
    output: Path,
    *,
    arms: Mapping[str, Any],
    preflight: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    roundtrip_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    shard_count: int,
) -> dict[str, Any]:
    expected_counts = {
        "prediction_rows": 100,
        "recovery_rows": 50,
        "formula_checks": 50,
        "roundtrip_checks": 100,
    }
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "formula_checks": len(formula_checks),
        "roundtrip_checks": len(roundtrip_checks),
    }
    max_formula = max(
        float(row["current_graph_proxy_raw_target_max_abs_error"])
        for row in formula_checks
    )
    max_roundtrip = max(
        float(row["max_abs_output_to_raw_roundtrip_error"])
        for row in roundtrip_checks
    )
    audit = {
        **dict(preflight),
        "passed": bool(
            preflight["passed"]
            and counts == expected_counts
            and max_formula <= 1e-7
            and max_roundtrip <= 1e-6
        ),
        "counts": counts,
        "counts_match": counts == expected_counts,
        "maximum_current_graph_proxy_raw_target_error": max_formula,
        "maximum_output_to_raw_roundtrip_error": max_roundtrip,
    }
    if not audit["passed"]:
        _write_json(output / "contract_audit.json", audit)
        raise RuntimeError("Final loss ablation contract audit failed.")

    prediction_aggregate, tail_rows = _global_prediction_aggregates(arrays)
    recovery_aggregate = _aggregate_recovery(recovery_rows)
    paired_rows, paired_summary = _paired_comparisons(prediction_rows, recovery_rows)
    huber_tail = _row_by(tail_rows, split="test", arm=ARMS[0])
    mse_tail = _row_by(tail_rows, split="test", arm=ARMS[1])
    huber_recovery = _row_by(recovery_aggregate, arm=ARMS[0])
    mse_recovery = _row_by(recovery_aggregate, arm=ARMS[1])
    conclusion = {
        "mse_lowers_test_top_10_percent_error": (
            mse_tail["top_10_percent_mean_raw_error_epe"]
            < huber_tail["top_10_percent_mean_raw_error_epe"]
        ),
        "mse_lowers_test_top_1_percent_error": (
            mse_tail["top_1_percent_mean_raw_error_epe"]
            < huber_tail["top_1_percent_mean_raw_error_epe"]
        ),
        "mse_lowers_mean_test_chamfer": (
            mse_recovery["reconstruction_chamfer"]
            < huber_recovery["reconstruction_chamfer"]
        ),
        "mse_lowers_mean_test_p2s": (
            mse_recovery["reconstruction_point_to_surface"]
            < huber_recovery["reconstruction_point_to_surface"]
        ),
        "mse_improves_mean_test_normal_consistency": (
            mse_recovery["reconstruction_normal_consistency"]
            > huber_recovery["reconstruction_normal_consistency"]
        ),
    }
    conclusion["tail_reduction_corresponds_to_chamfer_and_p2s_improvement"] = bool(
        conclusion["mse_lowers_test_top_10_percent_error"]
        and conclusion["mse_lowers_test_top_1_percent_error"]
        and conclusion["mse_lowers_mean_test_chamfer"]
        and conclusion["mse_lowers_mean_test_p2s"]
    )
    summary = {
        "experiment": "Sofa50 Arm B Huber(delta=0.01) vs raw MSE",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "evaluation_shards": shard_count,
        "arms": dict(arms),
        "contract_audit": audit,
        "prediction_aggregate": prediction_aggregate,
        "gt_raw_laplacian_magnitude_groups": tail_rows,
        "recovery_aggregate": recovery_aggregate,
        "paired_summary": paired_summary,
        "conclusion": conclusion,
    }
    _write_json(output / "contract_audit.json", audit)
    _write_json(output / "loss_ablation_summary.json", summary)
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "prediction_aggregate.csv", prediction_aggregate)
    _write_csv(output / "gt_raw_laplacian_magnitude_groups.csv", tail_rows)
    _write_csv(output / "recovery_per_sample.csv", recovery_rows)
    _write_csv(output / "recovery_aggregate.csv", recovery_aggregate)
    _write_csv(output / "paired_per_sample.csv", paired_rows)
    _write_json(output / "paired_summary.json", paired_summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _global_prediction_aggregates(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for arm in ARMS:
            prefix = f"{split}__{arm}"
            prediction = torch.from_numpy(arrays[f"{prefix}__prediction"]).double()
            target = torch.from_numpy(arrays[f"{prefix}__target"]).double()
            weight = torch.from_numpy(arrays[f"{prefix}__recovery_weight"]).double()
            valid = torch.ones(len(prediction), dtype=torch.bool)
            aggregate.append(
                {
                    "split": split,
                    "arm": arm,
                    "vertex_count": len(prediction),
                    **_raw_metrics(prediction, target, weight, valid),
                }
            )
            magnitude = torch.linalg.vector_norm(target, dim=-1).numpy()
            error = torch.linalg.vector_norm(prediction - target, dim=-1).numpy()
            order = np.argsort(-magnitude, kind="stable")
            top10_count = max(1, int(math.ceil(0.10 * len(order))))
            top1_count = max(1, int(math.ceil(0.01 * len(order))))
            top10 = order[:top10_count]
            top1 = order[:top1_count]
            bottom90 = order[top10_count:]
            tail_rows.append(
                {
                    "split": split,
                    "arm": arm,
                    "global_vertex_count": len(order),
                    "bottom_90_percent_vertex_count": len(bottom90),
                    "bottom_90_percent_mean_raw_error_epe": float(error[bottom90].mean()),
                    "top_10_percent_vertex_count": len(top10),
                    "top_10_percent_minimum_gt_raw_laplacian_magnitude": float(
                        magnitude[top10].min()
                    ),
                    "top_10_percent_mean_raw_error_epe": float(error[top10].mean()),
                    "top_1_percent_vertex_count": len(top1),
                    "top_1_percent_minimum_gt_raw_laplacian_magnitude": float(
                        magnitude[top1].min()
                    ),
                    "top_1_percent_mean_raw_error_epe": float(error[top1].mean()),
                }
            )
    return aggregate, tail_rows


def _aggregate_recovery(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != 25:
            raise RuntimeError(f"Expected 25 recovery rows for {arm}, found {len(selected)}.")
        output.append(
            {
                "arm": arm,
                "sample_count": len(selected),
                **{field: _mean(selected, field) for field in GEOMETRY_FIELDS},
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
    return output


def _paired_comparisons(
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction = {
        (str(row["split"]), str(row["sample_id"]), str(row["arm"])): row
        for row in prediction_rows
    }
    recovery = {
        (str(row["sample_id"]), str(row["arm"])): row for row in recovery_rows
    }
    fields = (*RAW_METRIC_FIELDS,)
    geometry = (
        "reconstruction_chamfer",
        "reconstruction_point_to_surface",
        "reconstruction_normal_consistency",
        "introduced_flipped_faces",
    )
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        sample_ids = sorted(
            {key[1] for key in prediction if key[0] == split and key[2] == ARMS[0]}
        )
        for sample_id in sample_ids:
            huber = prediction[(split, sample_id, ARMS[0])]
            mse = prediction[(split, sample_id, ARMS[1])]
            row: dict[str, Any] = {
                "split": split,
                "sample_id": sample_id,
                "object_id": huber.get("object_id"),
                "variant_index": huber.get("variant_index"),
            }
            for field in fields:
                row[f"huber_{field}"] = huber[field]
                row[f"mse_{field}"] = mse[field]
                row[f"mse_minus_huber_{field}"] = float(mse[field]) - float(huber[field])
            if split == "test":
                huber_recovery = recovery[(sample_id, ARMS[0])]
                mse_recovery = recovery[(sample_id, ARMS[1])]
                for field in geometry:
                    row[f"huber_{field}"] = huber_recovery[field]
                    row[f"mse_{field}"] = mse_recovery[field]
                    row[f"mse_minus_huber_{field}"] = (
                        float(mse_recovery[field]) - float(huber_recovery[field])
                    )
            rows.append(row)
    test_rows = [row for row in rows if row["split"] == "test"]
    summary = {
        "delta_definition": "raw_MSE minus Huber_delta_0p01",
        "validation_sample_count": sum(row["split"] == "validation" for row in rows),
        "test_sample_count": len(test_rows),
        "test_mse_wins_lower_raw_epe": sum(
            row["mse_minus_huber_raw_epe"] < 0 for row in test_rows
        ),
        "test_mse_wins_lower_chamfer": sum(
            row["mse_minus_huber_reconstruction_chamfer"] < 0 for row in test_rows
        ),
        "test_mse_wins_lower_p2s": sum(
            row["mse_minus_huber_reconstruction_point_to_surface"] < 0
            for row in test_rows
        ),
        "test_mse_wins_higher_normal_consistency": sum(
            row["mse_minus_huber_reconstruction_normal_consistency"] > 0
            for row in test_rows
        ),
        "test_mse_wins_no_more_introduced_flips": sum(
            row["mse_minus_huber_introduced_flipped_faces"] <= 0
            for row in test_rows
        ),
    }
    return rows, summary


def _arm_metadata(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        arm: {
            "run_dir": str(spec["run_dir"]),
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "optimizer_steps": spec["optimizer_steps"],
            "loss": spec["config"]["training"]["loss"],
            "native_best_validation_loss": spec["native_metrics"].get("best_selection_loss"),
            "native_final_validation_loss": spec["native_metrics"].get("final_validation_loss"),
            "runtime_seconds": spec["native_metrics"].get("runtime_seconds"),
            "distributed_world_size": spec["native_metrics"].get("distributed_world_size"),
            "global_batch_meshes": spec["native_metrics"].get("global_batch_meshes"),
        }
        for arm, spec in specs.items()
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 Arm B Raw-Loss Ablation",
        "",
        "## Contract",
        "",
        "- Compared `Huber(delta=0.01)` with raw component-wise MSE.",
        "- Both checkpoints use C2F2, 28 views, current-query/current-graph raw Laplacian targets, seed 7, local jitter off, and 20,000 optimizer steps.",
        "- The 3-GPU MSE run retains accumulation=2 per rank, so global batch is 6 versus the baseline's 2. Its validation interval is 510 versus 500 optimizer steps. These authorized resource-driven differences prevent a strict single-variable training claim.",
        f"- Contract audit passed: `{summary['contract_audit']['passed']}`.",
        "",
        "## Global raw prediction metrics",
        "",
        "| Split | Arm | EPE | RMS | Max | Cosine | Recovery-weighted RMS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["prediction_aggregate"]:
        lines.append(
            f"| {row['split']} | {row['arm']} | {_f(row['raw_epe'])} | "
            f"{_f(row['raw_residual_rms'])} | {_f(row['raw_residual_maximum'])} | "
            f"{_f(row['raw_global_cosine'])} | {_f(row['recovery_weighted_raw_residual_rms'])} |"
        )
    lines.extend(
        [
            "",
            "## Global GT raw-Laplacian magnitude groups",
            "",
            "| Split | Arm | Bottom 90% EPE | Top 10% EPE | Top 1% EPE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["gt_raw_laplacian_magnitude_groups"]:
        lines.append(
            f"| {row['split']} | {row['arm']} | "
            f"{_f(row['bottom_90_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_10_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_1_percent_mean_raw_error_epe'])} |"
        )
    lines.extend(
        [
            "",
            "## Test downstream recovery",
            "",
            "| Arm | Chamfer | P2S | Normal consistency | Introduced flips | Improved / 25 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["recovery_aggregate"]:
        lines.append(
            f"| {row['arm']} | {_f(row['reconstruction_chamfer'])} | "
            f"{_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['reconstruction_normal_consistency'])} | "
            f"{row['introduced_flipped_faces']} | {row['improved_over_initial']} |"
        )
    paired = summary["paired_summary"]
    result = summary["conclusion"]
    lines.extend(
        [
            "",
            "## Paired result",
            "",
            f"- MSE lower test raw EPE: {paired['test_mse_wins_lower_raw_epe']}/25 samples.",
            f"- MSE lower Chamfer: {paired['test_mse_wins_lower_chamfer']}/25 samples.",
            f"- MSE lower P2S: {paired['test_mse_wins_lower_p2s']}/25 samples.",
            f"- MSE higher normal consistency: {paired['test_mse_wins_higher_normal_consistency']}/25 samples.",
            f"- MSE no more introduced flips: {paired['test_mse_wins_no_more_introduced_flips']}/25 samples.",
            "",
            "## Interpretation",
            "",
            f"- MSE lowers global test top-10% GT-magnitude error: `{result['mse_lowers_test_top_10_percent_error']}`.",
            f"- MSE lowers global test top-1% GT-magnitude error: `{result['mse_lowers_test_top_1_percent_error']}`.",
            f"- Tail reduction corresponds to lower mean Chamfer and P2S: `{result['tail_reduction_corresponds_to_chamfer_and_p2s_improvement']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _row_by(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}.")
    return matches[0]


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _f(value: Any) -> str:
    return f"{float(value):.9g}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

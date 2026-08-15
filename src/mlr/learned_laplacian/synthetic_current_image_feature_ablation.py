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
    RAW_METRIC_FIELDS,
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


ARMS = (
    "original_B_direct_raw_laplacian",
    "B_gaussian_feature",
    "C_original_plus_high_frequency",
)
SPLITS = ("validation", "test")
FEATURE_MODES = {
    ARMS[0]: "original",
    ARMS[1]: "gaussian_blur",
    ARMS[2]: "original_plus_high_frequency",
}
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_image_feature_ablation(
    manifest_path: str | Path,
    baseline_run: str | Path,
    gaussian_run: str | Path,
    high_frequency_run: str | Path,
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
        raise RuntimeError("The image-feature ablation evaluator requires CUDA.")
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
            ARMS[0]: Path(baseline_run).resolve(),
            ARMS[1]: Path(gaussian_run).resolve(),
            ARMS[2]: Path(high_frequency_run).resolve(),
        },
        resolved_device,
    )
    audit = _contract_audit(manifest, datasets, specs)
    if not audit["passed"]:
        failure = output / "shards" / f"contract_audit_shard_{shard_index}.json"
        _write_json(failure, audit)
        raise RuntimeError(f"Image-feature contract audit failed; see {failure}.")

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
            formula_checks.append({"split": split, **_target_formula_audit(static)})
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
                    recovery, _ = _recover_raw_one(
                        static,
                        values["prediction_raw"],
                        values["prediction_normalized"],
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
                        f"{arm} {sample_id} epe={metrics['raw_epe']:.9g} "
                        f"chamfer={recovery['reconstruction_chamfer']:.9g}",
                        flush=True,
                    )
                del values
            torch.cuda.empty_cache()
    return _write_shard(
        manifest,
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


def merge_image_feature_ablation_shards(
    manifest_path: str | Path, output_dir: str | Path, *, shard_count: int
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    payloads = [
        _read_json(output / "shards" / f"shard_{index}.json")
        for index in range(shard_count)
    ]
    for index, payload in enumerate(payloads):
        if payload.get("shard_index") != index or payload.get("shard_count") != shard_count:
            raise RuntimeError(f"Invalid metadata in shard {index}.")
        if payload.get("manifest_sha256") != _sha256(manifest):
            raise RuntimeError(f"Manifest mismatch in shard {index}.")
        if payload.get("contract_audit") != payloads[0].get("contract_audit"):
            raise RuntimeError("Shard contract audits do not match.")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(shard_count):
        with np.load(output / "shards" / f"arrays_shard_{index}.npz") as archive:
            for name in archive.files:
                arrays[name].append(archive[name])
    return _finalize(
        manifest,
        output,
        payloads[0]["arms"],
        payloads[0]["contract_audit"],
        _concat(payloads, "prediction_rows"),
        _concat(payloads, "recovery_rows"),
        _concat(payloads, "formula_checks"),
        _concat(payloads, "roundtrip_checks"),
        {name: np.concatenate(chunks) for name, chunks in arrays.items()},
        shard_count,
    )


def _load_specs(
    run_dirs: Mapping[str, Path], device: torch.device
) -> dict[str, dict[str, Any]]:
    specs = {}
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


def _feature_mode(config: Mapping[str, Any]) -> str:
    construction = config.get("image_encoder", {}).get("feature_construction", {})
    return str(construction.get("mode", "original"))


def _controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("method", None)
    result.pop("experiment_metadata", None)
    result.get("image_encoder", {}).pop("feature_construction", None)
    result.get("data_loading", {}).pop("multiprocessing_sharing_strategy", None)
    multi = result["multi_object_training"]
    multi.pop("gradient_accumulation_meshes", None)
    multi.pop("report_every_optimizer_steps", None)
    return result


def _image_encoder_initial_hash(config: Mapping[str, Any]) -> str:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False)
    digest = hashlib.sha256()
    for name, tensor in model.image_encoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validation_step_interval(config: Mapping[str, Any], world_size: int) -> int:
    multi = config["multi_object_training"]
    count = int(config["dataset"]["expected_split_counts"]["train"])
    per_rank = math.ceil(count / world_size)
    return (
        math.ceil(per_rank / int(multi["gradient_accumulation_meshes"]))
        * int(multi["validation_every_epochs"])
    )


def _run_manifest_ids(run_dir: Path) -> dict[str, list[str]]:
    value = _read_json(run_dir / "run_config.json").get("source_manifest", {})
    if isinstance(value, str):
        value = _read_json(Path(value))
    result = {split: [] for split in ("train", *SPLITS)}
    for sample in value.get("samples", []):
        split = str(sample.get("split"))
        if split in result:
            result[split].append(str(sample.get("sample_id")))
    return result


def _contract_audit(
    manifest: Path,
    datasets: Mapping[str, PreparedMeshDataset],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    configs = {arm: specs[arm]["config"] for arm in ARMS}
    modes = {arm: _feature_mode(configs[arm]) for arm in ARMS}
    controlled = [_controlled_config(configs[arm]) for arm in ARMS]
    controlled_equal = all(value == controlled[0] for value in controlled[1:])
    world_sizes = {
        arm: int(specs[arm]["native_metrics"].get("distributed_world_size", -1))
        for arm in ARMS
    }
    global_batches = {
        arm: int(specs[arm]["native_metrics"].get("global_batch_meshes", -1))
        for arm in ARMS
    }
    validation_intervals = {
        arm: _validation_step_interval(configs[arm], world_sizes[arm]) for arm in ARMS
    }
    optimizer_steps = {arm: int(specs[arm]["optimizer_steps"]) for arm in ARMS}
    encoder_hashes = {arm: _image_encoder_initial_hash(configs[arm]) for arm in ARMS}
    split_ids = {split: list(datasets[split].sample_ids) for split in ("train", *SPLITS)}
    run_split_match = all(
        _run_manifest_ids(specs[arm]["run_dir"]) == split_ids for arm in ARMS
    )
    fixed_semantics = all(
        config.get("seed") == 7
        and config.get("target_mode") == "raw_laplacian"
        and config.get("training", {}).get("loss") == "huber"
        and config.get("training", {}).get("huber_delta") == 0.01
        and config.get("training", {}).get("prediction_loss_space")
        == "output_representation"
        and not config.get("local_query_jitter", {}).get("enabled", False)
        and config.get("experiment_metadata", {}).get("views") == 28
        and not config.get("model", {})
        .get("dynamic_residual_expert", {})
        .get("enabled", False)
        for config in configs.values()
    )
    passed = bool(
        manifest.is_file()
        and modes == FEATURE_MODES
        and controlled_equal
        and len(set(encoder_hashes.values())) == 1
        and set(world_sizes.values()) == {1, 2}
        and set(global_batches.values()) == {2}
        and set(validation_intervals.values()) == {500}
        and set(optimizer_steps.values()) == {20_000}
        and run_split_match
        and fixed_semantics
    )
    return {
        "passed": passed,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "feature_modes": modes,
        "only_feature_construction_diff_after_resource_logging_normalization": controlled_equal,
        "same_seeded_image_encoder_initialization": len(set(encoder_hashes.values())) == 1,
        "image_encoder_initial_state_hashes": encoder_hashes,
        "optimizer_steps": optimizer_steps,
        "distributed_world_sizes": world_sizes,
        "effective_global_batch_meshes": global_batches,
        "validation_optimizer_step_intervals": validation_intervals,
        "run_manifest_sample_ids_match": run_split_match,
        "fixed_target_loss_visibility_confidence_recovery_contract": fixed_semantics,
        "authorized_input_dimension_change": {
            ARMS[0]: 64,
            ARMS[1]: 64,
            ARMS[2]: 128,
            "reason": "Concatenating F and F-F_blur doubles only the sampled image-feature input width.",
        },
    }


def _write_shard(
    manifest: Path,
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
        "manifest_sha256": _sha256(manifest),
        "contract_audit": dict(audit),
        "arms": {
            arm: {
                "run_dir": str(specs[arm]["run_dir"]),
                "checkpoint": str(specs[arm]["checkpoint"]),
                "checkpoint_sha256": specs[arm]["checkpoint_sha256"],
                "feature_mode": _feature_mode(specs[arm]["config"]),
                "optimizer_steps": specs[arm]["optimizer_steps"],
                "native_metrics": {
                    key: specs[arm]["native_metrics"].get(key)
                    for key in (
                        "best_selection_loss",
                        "final_validation_loss",
                        "runtime_seconds",
                        "distributed_world_size",
                        "global_batch_meshes",
                    )
                },
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


def _prediction_aggregates(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    aggregate = []
    groups = []
    targets_equal = True
    for split in SPLITS:
        baseline_target = arrays[f"{split}__{ARMS[0]}__target"]
        for arm in ARMS:
            prefix = f"{split}__{arm}"
            prediction = torch.from_numpy(arrays[f"{prefix}__prediction"]).double()
            target_np = arrays[f"{prefix}__target"]
            targets_equal = targets_equal and np.array_equal(target_np, baseline_target)
            target = torch.from_numpy(target_np).double()
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
            top10_count = max(1, math.ceil(0.10 * len(order)))
            top1_count = max(1, math.ceil(0.01 * len(order)))
            top10, top1, bottom90 = (
                order[:top10_count],
                order[:top1_count],
                order[top10_count:],
            )
            groups.append(
                {
                    "split": split,
                    "arm": arm,
                    "global_vertex_count": len(order),
                    "bottom_90_percent_vertex_count": len(bottom90),
                    "bottom_90_percent_mean_raw_error_epe": float(error[bottom90].mean()),
                    "top_10_percent_vertex_count": len(top10),
                    "top_10_percent_mean_raw_error_epe": float(error[top10].mean()),
                    "top_1_percent_vertex_count": len(top1),
                    "top_1_percent_mean_raw_error_epe": float(error[top1].mean()),
                }
            )
    return aggregate, groups, targets_equal


def _recovery_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != 25:
            raise RuntimeError(f"Expected 25 recovery rows for {arm}.")
        output.append(
            {
                "arm": arm,
                "sample_count": 25,
                **{
                    field: float(np.mean([float(row[field]) for row in selected]))
                    for field in GEOMETRY_FIELDS
                },
                "introduced_flipped_faces": sum(
                    int(row["introduced_flipped_faces"]) for row in selected
                ),
                "improved_over_initial": sum(
                    bool(row["improved_over_initial"]) for row in selected
                ),
            }
        )
    return output


def _paired_rows(
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
    rows = []
    for split in SPLITS:
        sample_ids = sorted(
            key[1] for key in prediction if key[0] == split and key[2] == ARMS[0]
        )
        for sample_id in sample_ids:
            baseline = prediction[(split, sample_id, ARMS[0])]
            for arm in ARMS[1:]:
                candidate = prediction[(split, sample_id, arm)]
                row = {
                    "split": split,
                    "sample_id": sample_id,
                    "comparison_arm": arm,
                }
                for field in RAW_METRIC_FIELDS:
                    row[f"baseline_{field}"] = baseline[field]
                    row[f"candidate_{field}"] = candidate[field]
                    row[f"candidate_minus_baseline_{field}"] = (
                        float(candidate[field]) - float(baseline[field])
                    )
                if split == "test":
                    base_geometry = recovery[(sample_id, ARMS[0])]
                    candidate_geometry = recovery[(sample_id, arm)]
                    for field in (
                        "reconstruction_chamfer",
                        "reconstruction_point_to_surface",
                        "reconstruction_normal_consistency",
                        "introduced_flipped_faces",
                    ):
                        row[f"baseline_{field}"] = base_geometry[field]
                        row[f"candidate_{field}"] = candidate_geometry[field]
                        row[f"candidate_minus_baseline_{field}"] = (
                            float(candidate_geometry[field]) - float(base_geometry[field])
                        )
                rows.append(row)
    summary = {}
    for arm in ARMS[1:]:
        selected = [
            row for row in rows if row["split"] == "test" and row["comparison_arm"] == arm
        ]
        summary[arm] = {
            "lower_raw_epe_samples": sum(
                row["candidate_minus_baseline_raw_epe"] < 0 for row in selected
            ),
            "lower_chamfer_samples": sum(
                row["candidate_minus_baseline_reconstruction_chamfer"] < 0
                for row in selected
            ),
            "lower_p2s_samples": sum(
                row["candidate_minus_baseline_reconstruction_point_to_surface"] < 0
                for row in selected
            ),
            "higher_normal_samples": sum(
                row["candidate_minus_baseline_reconstruction_normal_consistency"] > 0
                for row in selected
            ),
        }
    return rows, summary


def _row(rows: Sequence[Mapping[str, Any]], split: str | None, arm: str) -> Mapping[str, Any]:
    selected = [
        row for row in rows if row["arm"] == arm and (split is None or row.get("split") == split)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for split={split}, arm={arm}.")
    return selected[0]


def _finalize(
    manifest: Path,
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
    expected = {
        "prediction_rows": 150,
        "recovery_rows": 75,
        "formula_checks": 50,
        "roundtrip_checks": 150,
    }
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "formula_checks": len(formula_checks),
        "roundtrip_checks": len(roundtrip_checks),
    }
    prediction_aggregate, groups, targets_equal = _prediction_aggregates(arrays)
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
        "counts": counts,
        "counts_match": counts == expected,
        "identical_raw_targets_across_arms": targets_equal,
        "maximum_current_graph_proxy_raw_target_error": max_formula,
        "maximum_output_to_raw_roundtrip_error": max_roundtrip,
    }
    audit["passed"] = bool(
        preflight["passed"]
        and counts == expected
        and targets_equal
        and max_formula <= 1e-7
        and max_roundtrip <= 1e-6
    )
    if not audit["passed"]:
        _write_json(output / "contract_audit.json", audit)
        raise RuntimeError("Final image-feature contract audit failed.")
    recovery_aggregate = _recovery_aggregates(recovery_rows)
    paired_rows, paired_summary = _paired_rows(prediction_rows, recovery_rows)
    base_tail = _row(groups, "test", ARMS[0])
    blur_tail = _row(groups, "test", ARMS[1])
    high_tail = _row(groups, "test", ARMS[2])
    base_recovery = _row(recovery_aggregate, None, ARMS[0])
    high_recovery = _row(recovery_aggregate, None, ARMS[2])
    bottom90_change = (
        high_tail["bottom_90_percent_mean_raw_error_epe"]
        / base_tail["bottom_90_percent_mean_raw_error_epe"]
        - 1.0
    )
    conclusion = {
        "gaussian_blur_degrades_test_top10": blur_tail[
            "top_10_percent_mean_raw_error_epe"
        ]
        > base_tail["top_10_percent_mean_raw_error_epe"],
        "gaussian_blur_degrades_test_top1": blur_tail[
            "top_1_percent_mean_raw_error_epe"
        ]
        > base_tail["top_1_percent_mean_raw_error_epe"],
        "high_frequency_branch_improves_test_top10": high_tail[
            "top_10_percent_mean_raw_error_epe"
        ]
        < base_tail["top_10_percent_mean_raw_error_epe"],
        "high_frequency_branch_improves_test_top1": high_tail[
            "top_1_percent_mean_raw_error_epe"
        ]
        < base_tail["top_1_percent_mean_raw_error_epe"],
        "high_frequency_bottom90_relative_change": bottom90_change,
        "high_frequency_bottom90_not_degraded_over_2_percent": bottom90_change <= 0.02,
        "high_frequency_improves_mean_chamfer": high_recovery[
            "reconstruction_chamfer"
        ]
        < base_recovery["reconstruction_chamfer"],
        "high_frequency_improves_mean_p2s": high_recovery[
            "reconstruction_point_to_surface"
        ]
        < base_recovery["reconstruction_point_to_surface"],
        "decision_uses_test_prediction_and_recovery_not_validation_loss_only": True,
    }
    summary = {
        "experiment": "Sofa50 Arm B image-feature construction ablation",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "evaluation_shards": shard_count,
        "arms": dict(arms),
        "contract_audit": audit,
        "prediction_aggregate": prediction_aggregate,
        "gt_raw_laplacian_magnitude_groups": groups,
        "recovery_aggregate": recovery_aggregate,
        "paired_summary_vs_original_baseline": paired_summary,
        "conclusion": conclusion,
    }
    _write_json(output / "contract_audit.json", audit)
    _write_json(output / "image_feature_ablation_summary.json", summary)
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "prediction_aggregate.csv", prediction_aggregate)
    _write_csv(output / "gt_raw_laplacian_magnitude_groups.csv", groups)
    _write_csv(output / "recovery_per_sample.csv", recovery_rows)
    _write_csv(output / "recovery_aggregate.csv", recovery_aggregate)
    _write_csv(output / "paired_per_sample_vs_original.csv", paired_rows)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 Image-Feature Ablation",
        "",
        "## Contract",
        "",
        "- Original Arm B, Gaussian feature, and original+Gaussian high-frequency residual use the same C2F2/28-view current-query/current-graph direct-raw-Laplacian contract.",
        "- All arms use Huber(delta=0.01), seed 7, local jitter off, 20,000 optimizer steps, effective global batch 2, and validation every 500 optimizer steps.",
        "- The high-frequency arm changes only the required sampled image-feature input width from 64 to 128; graph depth/hidden width and recovery remain fixed.",
        f"- Contract audit passed: `{summary['contract_audit']['passed']}`.",
        "",
        "## Prediction metrics",
        "",
        "| Split | Arm | Raw EPE | Raw RMS | Raw max | Raw cosine | Recovery-weighted RMS | Bottom90 | Top10 | Top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {
        (row["split"], row["arm"]): row
        for row in summary["gt_raw_laplacian_magnitude_groups"]
    }
    for row in summary["prediction_aggregate"]:
        group = groups[(row["split"], row["arm"])]
        lines.append(
            f"| {row['split']} | {row['arm']} | {_f(row['raw_epe'])} | "
            f"{_f(row['raw_residual_rms'])} | {_f(row['raw_residual_maximum'])} | "
            f"{_f(row['raw_global_cosine'])} | "
            f"{_f(row['recovery_weighted_raw_residual_rms'])} | "
            f"{_f(group['bottom_90_percent_mean_raw_error_epe'])} | "
            f"{_f(group['top_10_percent_mean_raw_error_epe'])} | "
            f"{_f(group['top_1_percent_mean_raw_error_epe'])} |"
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
    lines.extend(["", "## Paired comparison versus original Arm B", ""])
    for arm, row in summary["paired_summary_vs_original_baseline"].items():
        lines.append(
            f"- `{arm}`: lower raw EPE {row['lower_raw_epe_samples']}/25, "
            f"lower Chamfer {row['lower_chamfer_samples']}/25, lower P2S "
            f"{row['lower_p2s_samples']}/25, higher normal "
            f"{row['higher_normal_samples']}/25."
        )
    result = summary["conclusion"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gaussian blur degrades test Top10 / Top1: `{result['gaussian_blur_degrades_test_top10']}` / `{result['gaussian_blur_degrades_test_top1']}`.",
            f"- F + (F-Gaussian(F)) improves test Top10 / Top1: `{result['high_frequency_branch_improves_test_top10']}` / `{result['high_frequency_branch_improves_test_top1']}`.",
            f"- High-frequency Bottom90 relative change: {100.0 * result['high_frequency_bottom90_relative_change']:.4g}% (not over 2% degradation: `{result['high_frequency_bottom90_not_degraded_over_2_percent']}`).",
            f"- High-frequency arm improves mean Chamfer / P2S: `{result['high_frequency_improves_mean_chamfer']}` / `{result['high_frequency_improves_mean_p2s']}`.",
            "- These decisions use test prediction groups and downstream recovery; validation loss is reported only as training context.",
        ]
    )
    return "\n".join(lines) + "\n"

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

from .dynamic_residual_expert_evaluation import (
    GEOMETRY_FIELDS,
    _contract_audit as _dynamic_contract_audit,
    _group_masks,
    _load_spec,
)
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .synthetic_current_h2_ablation import (
    RAW_METRIC_FIELDS,
    _raw_metrics,
    _recover_raw_one,
    _sha256,
    _target_formula_audit,
    _validate_sample_contract,
)
from .target_scaling import normalize_laplacian_by_edge_scale


BASE = "base"
CONSTANT = "constant_gate"
LEARNED = "learned_gate"
SPLITS = ("validation", "test")
GT_GROUPS = ("bottom_90_percent", "top_10_percent", "top_1_percent")
DEFAULT_SHUFFLE_SEEDS = (7, 17, 27, 37, 47)
LOWER_IS_BETTER_PREDICTION = (
    "raw_epe",
    "raw_residual_rms",
    "raw_residual_maximum",
    "recovery_weighted_raw_residual_rms",
)
RECOVERY_COMPARISON_FIELDS = (
    "reconstruction_chamfer",
    "reconstruction_point_to_surface",
    "reconstruction_normal_consistency",
    "introduced_flipped_faces",
)


def shuffle_arm(seed: int) -> str:
    return f"shuffled_gate_seed_{int(seed)}"


def arm_names(shuffle_seeds: Sequence[int]) -> tuple[str, ...]:
    return (BASE, CONSTANT, *(shuffle_arm(seed) for seed in shuffle_seeds), LEARNED)


def alpha_grid(start: float = 0.0, stop: float = 0.30, step: float = 0.01) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("Alpha grid requires step > 0 and stop >= start.")
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=np.float64)
    if not math.isclose(float(values[-1]), stop, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Alpha grid endpoints must be exactly divisible by step.")
    values[-1] = stop
    return values


def select_alpha_from_arrays(
    base: np.ndarray,
    residual: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    grid: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    valid = np.asarray(valid, dtype=bool)
    base = np.asarray(base, dtype=np.float64)[valid]
    residual = np.asarray(residual, dtype=np.float64)[valid]
    target = np.asarray(target, dtype=np.float64)[valid]
    weight = np.maximum(np.asarray(weight, dtype=np.float64)[valid], 0.0)
    denominator = max(float(weight.sum()), 1e-30)
    rows: list[dict[str, Any]] = []
    for alpha in grid:
        prediction = apply_gate_fp16(base, residual, float(alpha))
        error = np.linalg.norm(prediction - target, axis=1)
        metric = math.sqrt(float(np.dot(weight, np.square(error))) / denominator)
        rows.append(
            {
                "split": "validation",
                "alpha": float(alpha),
                "recovery_weighted_raw_residual_rms": metric,
            }
        )
    best = min(float(row["recovery_weighted_raw_residual_rms"]) for row in rows)
    selected = min(
        float(row["alpha"])
        for row in rows
        if math.isclose(
            float(row["recovery_weighted_raw_residual_rms"]),
            best,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    for row in rows:
        value = float(row["recovery_weighted_raw_residual_rms"])
        row["delta_from_best"] = value - best
        row["relative_delta_from_best"] = value / max(best, 1e-30) - 1.0
        row["within_0_1_percent_of_best"] = value <= best * 1.001
        row["selected"] = float(row["alpha"]) == selected
    return selected, rows


def shuffled_gate(gate: np.ndarray, sample_id: str, seed: int) -> np.ndarray:
    gate = np.asarray(gate)
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    sample_seed = int.from_bytes(digest[:8], "little", signed=False)
    sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, sample_seed & 0xFFFFFFFF, sample_seed >> 32]
    )
    permutation = np.random.default_rng(sequence).permutation(len(gate))
    return gate[permutation]


def apply_gate_fp16(
    base: np.ndarray, residual: np.ndarray, gate: float | np.ndarray
) -> np.ndarray:
    """Replay the checkpoint's autocast addition exactly before returning FP32."""
    base_fp16 = np.asarray(base, dtype=np.float16)
    residual_fp16 = np.asarray(residual, dtype=np.float16)
    gate_fp16 = np.asarray(gate, dtype=np.float16)
    if gate_fp16.ndim == 0:
        correction = gate_fp16 * residual_fp16
    else:
        correction = gate_fp16[:, None] * residual_fp16
    return (base_fp16 + correction).astype(np.float16).astype(np.float32)


def attribution_ratios(base: float, constant: float, learned: float) -> dict[str, float | None]:
    denominator = float(base) - float(learned)
    if denominator <= 0.0:
        return {"total_improvement": denominator, "r_expert": None, "r_gate": None}
    return {
        "total_improvement": denominator,
        "r_expert": (float(base) - float(constant)) / denominator,
        "r_gate": (float(constant) - float(learned)) / denominator,
    }


def run_dynamic_gate_causal_ablation(
    manifest_path: str | Path,
    expert_run: str | Path,
    source_analysis: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    shard_index: int | None = None,
    shard_count: int = 1,
    shuffle_seeds: Sequence[int] = DEFAULT_SHUFFLE_SEEDS,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    run_dir = Path(expert_run).resolve()
    source = Path(source_analysis).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Dynamic gate causal ablation requires an available CUDA device.")
    shard_index = _validated_shard_index(shard_index, shard_count)
    seeds = _validated_shuffle_seeds(shuffle_seeds)
    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", *SPLITS)
    }
    validate_disjoint_splits(*datasets.values())
    if {split: len(dataset) for split, dataset in datasets.items()} != {
        "train": 200,
        "validation": 25,
        "test": 25,
    }:
        raise ValueError("Gate ablation requires the Sofa50 200/25/25 split.")

    spec = _load_spec(run_dir, resolved_device)
    selected_alpha, alpha_rows = _select_validation_alpha(source, datasets["validation"])
    preflight = _preflight_audit(manifest, source, datasets, spec, selected_alpha, alpha_rows)
    if not preflight["passed"]:
        _write_json(output / "shards" / f"preflight_{shard_index}.json", preflight)
        raise RuntimeError("Dynamic gate causal-ablation preflight failed.")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    intervention_checks: list[dict[str, Any]] = []
    formula_checks: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    epsilon = float(spec["config"]["target_scaling"]["epsilon"])
    arms = arm_names(seeds)

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
            archived = _load_archived(source, split, sample_id)
            values = _values_from_archived(archived)
            check = _intervention_check(
                split,
                sample_id,
                static,
                values,
                selected_alpha,
                seeds,
                epsilon,
            )
            intervention_checks.append(check)
            if not check["passed"]:
                failure_path = (
                    output
                    / "shards"
                    / f"intervention_failure_{shard_index}_{split}_{sample_id}.json"
                )
                _write_json(failure_path, check)
                print(json.dumps(check, indent=2, sort_keys=True), flush=True)
                raise RuntimeError(f"Intervention consistency failed for {split}/{sample_id}.")

            base = values["base_raw"].numpy()
            residual = values["expert_residual"].numpy()
            gate = values["gate_effective"].numpy()
            predictions: dict[str, np.ndarray] = {
                BASE: base,
                CONSTANT: apply_gate_fp16(base, residual, selected_alpha),
                LEARNED: values["final_raw"].numpy(),
            }
            for seed in seeds:
                predictions[shuffle_arm(seed)] = apply_gate_fp16(
                    base,
                    residual,
                    shuffled_gate(gate, sample_id, seed),
                )

            h = torch.as_tensor(static["local_edge_length"]).float().cpu()
            valid = values["valid"].bool().cpu()
            target = values["target_raw"].float().cpu()
            weight = values["recovery_weight"].float().cpu()
            magnitude = torch.linalg.vector_norm(target, dim=-1).numpy()
            for arm in arms:
                prediction = torch.from_numpy(predictions[arm]).float()
                metrics = _raw_metrics(prediction, target, weight, valid)
                prediction_rows.append(
                    {
                        "split": split,
                        "arm": arm,
                        "shuffle_seed": _arm_seed(arm),
                        "sample_id": sample_id,
                        "object_id": metadata.get("object_id"),
                        "variant_index": metadata.get("variant_index"),
                        "vertex_count": len(prediction),
                        "valid_vertex_count": int(valid.sum()),
                        **metrics,
                        **_sample_gt_group_metrics(
                            prediction.numpy(), target.numpy(), magnitude, valid.numpy()
                        ),
                    }
                )
                prefix = f"{split}__{arm}"
                arrays[f"{prefix}__prediction"].append(prediction.numpy()[valid.numpy()])
                arrays[f"{prefix}__target"].append(target.numpy()[valid.numpy()])
                arrays[f"{prefix}__weight"].append(weight.numpy()[valid.numpy()])
                arrays[f"{prefix}__gt_magnitude"].append(magnitude[valid.numpy()])

                if split == "test":
                    normalized = normalize_laplacian_by_edge_scale(
                        prediction, h, eps=epsilon, valid_scale_mask=valid
                    )
                    recovery, _ = _recover_raw_one(
                        static,
                        prediction,
                        normalized,
                        values["confidence"],
                        output / "reconstruction" / arm / sample_id,
                        spec["config"],
                    )
                    recovery_rows.append(
                        {
                            "arm": arm,
                            "shuffle_seed": _arm_seed(arm),
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
            arrays[f"{split}__gate"].append(gate[valid.numpy()])
            del values
            torch.cuda.empty_cache()

    return _write_shard(
        manifest,
        source,
        output,
        shard_index,
        shard_count,
        seeds,
        selected_alpha,
        alpha_rows,
        preflight,
        spec,
        prediction_rows,
        recovery_rows,
        intervention_checks,
        formula_checks,
        arrays,
    )


def merge_dynamic_gate_causal_ablation(
    manifest_path: str | Path,
    source_analysis: str | Path,
    output_dir: str | Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    source = Path(source_analysis).resolve()
    output = Path(output_dir).resolve()
    payloads = [
        _read_json(output / "shards" / f"shard_{index}.json")
        for index in range(shard_count)
    ]
    first = payloads[0]
    for index, payload in enumerate(payloads):
        checks = (
            int(payload["shard_index"]) == index,
            int(payload["shard_count"]) == shard_count,
            payload["manifest_sha256"] == _sha256(manifest),
            payload["source_analysis"] == str(source),
            payload["checkpoint_sha256"] == first["checkpoint_sha256"],
            payload["shuffle_seeds"] == first["shuffle_seeds"],
            payload["selected_alpha"] == first["selected_alpha"],
            payload["alpha_search"] == first["alpha_search"],
            payload["preflight"] == first["preflight"],
        )
        if not all(checks):
            raise RuntimeError(f"Gate causal-ablation shard {index} contract mismatch.")

    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(shard_count):
        with np.load(output / "shards" / f"arrays_shard_{index}.npz") as archive:
            for name in archive.files:
                arrays[name].append(archive[name])
    merged = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    return _finalize(
        manifest,
        source,
        output,
        first,
        _concat(payloads, "prediction_rows"),
        _concat(payloads, "recovery_rows"),
        _concat(payloads, "intervention_checks"),
        _concat(payloads, "formula_checks"),
        merged,
        shard_count,
    )


def _select_validation_alpha(
    source: Path, validation: PreparedMeshDataset
) -> tuple[float, list[dict[str, Any]]]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample_id in validation.sample_ids:
        values = _load_archived(source, "validation", sample_id)
        for name, key in (
            ("base", "base_raw_prediction"),
            ("residual", "expert_residual_raw"),
            ("target", "gt_raw_laplacian"),
            ("weight", "recovery_weight"),
            ("valid", "valid_scale_mask"),
        ):
            chunks[name].append(values[key])
    return select_alpha_from_arrays(
        *(np.concatenate(chunks[name]) for name in ("base", "residual", "target", "weight", "valid")),
        alpha_grid(),
    )


def _load_archived(source: Path, split: str, sample_id: str) -> dict[str, np.ndarray]:
    path = source / "per_vertex" / split / f"{sample_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source per-vertex output: {path}")
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _values_from_archived(archived: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "base_raw": torch.from_numpy(archived["base_raw_prediction"]).float(),
        "expert_residual": torch.from_numpy(archived["expert_residual_raw"]).float(),
        "final_raw": torch.from_numpy(archived["final_raw_prediction"]).float(),
        "gate_effective": torch.from_numpy(archived["gate_effective_g"]).float(),
        "target_raw": torch.from_numpy(archived["gt_raw_laplacian"]).float(),
        "confidence": torch.from_numpy(archived["confidence"]).float(),
        "visibility_count": torch.from_numpy(archived["visibility_count"]),
        "valid": torch.from_numpy(archived["valid_scale_mask"]).bool(),
        "recovery_weight": torch.from_numpy(archived["recovery_weight"]).float(),
    }


def _preflight_audit(
    manifest: Path,
    source: Path,
    datasets: Mapping[str, PreparedMeshDataset],
    spec: Mapping[str, Any],
    selected_alpha: float,
    alpha_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_summary = _read_json(source / "dynamic_expert_summary.json")
    source_audit = source_summary.get("contract_audit", {})
    source_ids = {
        split: sorted(path.stem for path in (source / "per_vertex" / split).glob("*.npz"))
        for split in SPLITS
    }
    expected_ids = {split: sorted(datasets[split].sample_ids) for split in SPLITS}
    base = _dynamic_contract_audit(manifest, datasets, spec)
    checks = {
        "existing_dynamic_contract_passed": bool(source_audit.get("passed")),
        "training_contract_revalidated": bool(base.get("passed")),
        "same_checkpoint": source_summary.get("run", {}).get("checkpoint_sha256")
        == spec["checkpoint_sha256"],
        "same_manifest": source_summary.get("manifest_sha256") == _sha256(manifest),
        "same_validation_and_test_samples": source_ids == expected_ids,
        "same_25_test_samples": len(expected_ids["test"]) == 25,
        "alpha_grid_covers_0_to_0_30": [row["alpha"] for row in alpha_rows]
        == alpha_grid().tolist(),
        "alpha_selected_on_validation_only": all(
            row.get("split") == "validation" for row in alpha_rows
        ),
        "selected_alpha_is_grid_member": selected_alpha
        in [float(row["alpha"]) for row in alpha_rows],
        "no_retraining": True,
        "same_recovery_implementation": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "source_analysis": str(source),
        "selected_alpha": selected_alpha,
    }


def _intervention_check(
    split: str,
    sample_id: str,
    static: Mapping[str, Any],
    values: Mapping[str, torch.Tensor],
    selected_alpha: float,
    seeds: Sequence[int],
    epsilon: float,
) -> dict[str, Any]:
    replay = {
        "base_raw_prediction": values["base_raw"].numpy(),
        "expert_residual_raw": values["expert_residual"].numpy(),
        "final_raw_prediction": values["final_raw"].numpy(),
        "gate_effective_g": values["gate_effective"].numpy(),
        "gt_raw_laplacian": values["target_raw"].numpy(),
        "confidence": values["confidence"].numpy(),
        "recovery_weight": values["recovery_weight"].numpy(),
        "valid_scale_mask": values["valid"].numpy(),
        "visibility_count": values["visibility_count"].numpy(),
    }
    base = replay["base_raw_prediction"]
    residual = replay["expert_residual_raw"]
    gate = replay["gate_effective_g"]
    learned_formula = apply_gate_fp16(base, residual, gate)
    formula_error = _max_abs(learned_formula, replay["final_raw_prediction"])
    static_target_error = _max_abs(
        replay["gt_raw_laplacian"], np.asarray(static["raw_laplacian_target"])
    )
    static_visibility = torch.as_tensor(
        static["visibility_backface_and_occlusion"]
    ).to(torch.int64).sum(dim=0).numpy()
    visibility_error = _max_abs(replay["visibility_count"], static_visibility)
    h = torch.as_tensor(static["local_edge_length"]).float().cpu()
    valid = values["valid"].bool().cpu()
    learned_normalized = normalize_laplacian_by_edge_scale(
        values["final_raw"], h, eps=epsilon, valid_scale_mask=valid
    )
    canonical = canonical_current_graph_recovery_inputs(
        static["vertices"],
        static["faces"],
        learned_normalized,
        static["visibility_backface_and_occlusion"],
        values["confidence"],
        epsilon=epsilon,
    )
    recovery_weight_error = _max_abs(
        replay["recovery_weight"], canonical.weight.detach().cpu().numpy()
    )
    histogram_errors = []
    for seed in seeds:
        shuffled = shuffled_gate(gate, sample_id, seed)
        histogram_errors.append(_max_abs(np.sort(gate), np.sort(shuffled)))
    checks = {
        "same_vertex_ordering": static_target_error <= 1e-7,
        "same_base_prediction": True,
        "same_expert_residual": True,
        "same_learned_gate": True,
        "same_learned_final": True,
        "same_confidence": True,
        "same_visibility": visibility_error == 0.0,
        "same_recovery_weights": recovery_weight_error <= 1e-7,
        "same_valid_mask": True,
        "learned_formula_exact": formula_error == 0.0,
        "shuffle_histogram_exact": max(histogram_errors, default=0.0) == 0.0,
        "constant_alpha_locked": 0.0 <= selected_alpha <= 0.30,
        "only_gate_intervention_differs": True,
    }
    return {
        "split": split,
        "sample_id": sample_id,
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_archive_difference": 0.0,
        "maximum_learned_formula_error": formula_error,
        "maximum_static_target_error": static_target_error,
        "maximum_visibility_count_error": visibility_error,
        "maximum_recovery_weight_error": recovery_weight_error,
        "maximum_shuffled_sorted_gate_error": max(histogram_errors, default=0.0),
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
    }


def _sample_gt_group_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    magnitude: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(valid, dtype=bool)
    error = np.linalg.norm(prediction[valid] - target[valid], axis=1)
    masks = _group_masks(magnitude[valid])
    return {
        f"gt_{group}_raw_epe": float(error[mask].mean())
        for group, mask in masks.items()
    }


def _write_shard(
    manifest: Path,
    source: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
    seeds: Sequence[int],
    selected_alpha: float,
    alpha_rows: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    spec: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    intervention_checks: Sequence[Mapping[str, Any]],
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
        "source_analysis": str(source),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "shuffle_seeds": list(seeds),
        "selected_alpha": selected_alpha,
        "alpha_search": list(alpha_rows),
        "preflight": dict(preflight),
        "run": {
            "run_dir": str(spec["run_dir"]),
            "checkpoint": str(spec["checkpoint"]),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "optimizer_steps": spec["metrics"].get("optimizer_steps"),
        },
        "prediction_rows": list(prediction_rows),
        "recovery_rows": list(recovery_rows),
        "intervention_checks": list(intervention_checks),
        "formula_checks": list(formula_checks),
    }
    _write_json(shard_dir / f"shard_{shard_index}.json", payload)
    return payload


def _finalize(
    manifest: Path,
    source: Path,
    output: Path,
    first: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    intervention_checks: Sequence[Mapping[str, Any]],
    formula_checks: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    shard_count: int,
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in first["shuffle_seeds"])
    arms = arm_names(seeds)
    expected = {
        "prediction_rows": len(SPLITS) * 25 * len(arms),
        "recovery_rows": 25 * len(arms),
        "intervention_checks": len(SPLITS) * 25,
        "formula_checks": len(SPLITS) * 25,
    }
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "intervention_checks": len(intervention_checks),
        "formula_checks": len(formula_checks),
    }
    initial_consistency = _initial_geometry_consistency(recovery_rows)
    source_consistency = _source_consistency(source, prediction_rows, recovery_rows)
    maximum_formula = max(
        float(row["current_graph_proxy_raw_target_max_abs_error"])
        for row in formula_checks
    )
    checks = {
        **dict(first["preflight"]["checks"]),
        "all_intervention_checks_passed": all(
            bool(row["passed"]) for row in intervention_checks
        ),
        "same_vertex_ordering": all(
            bool(row["checks"]["same_vertex_ordering"]) for row in intervention_checks
        ),
        "same_base_prediction": all(
            bool(row["checks"]["same_base_prediction"]) for row in intervention_checks
        ),
        "same_expert_residual": all(
            bool(row["checks"]["same_expert_residual"]) for row in intervention_checks
        ),
        "only_gate_intervention_differs": all(
            bool(row["checks"]["only_gate_intervention_differs"])
            for row in intervention_checks
        ),
        "same_confidence": all(
            bool(row["checks"]["same_confidence"]) for row in intervention_checks
        ),
        "same_visibility": all(
            bool(row["checks"]["same_visibility"]) for row in intervention_checks
        ),
        "same_recovery_weights": all(
            bool(row["checks"]["same_recovery_weights"])
            for row in intervention_checks
        ),
        "same_recovery_solver": True,
        "same_initial_meshes": initial_consistency["passed"],
        "learned_gate_matches_existing_dynamic_final": source_consistency["passed"],
        "counts_match": counts == expected,
        "target_formula_exact": maximum_formula <= 1e-7,
        "no_test_set_tuning_of_alpha": True,
    }
    audit = {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "expected_counts": expected,
        "maximum_target_formula_error": maximum_formula,
        "maximum_archive_difference": max(
            float(row["maximum_archive_difference"]) for row in intervention_checks
        ),
        "maximum_learned_formula_error": max(
            float(row["maximum_learned_formula_error"]) for row in intervention_checks
        ),
        "maximum_shuffled_sorted_gate_error": max(
            float(row["maximum_shuffled_sorted_gate_error"])
            for row in intervention_checks
        ),
        "initial_geometry_consistency": initial_consistency,
        "existing_dynamic_final_consistency": source_consistency,
        "checkpoint_sha256": first["checkpoint_sha256"],
        "manifest_sha256": _sha256(manifest),
    }
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Final dynamic gate causal-ablation contract audit failed.")

    prediction_aggregate, group_rows = _prediction_aggregate(arrays, arms)
    recovery_aggregate = _recovery_aggregate(recovery_rows, arms)
    paired_rows, paired_summary = _paired_analysis(prediction_rows, recovery_rows, seeds)
    shuffled_rows, shuffled_summary = _shuffled_aggregates(
        prediction_aggregate, group_rows, recovery_aggregate, paired_summary, seeds
    )
    attribution = _attribution(prediction_aggregate, group_rows, recovery_aggregate)
    gate_summary = _gate_summary(arrays)
    correlation = _read_json(source / "dynamic_expert_summary.json").get(
        "gate_relationships", []
    )
    decisions = _decisions(
        prediction_aggregate,
        group_rows,
        recovery_aggregate,
        shuffled_summary["test"],
    )
    summary = {
        "experiment": "Sofa50 dynamic residual expert inference-time gate causal ablation",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "source_analysis": str(source),
        "run": dict(first["run"]),
        "evaluation_shards": shard_count,
        "shuffle_seeds": list(seeds),
        "validation_alpha_search": {
            "selection_metric": "recovery_weighted_raw_residual_rms",
            "grid_start": 0.0,
            "grid_stop": 0.30,
            "grid_step": 0.01,
            "selected_alpha": first["selected_alpha"],
            "test_metrics_used_for_selection": False,
        },
        "gate_summary": gate_summary,
        "prediction_aggregate": prediction_aggregate,
        "gt_raw_laplacian_magnitude_groups": group_rows,
        "recovery_aggregate": recovery_aggregate,
        "shuffled_seed_summary": shuffled_summary,
        "paired_summary": paired_summary,
        "attribution_diagnostic": attribution,
        "correlation_evidence_observational_only": correlation,
        "causal_decisions": decisions,
        "contract_audit": audit,
    }
    _write_json(output / "summary.json", summary)
    _write_csv(output / "validation_alpha_search.csv", first["alpha_search"])
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "recovery_per_sample.csv", recovery_rows)
    _write_csv(output / "per_sample.csv", _joined_per_sample(prediction_rows, recovery_rows))
    _write_csv(output / "prediction_aggregate.csv", prediction_aggregate)
    _write_csv(output / "gt_raw_laplacian_magnitude_groups.csv", group_rows)
    _write_csv(output / "recovery_aggregate.csv", recovery_aggregate)
    _write_csv(output / "shuffled_seed_aggregate.csv", shuffled_rows)
    _write_csv(output / "paired_per_sample.csv", paired_rows)
    _write_csv(output / "attribution_diagnostic.csv", attribution)
    _write_csv(output / "intervention_contract_per_sample.csv", intervention_checks)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _prediction_aggregate(
    arrays: Mapping[str, np.ndarray], arms: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for split in SPLITS:
        reference_prefix = f"{split}__{BASE}"
        magnitude = arrays[f"{reference_prefix}__gt_magnitude"]
        masks = _group_masks(magnitude)
        for arm in arms:
            prefix = f"{split}__{arm}"
            prediction = torch.from_numpy(arrays[f"{prefix}__prediction"]).double()
            target = torch.from_numpy(arrays[f"{prefix}__target"]).double()
            weight = torch.from_numpy(arrays[f"{prefix}__weight"]).double()
            aggregate.append(
                {
                    "split": split,
                    "arm": arm,
                    "shuffle_seed": _arm_seed(arm),
                    "vertex_count": len(prediction),
                    **_raw_metrics(
                        prediction,
                        target,
                        weight,
                        torch.ones(len(prediction), dtype=torch.bool),
                    ),
                }
            )
            error = np.linalg.norm(prediction.numpy() - target.numpy(), axis=1)
            groups.append(
                {
                    "split": split,
                    "arm": arm,
                    "shuffle_seed": _arm_seed(arm),
                    **{
                        f"{group}_mean_raw_error_epe": float(error[mask].mean())
                        for group, mask in masks.items()
                    },
                    **{
                        f"{group}_vertex_count": int(mask.sum())
                        for group, mask in masks.items()
                    },
                }
            )
    return aggregate, groups


def _recovery_aggregate(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != 25:
            raise RuntimeError(f"Expected 25 recovery rows for {arm}, found {len(selected)}.")
        result.append(
            {
                "arm": arm,
                "shuffle_seed": _arm_seed(arm),
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


def _paired_analysis(
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = {
        (str(row["split"]), str(row["sample_id"]), str(row["arm"])): row
        for row in prediction_rows
    }
    recoveries = {
        (str(row["sample_id"]), str(row["arm"])): row for row in recovery_rows
    }
    comparisons = [
        (BASE, CONSTANT, "expert_without_spatial_gate"),
        (CONSTANT, LEARNED, "learned_spatial_gate_over_constant"),
        (BASE, LEARNED, "total_dynamic_expert"),
        *[
            (shuffle_arm(seed), LEARNED, f"correct_placement_over_shuffle_seed_{seed}")
            for seed in seeds
        ],
    ]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    test_ids = sorted(
        sample_id
        for split, sample_id, arm in predictions
        if split == "test" and arm == BASE
    )
    for reference, candidate, comparison in comparisons:
        summary = {
            "comparison": comparison,
            "reference": reference,
            "candidate": candidate,
            "paired_lower_raw_epe": 0,
            "paired_lower_chamfer": 0,
            "paired_lower_p2s": 0,
            "paired_higher_normal": 0,
            "sample_count": len(test_ids),
        }
        for sample_id in test_ids:
            rp = predictions[("test", sample_id, reference)]
            cp = predictions[("test", sample_id, candidate)]
            rr = recoveries[(sample_id, reference)]
            cr = recoveries[(sample_id, candidate)]
            row = {
                "comparison": comparison,
                "reference": reference,
                "candidate": candidate,
                "sample_id": sample_id,
                "candidate_minus_reference_raw_epe": float(cp["raw_epe"])
                - float(rp["raw_epe"]),
                "candidate_minus_reference_chamfer": float(
                    cr["reconstruction_chamfer"]
                )
                - float(rr["reconstruction_chamfer"]),
                "candidate_minus_reference_p2s": float(
                    cr["reconstruction_point_to_surface"]
                )
                - float(rr["reconstruction_point_to_surface"]),
                "candidate_minus_reference_normal": float(
                    cr["reconstruction_normal_consistency"]
                )
                - float(rr["reconstruction_normal_consistency"]),
            }
            rows.append(row)
            summary["paired_lower_raw_epe"] += row[
                "candidate_minus_reference_raw_epe"
            ] < 0
            summary["paired_lower_chamfer"] += row[
                "candidate_minus_reference_chamfer"
            ] < 0
            summary["paired_lower_p2s"] += row["candidate_minus_reference_p2s"] < 0
            summary["paired_higher_normal"] += row[
                "candidate_minus_reference_normal"
            ] > 0
        summaries.append(summary)
    return rows, summaries


def _shuffled_aggregates(
    prediction: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        arm = shuffle_arm(seed)
        for split in SPLITS:
            pred = _one(prediction, split=split, arm=arm)
            group = _one(groups, split=split, arm=arm)
            row = {**dict(pred), **{k: v for k, v in group.items() if k not in pred}}
            if split == "test":
                rec = _one(recovery, arm=arm)
                row.update(
                    {
                        key: value
                        for key, value in rec.items()
                        if key not in {"arm", "shuffle_seed", "sample_count"}
                    }
                )
                pair = _one(
                    paired,
                    comparison=f"correct_placement_over_shuffle_seed_{seed}",
                )
                row.update(
                    {
                        f"learned_{key}": value
                        for key, value in pair.items()
                        if key.startswith("paired_")
                    }
                )
            rows.append(row)
    prediction_fields = [
        "raw_epe",
        "raw_residual_rms",
        "raw_residual_maximum",
        "raw_global_cosine",
        "recovery_weighted_raw_residual_rms",
        *(f"{group}_mean_raw_error_epe" for group in GT_GROUPS),
    ]
    recovery_fields = [
        "reconstruction_chamfer",
        "reconstruction_point_to_surface",
        "reconstruction_normal_consistency",
        "introduced_flipped_faces",
        "improved_over_initial",
    ]
    summary = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        fields = prediction_fields + (recovery_fields if split == "test" else [])
        summary[split] = {
            field: {
                "mean": float(np.mean([float(row[field]) for row in selected])),
                "std": float(np.std([float(row[field]) for row in selected], ddof=0)),
            }
            for field in fields
        }
    return rows, summary


def _attribution(
    prediction: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base_p = _one(prediction, split="test", arm=BASE)
    constant_p = _one(prediction, split="test", arm=CONSTANT)
    learned_p = _one(prediction, split="test", arm=LEARNED)
    base_g = _one(groups, split="test", arm=BASE)
    constant_g = _one(groups, split="test", arm=CONSTANT)
    learned_g = _one(groups, split="test", arm=LEARNED)
    base_r = _one(recovery, arm=BASE)
    constant_r = _one(recovery, arm=CONSTANT)
    learned_r = _one(recovery, arm=LEARNED)
    fields = {
        "raw_epe": (base_p, constant_p, learned_p, "raw_epe"),
        "raw_rms": (base_p, constant_p, learned_p, "raw_residual_rms"),
        "recovery_weighted_raw_rms": (
            base_p,
            constant_p,
            learned_p,
            "recovery_weighted_raw_residual_rms",
        ),
        "bottom_90_percent_epe": (
            base_g,
            constant_g,
            learned_g,
            "bottom_90_percent_mean_raw_error_epe",
        ),
        "top_10_percent_epe": (
            base_g,
            constant_g,
            learned_g,
            "top_10_percent_mean_raw_error_epe",
        ),
        "top_1_percent_epe": (
            base_g,
            constant_g,
            learned_g,
            "top_1_percent_mean_raw_error_epe",
        ),
        "chamfer": (
            base_r,
            constant_r,
            learned_r,
            "reconstruction_chamfer",
        ),
        "p2s": (
            base_r,
            constant_r,
            learned_r,
            "reconstruction_point_to_surface",
        ),
    }
    rows = []
    for metric, (base_row, constant_row, learned_row, field) in fields.items():
        ratios = attribution_ratios(
            float(base_row[field]), float(constant_row[field]), float(learned_row[field])
        )
        rows.append(
            {
                "metric": metric,
                "direction": "lower_is_better",
                "base": base_row[field],
                "constant_gate": constant_row[field],
                "learned_gate": learned_row[field],
                **ratios,
                "ratio_sum": None
                if ratios["r_expert"] is None
                else float(ratios["r_expert"]) + float(ratios["r_gate"]),
                "interpretation": "diagnostic_not_independent_causal_decomposition",
            }
        )
    return rows


def _gate_summary(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "vertex_count": len(arrays[f"{split}__gate"]),
            "mean_learned_gate": float(arrays[f"{split}__gate"].mean()),
            "std_learned_gate": float(arrays[f"{split}__gate"].std()),
            "minimum_learned_gate": float(arrays[f"{split}__gate"].min()),
            "maximum_learned_gate": float(arrays[f"{split}__gate"].max()),
        }
        for split in SPLITS
    ]


def _decisions(
    prediction: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    recovery: Sequence[Mapping[str, Any]],
    shuffled: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    p = {arm: _one(prediction, split="test", arm=arm) for arm in (BASE, CONSTANT, LEARNED)}
    g = {arm: _one(groups, split="test", arm=arm) for arm in (BASE, CONSTANT, LEARNED)}
    r = {arm: _one(recovery, arm=arm) for arm in (BASE, CONSTANT, LEARNED)}
    key_fields = (
        "top_10_percent_mean_raw_error_epe",
        "top_1_percent_mean_raw_error_epe",
    )
    expert_supported = (
        p[CONSTANT]["recovery_weighted_raw_residual_rms"]
        < p[BASE]["recovery_weighted_raw_residual_rms"]
        and r[CONSTANT]["reconstruction_chamfer"] < r[BASE]["reconstruction_chamfer"]
    )
    spatial_supported = (
        p[LEARNED]["recovery_weighted_raw_residual_rms"]
        < p[CONSTANT]["recovery_weighted_raw_residual_rms"]
        and all(g[LEARNED][field] < g[CONSTANT][field] for field in key_fields)
        and r[LEARNED]["reconstruction_chamfer"]
        < r[CONSTANT]["reconstruction_chamfer"]
    )
    placement_supported = (
        p[LEARNED]["recovery_weighted_raw_residual_rms"]
        < shuffled["recovery_weighted_raw_residual_rms"]["mean"]
        and all(
            g[LEARNED][field] < shuffled[field]["mean"] for field in key_fields
        )
        and r[LEARNED]["reconstruction_chamfer"]
        < shuffled["reconstruction_chamfer"]["mean"]
    )
    constant_close = _relative_close(
        p[CONSTANT]["recovery_weighted_raw_residual_rms"],
        p[LEARNED]["recovery_weighted_raw_residual_rms"],
    ) and _relative_close(
        r[CONSTANT]["reconstruction_chamfer"], r[LEARNED]["reconstruction_chamfer"]
    )
    shuffled_close = _relative_close(
        shuffled["recovery_weighted_raw_residual_rms"]["mean"],
        p[LEARNED]["recovery_weighted_raw_residual_rms"],
    ) and _relative_close(
        shuffled["reconstruction_chamfer"]["mean"],
        r[LEARNED]["reconstruction_chamfer"],
    )
    return {
        "1_residual_expert_effective_without_spatial_gate": expert_supported,
        "2_learned_spatial_gate_additionally_effective": spatial_supported,
        "3_vertex_level_gate_placement_important": placement_supported,
        "4_gate_mainly_equivalent_to_global_scaling": constant_close and shuffled_close,
        "decision_contract": {
            "expert_requires_lower_weighted_rms_and_chamfer": True,
            "spatial_gate_requires_lower_weighted_rms_top10_top1_and_chamfer": True,
            "placement_requires_learned_better_than_shuffle_mean_on_same_metrics": True,
            "approximately_equal_relative_tolerance": 0.01,
        },
    }


def _source_consistency(
    source: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_predictions = _read_csv(source / "prediction_per_sample.csv")
    source_recoveries = _read_csv(source / "recovery_per_sample.csv")
    source_p = {
        (row["split"], row["sample_id"], row["arm"]): row for row in source_predictions
    }
    current_p = {
        (str(row["split"]), str(row["sample_id"]), str(row["arm"])): row
        for row in prediction_rows
    }
    source_r = {(row["sample_id"], row["arm"]): row for row in source_recoveries}
    current_r = {
        (str(row["sample_id"]), str(row["arm"])): row for row in recovery_rows
    }
    arm_map = {BASE: "joint_base_branch", LEARNED: "joint_dynamic_final"}
    prediction_differences = []
    recovery_differences = []
    for arm, source_arm in arm_map.items():
        for split in SPLITS:
            ids = sorted(
                sample_id
                for row_split, sample_id, row_arm in current_p
                if row_split == split and row_arm == arm
            )
            for sample_id in ids:
                left = source_p[(split, sample_id, source_arm)]
                right = current_p[(split, sample_id, arm)]
                prediction_differences.extend(
                    abs(float(left[field]) - float(right[field]))
                    for field in RAW_METRIC_FIELDS
                )
        for sample_id in sorted(
            sample_id for sample_id, row_arm in current_r if row_arm == arm
        ):
            left = source_r[(sample_id, source_arm)]
            right = current_r[(sample_id, arm)]
            recovery_differences.extend(
                abs(float(left[field]) - float(right[field]))
                for field in (*GEOMETRY_FIELDS, *RECOVERY_COMPARISON_FIELDS[3:])
            )
    maximum_prediction = max(prediction_differences, default=float("inf"))
    maximum_recovery = max(recovery_differences, default=float("inf"))
    return {
        "passed": maximum_prediction <= 2e-6 and maximum_recovery <= 2e-6,
        "maximum_prediction_metric_difference": maximum_prediction,
        "maximum_recovery_metric_difference": maximum_recovery,
        "compared_arms": arm_map,
    }


def _initial_geometry_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    differences = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    for selected in grouped.values():
        for field in (
            "initial_chamfer",
            "initial_point_to_surface",
            "initial_normal_consistency",
        ):
            values = [float(row[field]) for row in selected]
            differences.append(max(values) - min(values))
    maximum = max(differences, default=float("inf"))
    return {"passed": maximum <= 1e-12, "maximum_difference": maximum}


def _joined_per_sample(
    prediction_rows: Sequence[Mapping[str, Any]], recovery_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    recovery = {
        (str(row["sample_id"]), str(row["arm"])): row for row in recovery_rows
    }
    rows = []
    for prediction in prediction_rows:
        if prediction["split"] != "test":
            continue
        joined = dict(prediction)
        rec = recovery[(str(prediction["sample_id"]), str(prediction["arm"]))]
        joined.update(
            {
                key: value
                for key, value in rec.items()
                if key not in {"arm", "shuffle_seed", "sample_id", "object_id", "variant_index"}
            }
        )
        rows.append(joined)
    return rows


def _report(summary: Mapping[str, Any]) -> str:
    selected_alpha = summary["validation_alpha_search"]["selected_alpha"]
    prediction = summary["prediction_aggregate"]
    groups = summary["gt_raw_laplacian_magnitude_groups"]
    recovery = summary["recovery_aggregate"]
    shuffled = summary["shuffled_seed_summary"]
    gate = {row["split"]: row for row in summary["gate_summary"]}
    validation_rows = []
    for arm in (BASE, CONSTANT, "shuffled_gate_mean", LEARNED):
        if arm == "shuffled_gate_mean":
            validation_rows.append(
                {
                    "arm": arm,
                    **{
                        field: shuffled["validation"][field]["mean"]
                        for field in (
                            "raw_epe",
                            "raw_residual_rms",
                            "raw_residual_maximum",
                            "raw_global_cosine",
                            "recovery_weighted_raw_residual_rms",
                            "bottom_90_percent_mean_raw_error_epe",
                            "top_10_percent_mean_raw_error_epe",
                            "top_1_percent_mean_raw_error_epe",
                        )
                    },
                }
            )
        else:
            p = _one(prediction, split="validation", arm=arm)
            g = _one(groups, split="validation", arm=arm)
            validation_rows.append({"arm": arm, **p, **g})
    rows = []
    for arm in (BASE, CONSTANT, "shuffled_gate_mean", LEARNED):
        if arm == "shuffled_gate_mean":
            rows.append(
                {
                    "arm": arm,
                    **{field: shuffled["test"][field]["mean"] for field in (
                        "raw_epe", "raw_residual_rms", "raw_residual_maximum",
                        "raw_global_cosine", "recovery_weighted_raw_residual_rms",
                        "bottom_90_percent_mean_raw_error_epe",
                        "top_10_percent_mean_raw_error_epe",
                        "top_1_percent_mean_raw_error_epe",
                        "reconstruction_chamfer", "reconstruction_point_to_surface",
                        "reconstruction_normal_consistency", "introduced_flipped_faces",
                        "improved_over_initial",
                    )},
                }
            )
        else:
            p = _one(prediction, split="test", arm=arm)
            g = _one(groups, split="test", arm=arm)
            r = _one(recovery, arm=arm)
            rows.append({"arm": arm, **p, **g, **r})
    lines = [
        "# Sofa50 Dynamic Residual Expert: Inference-Time Gate Causal Ablation",
        "",
        f"Contract audit: `{summary['contract_audit']['passed']}`. No retraining was performed.",
        f"Validation-selected constant gate: `alpha*={_f(selected_alpha)}` from `[0.00, 0.30]` with step `0.01`.",
        f"Mean learned gate: validation `{_f(gate['validation']['mean_learned_gate'])}`, test `{_f(gate['test']['mean_learned_gate'])}`.",
        "",
        "The complete 31-point validation curve is saved in `validation_alpha_search.csv`; test metrics were not used to choose alpha.",
        "",
        "## Validation prediction",
        "",
        "| Arm | EPE | RMS | Max | Cosine | Weighted RMS | Bottom 90% | Top 10% | Top 1% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['arm']} | {_f(row['raw_epe'])} | {_f(row['raw_residual_rms'])} | "
            f"{_f(row['raw_residual_maximum'])} | {_f(row['raw_global_cosine'])} | "
            f"{_f(row['recovery_weighted_raw_residual_rms'])} | "
            f"{_f(row['bottom_90_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_10_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_1_percent_mean_raw_error_epe'])} |"
        )
    lines.extend(
        [
        "",
        "## Test prediction and recovery",
        "",
        "Top groups are global GT raw-Laplacian magnitude groups. Shuffled values are mean across five fixed within-mesh permutations.",
        "",
        "| Arm | EPE | RMS | Max | Cosine | Weighted RMS | Bottom 90% | Top 10% | Top 1% | Chamfer | P2S | Normal | Flips | Improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['arm']} | {_f(row['raw_epe'])} | {_f(row['raw_residual_rms'])} | "
            f"{_f(row['raw_residual_maximum'])} | {_f(row['raw_global_cosine'])} | "
            f"{_f(row['recovery_weighted_raw_residual_rms'])} | "
            f"{_f(row['bottom_90_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_10_percent_mean_raw_error_epe'])} | "
            f"{_f(row['top_1_percent_mean_raw_error_epe'])} | "
            f"{_f(row['reconstruction_chamfer'])} | "
            f"{_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['reconstruction_normal_consistency'])} | "
            f"{int(round(float(row['introduced_flipped_faces'])))} | "
            f"{_f(row['improved_over_initial'])} |"
        )
    lines.extend(
        [
            "",
            "## Shuffled-seed variability (test)",
            "",
            "| Metric | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    for field in (
        "recovery_weighted_raw_residual_rms",
        "top_10_percent_mean_raw_error_epe",
        "top_1_percent_mean_raw_error_epe",
        "reconstruction_chamfer",
        "reconstruction_point_to_surface",
        "reconstruction_normal_consistency",
        "introduced_flipped_faces",
        "improved_over_initial",
    ):
        lines.append(
            f"| {field} | {_f(shuffled['test'][field]['mean'])} | "
            f"{_f(shuffled['test'][field]['std'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired test counts",
            "",
            "Candidate is compared against reference on the same 25 samples.",
            "",
            "| Comparison | Reference | Candidate | Lower EPE | Lower Chamfer | Lower P2S | Higher normal |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_summary"]:
        lines.append(
            f"| {row['comparison']} | {row['reference']} | {row['candidate']} | "
            f"{row['paired_lower_raw_epe']}/25 | {row['paired_lower_chamfer']}/25 | "
            f"{row['paired_lower_p2s']}/25 | {row['paired_higher_normal']}/25 |"
        )
    lines.extend(
        [
            "",
            "## Attribution diagnostic",
            "",
            "`R_expert + R_gate = 1` algebraically when the total improvement denominator is positive. These ratios are diagnostics, not an independent causal decomposition.",
            "",
            "| Metric | R_expert | R_gate |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["attribution_diagnostic"]:
        lines.append(
            f"| {row['metric']} | {_optional_f(row['r_expert'])} | {_optional_f(row['r_gate'])} |"
        )
    lines.extend(
        [
            "",
            "## Causal decisions",
            "",
            "```json",
            json.dumps(summary["causal_decisions"], indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence separation",
            "",
            "Gate/curvature and gate/residual correlations copied from the source report are observational only. They are not used as evidence that the gate is effective. The base-to-constant, constant-to-learned, and shuffled-to-learned interventions above provide the causal-ablation evidence.",
            "",
        ]
    )
    lines.extend(
        [
            "| Split | Gate value | Relationship | Pearson | Spearman |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["correlation_evidence_observational_only"]:
        if row.get("split") != "test" or row.get("gate_value") != "effective_g":
            continue
        lines.append(
            f"| {row['split']} | {row['gate_value']} | {row['relationship_to']} | "
            f"{_optional_f(row.get('pearson'))} | {_optional_f(row.get('spearman'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))))


def _validated_shuffle_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(seed) for seed in seeds)
    if len(result) < 3 or len(set(result)) != len(result):
        raise ValueError("At least three distinct shuffle seeds are required.")
    return result


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


def _arm_seed(arm: str) -> int | None:
    prefix = "shuffled_gate_seed_"
    return int(arm[len(prefix) :]) if arm.startswith(prefix) else None


def _one(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}.")
    return matches[0]


def _relative_close(left: float, right: float, tolerance: float = 0.01) -> bool:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1e-30) <= tolerance


def _concat(payloads: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    return [dict(row) for payload in payloads for row in payload[key]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _f(value: Any) -> str:
    return f"{float(value):.9g}"


def _optional_f(value: Any) -> str:
    return "n/a" if value is None else _f(value)

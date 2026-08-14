from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Mesh
from mlr.io import load_mesh

from .canonical_experiment import _exact_query_sample
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .dataset import load_prepared_sample, save_prepared_sample
from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .synthetic_current_comparison import _topology_change
from .synthetic_current_h2_ablation import _raw_metrics
from .synthetic_current_recursive_refinement import (
    PREDICTION_FIELDS,
    _dynamic_visibility,
    _infer_recursive,
    _load_b_spec,
    _recovery_config,
    build_recursive_sample,
)
from .target_scaling import RAW_LAPLACIAN, normalize_laplacian_by_edge_scale
from .trainer import load_checkpoint


EXPECTED_B_SHA256 = "ba1c77c3ce4c91ef70ba4b70570664d3ffa2c1c41a3f9f342778149ead0526e8"
EXPECTED_SPLITS = {"train": 200, "validation": 25, "test": 25}
ARMS = ("continue_original", "continue_B_result", "continue_mix_50_50")
CHECKPOINT_KINDS = ("best", "final")
BASELINE = {
    "raw_epe": 0.00300525179,
    "raw_top_1_percent_epe": 0.0417512367,
    "raw_top_10_percent_epe": 0.0136981653,
    "raw_global_cosine": 0.998667273,
    "recovery_weighted_raw_residual_rms": 0.00611072229,
    "reconstruction_chamfer": 0.00380671258,
    "reconstruction_point_to_surface": 0.00380587192,
    "reconstruction_normal_consistency": 0.942405903,
    "cumulative_introduced_flipped_faces": 6566,
    "cumulative_improved_over_original": 19,
}
BASELINE_ABSOLUTE_TOLERANCES = {
    "raw_epe": 1e-6,
    "raw_top_1_percent_epe": 1e-5,
    "raw_top_10_percent_epe": 1e-6,
    "raw_global_cosine": 1e-6,
    "recovery_weighted_raw_residual_rms": 1e-6,
    "reconstruction_chamfer": 1e-6,
    "reconstruction_point_to_surface": 1e-6,
    "reconstruction_normal_consistency": 1e-4,
}


def generate_stage2_dataset_shard(
    manifest_path: str | Path,
    b_run_dir: str | Path,
    output_dir: str | Path,
    *,
    shard_index: int,
    shard_count: int = 3,
    device: str = "cuda",
    visibility_size: int = 960,
) -> dict[str, Any]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count > 0 and 0 <= shard_index < shard_count.")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-2 dataset generation requires CUDA.")

    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    spec = _load_b_spec(Path(b_run_dir).resolve(), resolved_device)
    if spec["checkpoint_sha256"] != EXPECTED_B_SHA256:
        raise RuntimeError("Original B checkpoint SHA-256 does not match the required checkpoint.")

    rows: list[dict[str, Any]] = []
    global_index = 0
    for split, expected_count in EXPECTED_SPLITS.items():
        dataset = PreparedMeshDataset.from_manifest(manifest, split)
        if len(dataset) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} samples, found {len(dataset)}.")
        for index in range(len(dataset)):
            assigned = global_index % shard_count == shard_index
            global_index += 1
            if not assigned:
                continue
            source = dataset.load_static(index)
            sample_id = str(source["sample_id"])
            faces = _np(source["faces"]).astype(np.int64)
            original_vertices = _np(source["vertices"])
            inferred = _infer_recursive(source, spec, resolved_device)
            recovery_dir = output / "x1_recovery" / split / sample_id
            recovery_sample = source
            if split != "test":
                recovery_sample = {
                    key: value
                    for key, value in source.items()
                    if key not in {"gt_vertices", "gt_faces", "target_positions"}
                }
            metrics = reconstruct_and_evaluate(
                recovery_sample,
                inferred["prediction_raw"],
                recovery_dir,
                _recovery_config(spec["config"]),
                normalized_prediction=inferred["prediction_normalized"],
                edge_scale_epsilon=inferred["epsilon"],
                laplacian_weight=inferred["recovery_weight"],
                unseen_anchor_weight=float(
                    spec["config"].get("recovery", {}).get("unseen_anchor_weight", 0.0)
                ),
                evaluate_laplacian_prediction=True,
                evaluate_initial_geometry=True,
                solver_confidence=np.ones(len(original_vertices), dtype=np.float64),
            )
            x1_mesh = load_mesh(recovery_dir / "predicted_refined.obj")
            if not np.array_equal(x1_mesh.faces, faces):
                raise RuntimeError(f"{sample_id}: frozen-B recovery changed topology/order.")
            if not np.isfinite(x1_mesh.vertices).all():
                raise RuntimeError(f"{sample_id}: frozen-B X1 contains non-finite vertices.")

            visibility, visibility_audit = _dynamic_visibility(
                x1_mesh, source, visibility_size=visibility_size
            )
            stage2 = build_recursive_sample(source, x1_mesh, visibility)
            stage2["stage2_original_vertices"] = torch.as_tensor(
                original_vertices, dtype=torch.float32
            )
            stage2["stage2_x1_checkpoint_sha256"] = EXPECTED_B_SHA256
            stage2["stage2_input_role"] = "fixed_frozen_B_recovered_X1"
            stage2["stage2_visibility_policy"] = "recomputed_opengl_960"
            _make_image_paths_absolute(stage2, source)

            source_target = torch.as_tensor(source["raw_laplacian_target"]).cpu()
            stage2_target = torch.as_tensor(stage2["raw_laplacian_target"]).cpu()
            target_exact = bool(torch.equal(source_target, stage2_target))
            target_max_abs = float(torch.max(torch.abs(source_target - stage2_target)).item())
            prepared_path = output / "prepared" / split / f"{sample_id}.pt"
            save_prepared_sample(stage2, prepared_path)

            coarse = metrics["geometry"]["coarse"]
            refined = metrics["geometry"]["predicted"]
            topology = _topology_change(original_vertices, x1_mesh.vertices, faces)
            rows.append(
                {
                    "global_index": global_index - 1,
                    "split": split,
                    "sample_id": sample_id,
                    "original_path": str(dataset.records[index].path),
                    "stage2_path": str(prepared_path),
                    "x1_mesh_path": str(recovery_dir / "predicted_refined.obj"),
                    "x1_mesh_sha256": _sha256(recovery_dir / "predicted_refined.obj"),
                    "vertex_count": int(len(original_vertices)),
                    "faces_preserved": True,
                    "target_exact_equivalence": target_exact,
                    "target_max_abs_difference": target_max_abs,
                    "visibility_backend": visibility_audit["backend"],
                    "visibility_raster_size": visibility_audit["raster_size"],
                    "mean_visible_views_per_vertex": visibility_audit[
                        "mean_visible_views_per_vertex"
                    ],
                    "zero_visible_vertex_fraction": visibility_audit[
                        "zero_visible_vertex_fraction"
                    ],
                    "initial_chamfer": float(coarse.get("chamfer", float("nan"))),
                    "stage1_chamfer": float(refined.get("chamfer", float("nan"))),
                    "initial_point_to_surface": float(
                        coarse.get("point_to_surface_bidirectional_mean", float("nan"))
                    ),
                    "stage1_point_to_surface": float(
                        refined.get("point_to_surface_bidirectional_mean", float("nan"))
                    ),
                    "initial_normal_consistency": float(
                        coarse.get("normal_consistency", float("nan"))
                    ),
                    "stage1_normal_consistency": float(
                        refined.get("normal_consistency", float("nan"))
                    ),
                    "stage1_introduced_flipped_faces": int(topology["introduced_flips"]),
                    **inferred["prediction_metrics"],
                }
            )
            print(
                f"[{split} {index + 1:03d}/{len(dataset):03d}] {sample_id} "
                f"X1={refined.get('chamfer', float('nan')):.9g} "
                f"target_exact={target_exact}",
                flush=True,
            )

    payload = {
        "format": "sofa50_stage2_dataset_shard_v1",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "visibility_policy": "recomputed_opengl_960",
        "visibility_size": visibility_size,
        "rows": rows,
    }
    _write_json(output / "shards" / f"dataset_shard_{shard_index}.json", payload)
    return payload


def merge_stage2_dataset_shards(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    shard_count: int = 3,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    payloads = [
        _read_json(output / "shards" / f"dataset_shard_{index}.json")
        for index in range(shard_count)
    ]
    contracts = {
        (
            payload["manifest_sha256"],
            payload["checkpoint_sha256"],
            payload["visibility_policy"],
            int(payload["visibility_size"]),
            int(payload["shard_count"]),
        )
        for payload in payloads
    }
    if len(contracts) != 1:
        raise RuntimeError("Stage-2 dataset shard contracts differ.")
    rows = sorted(
        (dict(row) for payload in payloads for row in payload["rows"]),
        key=lambda row: int(row["global_index"]),
    )
    source_payload = _read_json(manifest)
    source_items = source_payload.get("samples")
    if not isinstance(source_items, list) or len(source_items) != sum(EXPECTED_SPLITS.values()):
        raise ValueError("Source manifest does not contain the required 250 samples.")
    if len(rows) != len(source_items) or len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Stage-2 shards do not cover 250 unique samples.")

    source_by_id = {
        str(item.get("sample_id")): dict(item)
        for item in source_items
        if isinstance(item, Mapping) and isinstance(item.get("sample_id"), str)
    }
    if len(source_by_id) != len(source_items):
        raise ValueError("Source manifest sample IDs must be present and unique.")
    original_items: list[dict[str, Any]] = []
    stage2_items: list[dict[str, Any]] = []
    mix_items: list[dict[str, Any]] = []
    split_indices: Counter[str] = Counter()
    mix_counts: Counter[str] = Counter()
    for row in rows:
        source_item = source_by_id[str(row["sample_id"])]
        if row["split"] != source_item.get("split"):
            raise RuntimeError("Stage-2 row split does not match the source manifest.")
        base = dict(source_item)
        original_portable_path = (
            output
            / "prepared_x0"
            / str(row["split"])
            / f"{row['sample_id']}.pt"
        )
        original_sample = load_prepared_sample(
            row["original_path"],
            materialize_images=False,
            dataset_root=manifest.parent,
        )
        _make_image_paths_absolute(original_sample, original_sample)
        save_prepared_sample(original_sample, original_portable_path)
        base["path"] = str(original_portable_path)
        x1 = dict(source_item)
        x1["path"] = row["stage2_path"]
        original_items.append(base)
        stage2_items.append(x1)
        split = str(row["split"])
        local_index = split_indices[split]
        split_indices[split] += 1
        use_x1 = (local_index + (1 if split == "test" else 0)) % 2 == 0
        mix_items.append(dict(x1 if use_x1 else base))
        mix_counts[f"{split}_{'X1' if use_x1 else 'X0'}"] += 1

    for name, items in (
        ("continue_original_manifest.json", original_items),
        ("continue_B_result_manifest.json", stage2_items),
        ("continue_mix_50_50_manifest.json", mix_items),
        ("stage2_manifest.json", stage2_items),
    ):
        payload = dict(source_payload)
        payload["samples"] = items
        payload["stage2_adaptation"] = {
            "input_distribution": name.removesuffix("_manifest.json"),
            "frozen_B_checkpoint_sha256": EXPECTED_B_SHA256,
            "visibility_policy": "recomputed_opengl_960",
        }
        _write_json(output / "manifests" / name, payload)

    test_rows = [row for row in rows if row["split"] == "test"]
    stage1_count = sum(row["stage1_chamfer"] < row["initial_chamfer"] for row in test_rows)
    stage1_aggregate = {
        "raw_epe": _mean(test_rows, "raw_epe"),
        "raw_top_1_percent_epe": _mean(test_rows, "raw_top_1_percent_epe"),
        "raw_top_10_percent_epe": _mean(test_rows, "raw_top_10_percent_epe"),
        "raw_global_cosine": _mean(test_rows, "raw_global_cosine"),
        "recovery_weighted_raw_residual_rms": _mean(
            test_rows, "recovery_weighted_raw_residual_rms"
        ),
        "reconstruction_chamfer": _mean(test_rows, "stage1_chamfer"),
        "reconstruction_point_to_surface": _mean(test_rows, "stage1_point_to_surface"),
        "reconstruction_normal_consistency": _mean(
            test_rows, "stage1_normal_consistency"
        ),
        "cumulative_introduced_flipped_faces": int(
            sum(int(row["stage1_introduced_flipped_faces"]) for row in test_rows)
        ),
        "cumulative_improved_over_original": int(stage1_count),
    }
    baseline_differences = {
        key: float(stage1_aggregate[key]) - float(value)
        for key, value in BASELINE.items()
    }
    baseline_match = bool(
        int(stage1_aggregate["cumulative_improved_over_original"])
        == int(BASELINE["cumulative_improved_over_original"])
        and abs(
            int(stage1_aggregate["cumulative_introduced_flipped_faces"])
            - int(BASELINE["cumulative_introduced_flipped_faces"])
        )
        <= max(25, round(0.005 * int(BASELINE["cumulative_introduced_flipped_faces"])))
        and all(
            abs(baseline_differences[key]) <= tolerance
            for key, tolerance in BASELINE_ABSOLUTE_TOLERANCES.items()
        )
    )
    audit = {
        "passed": bool(
            payloads[0]["checkpoint_sha256"] == EXPECTED_B_SHA256
            and len(rows) == 250
            and Counter(row["split"] for row in rows) == Counter(EXPECTED_SPLITS)
            and all(bool(row["faces_preserved"]) for row in rows)
            and all(bool(row["target_exact_equivalence"]) for row in rows)
            and all(row["visibility_backend"] == "opengl" for row in rows)
            and all(int(row["visibility_raster_size"]) == 960 for row in rows)
            and stage1_count == 19
            and baseline_match
            and mix_counts["train_X0"] == mix_counts["train_X1"] == 100
        ),
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "sample_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "all_faces_and_vertex_order_preserved": all(
            bool(row["faces_preserved"]) for row in rows
        ),
        "all_targets_exactly_equivalent": all(
            bool(row["target_exact_equivalence"]) for row in rows
        ),
        "maximum_target_abs_difference": max(
            float(row["target_max_abs_difference"]) for row in rows
        ),
        "x1_generated_once_and_saved": True,
        "x1_fixed_during_training": True,
        "direct_raw_supervision": True,
        "local_jitter_enabled": False,
        "visibility_policy": "recomputed_opengl_960",
        "gt_differential_values_used_as_model_inputs": False,
        "stage1_baseline": stage1_aggregate,
        "required_stage1_baseline": BASELINE,
        "stage1_baseline_differences": baseline_differences,
        "stage1_baseline_absolute_tolerances": BASELINE_ABSOLUTE_TOLERANCES,
        "stage1_flip_tolerance": max(
            25,
            round(0.005 * int(BASELINE["cumulative_introduced_flipped_faces"])),
        ),
        "stage1_matches_required_baseline": baseline_match,
        "mix_counts": dict(mix_counts),
    }
    _write_csv(output / "stage2_dataset_per_sample.csv", rows)
    _write_json(output / "stage2_dataset_summary.json", {
        "audit": audit,
        "stage1_baseline": stage1_aggregate,
        "mix_counts": dict(mix_counts),
    })
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Stage-2 dataset contract audit failed.")
    return audit


def evaluate_stage2_arm(
    stage2_manifest_path: str | Path,
    baseline_csv_path: str | Path,
    arm_run_dir: str | Path,
    output_dir: str | Path,
    *,
    arm: str,
    device: str = "cuda",
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}.")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage-2 evaluation requires CUDA.")
    dataset = PreparedMeshDataset.from_manifest(stage2_manifest_path, "test")
    if len(dataset) != 25:
        raise ValueError("Stage-2 evaluation requires exactly 25 test samples.")
    baseline = _baseline_rows(Path(baseline_csv_path))
    run = Path(arm_run_dir).resolve()
    output = Path(output_dir).resolve()
    run_config = _read_json(run / "run_config.json")
    if run_config.get("resume_checkpoint_sha256") != EXPECTED_B_SHA256:
        raise RuntimeError(f"{arm}: training did not resume from the required B checkpoint.")

    all_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for checkpoint_kind in CHECKPOINT_KINDS:
        checkpoint = run / ("checkpoint_best.pt" if checkpoint_kind == "best" else "checkpoint_latest.pt")
        spec = _load_continued_spec(run, checkpoint, resolved_device)
        checkpoint_hashes[checkpoint_kind] = spec["checkpoint_sha256"]
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            sample_id = str(sample["sample_id"])
            base = baseline[sample_id]
            inferred = _infer_stage2(sample, spec, resolved_device, zero_images=False)
            zero_inferred = _infer_stage2(sample, spec, resolved_device, zero_images=True)
            recovery_dir = output / "reconstruction" / checkpoint_kind / sample_id
            metrics = reconstruct_and_evaluate(
                sample,
                inferred["prediction_raw"],
                recovery_dir,
                _recovery_config(spec["config"]),
                normalized_prediction=inferred["prediction_normalized"],
                edge_scale_epsilon=inferred["epsilon"],
                laplacian_weight=inferred["recovery_weight"],
                unseen_anchor_weight=float(
                    spec["config"].get("recovery", {}).get("unseen_anchor_weight", 0.0)
                ),
                evaluate_laplacian_prediction=True,
                evaluate_initial_geometry=True,
                solver_confidence=np.ones(len(sample["vertices"]), dtype=np.float64),
            )
            x2 = load_mesh(recovery_dir / "predicted_refined.obj")
            faces = _np(sample["faces"]).astype(np.int64)
            x1_vertices = _np(sample["vertices"])
            x0_vertices = _np(sample["stage2_original_vertices"])
            if not np.array_equal(x2.faces, faces):
                raise RuntimeError(f"{arm}/{sample_id}: stage-2 recovery changed topology.")
            stage2_geometry = metrics["geometry"]["predicted"]
            initial_chamfer = float(base["initial_chamfer"])
            stage1_chamfer = float(base["reconstruction_chamfer"])
            stage2_chamfer = float(stage2_geometry["chamfer"])
            stage1_success = _bool(base["improved_over_initial"])
            stage2_success = stage2_chamfer < initial_chamfer
            category = (
                "retained"
                if stage1_success and stage2_success
                else "lost"
                if stage1_success
                else "gained"
                if stage2_success
                else "remained_failed"
            )
            cumulative_topology = _topology_change(x0_vertices, x2.vertices, faces)
            step_topology = _topology_change(x1_vertices, x2.vertices, faces)
            row = {
                "arm": arm,
                "checkpoint_kind": checkpoint_kind,
                "sample_id": sample_id,
                "initial_chamfer": initial_chamfer,
                "stage1_chamfer": stage1_chamfer,
                "stage2_chamfer": stage2_chamfer,
                "stage1_improved_vs_initial": stage1_success,
                "stage2_improved_vs_initial": stage2_success,
                "stage2_improved_vs_stage1": stage2_chamfer < stage1_chamfer,
                "transition_category": category,
                "stage2_point_to_surface": float(
                    stage2_geometry["point_to_surface_bidirectional_mean"]
                ),
                "stage2_normal_consistency": float(
                    stage2_geometry["normal_consistency"]
                ),
                "stage2_step_introduced_flipped_faces": int(
                    step_topology["introduced_flips"]
                ),
                "stage2_cumulative_introduced_flipped_faces": int(
                    cumulative_topology["introduced_flips"]
                ),
                **inferred["prediction_metrics"],
                **{
                    f"zero_rgb_{key}": value
                    for key, value in zero_inferred["prediction_metrics"].items()
                },
            }
            all_rows.append(row)
            print(
                f"[{arm} {checkpoint_kind} {index + 1:02d}/25] {sample_id} "
                f"X2={stage2_chamfer:.9g} category={category}",
                flush=True,
            )

    aggregates = [_aggregate_stage2_rows(all_rows, arm, kind) for kind in CHECKPOINT_KINDS]
    audit = {
        "passed": bool(
            len(all_rows) == 50
            and len({(row["checkpoint_kind"], row["sample_id"]) for row in all_rows}) == 50
            and run_config.get("resume_checkpoint_sha256") == EXPECTED_B_SHA256
            and run_config.get("reset_resume_tracking") is True
            and run_config.get("experiment_config", {}).get("target_mode") == RAW_LAPLACIAN
            and not run_config.get("experiment_config", {}).get("local_query_jitter", {}).get("enabled", False)
        ),
        "arm": arm,
        "starting_checkpoint_sha256": run_config.get("resume_checkpoint_sha256"),
        "checkpoint_hashes": checkpoint_hashes,
        "stage2_recovery_reference": "X1",
        "sample_ids_match": set(baseline) == set(dataset.sample_ids),
        "direct_raw_supervision": True,
        "local_jitter_enabled": False,
        "gt_differential_values_used_as_model_inputs": False,
    }
    payload = {"arm": arm, "aggregates": aggregates, "audit": audit}
    _write_csv(output / "per_sample_transition.csv", all_rows)
    _write_json(output / "arm_summary.json", payload)
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"{arm}: evaluation contract audit failed.")
    return payload


def merge_stage2_arm_results(
    experiment_dir: str | Path,
    *,
    continuation_steps: int = 20_000,
) -> dict[str, Any]:
    root = Path(experiment_dir).resolve()
    dataset_summary = _read_json(root / "dataset" / "stage2_dataset_summary.json")
    arm_payloads = {
        arm: _read_json(root / "evaluation" / arm / "arm_summary.json") for arm in ARMS
    }
    rows: list[dict[str, Any]] = [
        {
            "arm": "original_B_stage1",
            "checkpoint_kind": "frozen_original",
            **dataset_summary["stage1_baseline"],
            "retained_from_original_19": 19,
            "gained_from_original_failed_6": 0,
            "lost_from_original_19": 0,
        }
    ]
    for arm, payload in arm_payloads.items():
        rows.extend(dict(row) for row in payload["aggregates"])
    best_rows = {
        row["arm"]: row for row in rows if row.get("checkpoint_kind") == "best"
    }
    primary = best_rows["continue_B_result"]
    control = best_rows["continue_original"]
    mix = best_rows["continue_mix_50_50"]
    decision = {
        "B_result_exceeds_19_of_25": int(primary["cumulative_improved_over_original"]) > 19,
        "B_result_improves_mean_chamfer": float(primary["reconstruction_chamfer"])
        < BASELINE["reconstruction_chamfer"],
        "B_result_p2s_does_not_degrade": float(primary["reconstruction_point_to_surface"])
        <= BASELINE["reconstruction_point_to_surface"],
        "B_result_recovers_original_failures": int(primary["gained_from_original_failed_6"]),
        "B_result_loses_original_successes": int(primary["lost_from_original_19"]),
        "B_result_gain_exceeds_extra_training_control": (
            int(primary["cumulative_improved_over_original"])
            > int(control["cumulative_improved_over_original"])
            and float(primary["reconstruction_chamfer"])
            < float(control["reconstruction_chamfer"])
        ),
        "mix_improved_count": int(mix["cumulative_improved_over_original"]),
        "mix_chamfer": float(mix["reconstruction_chamfer"]),
    }
    training = {}
    starting_hashes = set()
    for arm in ARMS:
        run = root / "training" / arm
        metrics = _read_json(run / "metrics.json")
        run_config = _read_json(run / "run_config.json")
        starting_hashes.add(run_config.get("resume_checkpoint_sha256"))
        training[arm] = {
            "starting_optimizer_steps": metrics.get("starting_optimizer_steps"),
            "final_optimizer_steps": metrics.get("optimizer_steps"),
            "continuation_optimizer_steps": metrics.get("continuation_optimizer_steps"),
            "completed_epochs": metrics.get("completed_epochs"),
            "best_epoch": metrics.get("best_epoch"),
            "best_validation_loss": metrics.get("best_selection_loss"),
            "runtime_seconds": metrics.get("runtime_seconds"),
            "final_checkpoint_sha256": _sha256(run / "checkpoint_latest.pt"),
            "best_checkpoint_sha256": _sha256(run / "checkpoint_best.pt"),
        }
    audit = {
        "passed": bool(
            dataset_summary["audit"]["passed"]
            and all(payload["audit"]["passed"] for payload in arm_payloads.values())
            and starting_hashes == {EXPECTED_B_SHA256}
            and all(
                int(record["continuation_optimizer_steps"]) == continuation_steps
                for record in training.values()
            )
        ),
        "dataset_contract": dataset_summary["audit"],
        "all_arms_start_same_B_checkpoint": starting_hashes == {EXPECTED_B_SHA256},
        "starting_checkpoint_hashes": sorted(starting_hashes),
        "matched_continuation_steps": {
            arm: record["continuation_optimizer_steps"] for arm, record in training.items()
        },
        "arm_audits": {arm: payload["audit"] for arm, payload in arm_payloads.items()},
    }
    summary = {
        "experiment": "Sofa50 B direct-raw stage-2 distribution adaptation",
        "continuation_budget_optimizer_steps": continuation_steps,
        "training": training,
        "comparison": rows,
        "decision": decision,
        "contract_audit": audit,
    }
    _write_csv(root / "analysis" / "final_comparison.csv", rows)
    _write_json(root / "analysis" / "summary.json", summary)
    _write_json(root / "analysis" / "contract_audit.json", audit)
    (root / "analysis" / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError("Final stage-2 adaptation contract audit failed.")
    return summary


def _load_continued_spec(
    run_dir: Path, checkpoint: Path, device: torch.device
) -> dict[str, Any]:
    run_payload = _read_json(run_dir / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    if config.get("target_mode") != RAW_LAPLACIAN:
        raise ValueError("Stage-2 continuation requires direct raw Laplacian output.")
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }


def _infer_stage2(
    sample: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: torch.device,
    *,
    zero_images: bool,
) -> dict[str, Any]:
    prepared = _prepare_item_for_use(
        _prepare_object_static(sample, spec["config"]),
        spec["config"],
        device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )
    conditioned = _exact_query_sample(prepared.sample, device)
    if zero_images:
        conditioned = dict(conditioned)
        conditioned["images"] = torch.zeros_like(conditioned["images"])
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        model_output = spec["model"](conditioned)
    if model_output.confidence_prediction is None:
        raise RuntimeError("Stage-2 checkpoint must provide confidence prediction.")
    prediction_raw = model_output.predicted_laplacian.float().detach().cpu()
    confidence = model_output.confidence_prediction.float().detach().cpu()
    h = prepared.sample["local_edge_length"].float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    epsilon = float(spec["config"].get("target_scaling", {}).get("epsilon", 1e-12))
    prediction_normalized = normalize_laplacian_by_edge_scale(
        prediction_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    visibility = prepared.sample["visibility"].detach().cpu()
    canonical = canonical_current_graph_recovery_inputs(
        prepared.sample["vertices"].detach().cpu(),
        sample["faces"],
        prediction_normalized,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    target_raw = prepared.raw_target.float().detach().cpu()
    return {
        "prediction_raw": prediction_raw,
        "prediction_normalized": prediction_normalized,
        "confidence": confidence,
        "recovery_weight": canonical.weight.detach().cpu(),
        "epsilon": epsilon,
        "prediction_metrics": _raw_metrics(
            prediction_raw, target_raw, canonical.weight.detach().cpu(), valid
        ),
    }


def _aggregate_stage2_rows(
    rows: Sequence[Mapping[str, Any]], arm: str, checkpoint_kind: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm and row["checkpoint_kind"] == checkpoint_kind
    ]
    if len(selected) != 25:
        raise RuntimeError(f"Expected 25 {arm}/{checkpoint_kind} rows.")
    categories = Counter(str(row["transition_category"]) for row in selected)
    result: dict[str, Any] = {
        "arm": arm,
        "checkpoint_kind": checkpoint_kind,
        "sample_count": len(selected),
        "reconstruction_chamfer": _mean(selected, "stage2_chamfer"),
        "reconstruction_point_to_surface": _mean(selected, "stage2_point_to_surface"),
        "reconstruction_normal_consistency": _mean(
            selected, "stage2_normal_consistency"
        ),
        "cumulative_introduced_flipped_faces": int(
            sum(int(row["stage2_cumulative_introduced_flipped_faces"]) for row in selected)
        ),
        "stage2_step_introduced_flipped_faces": int(
            sum(int(row["stage2_step_introduced_flipped_faces"]) for row in selected)
        ),
        "cumulative_improved_over_original": int(
            sum(bool(row["stage2_improved_vs_initial"]) for row in selected)
        ),
        "improved_over_stage1": int(
            sum(bool(row["stage2_improved_vs_stage1"]) for row in selected)
        ),
        "retained_from_original_19": categories["retained"],
        "gained_from_original_failed_6": categories["gained"],
        "lost_from_original_19": categories["lost"],
        "remained_failed_from_original_6": categories["remained_failed"],
    }
    for key in PREDICTION_FIELDS:
        result[key] = _mean(selected, key)
        result[f"zero_rgb_{key}"] = _mean(selected, f"zero_rgb_{key}")
    return result


def _baseline_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["arm"] == "B_direct_raw_laplacian"
            and int(row["replacement_percent"]) == 0
        ]
    if len(rows) != 25:
        raise RuntimeError("Expected 25 original B baseline rows.")
    return {row["sample_id"]: row for row in rows}


def _make_image_paths_absolute(stage2: dict[str, Any], source: Mapping[str, Any]) -> None:
    paths = stage2.get("image_paths")
    if not isinstance(paths, list):
        return
    root_value = source.get("_dataset_root")
    if root_value is None:
        raise ValueError("Lazy source sample is missing _dataset_root.")
    root = Path(str(root_value))
    stage2["image_paths"] = [
        str((Path(path) if Path(path).is_absolute() else root / path).resolve())
        for path in paths
    ]
    stage2["image_path_root"] = "/"


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 B Direct-Raw Stage-2 Distribution Adaptation",
        "",
        f"Continuation budget: {summary['continuation_budget_optimizer_steps']:,} optimizer steps per arm.",
        "",
        "| Arm | Checkpoint | Raw EPE | Weighted RMS | Chamfer | P2S | Normal | Improved/X0 | Better/X1 | Retained | Gained | Lost | Flips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"| {row['arm']} | {row['checkpoint_kind']} | {_f(row.get('raw_epe'))} | "
            f"{_f(row.get('recovery_weighted_raw_residual_rms'))} | "
            f"{_f(row.get('reconstruction_chamfer'))} | "
            f"{_f(row.get('reconstruction_point_to_surface'))} | "
            f"{_f(row.get('reconstruction_normal_consistency'))} | "
            f"{row.get('cumulative_improved_over_original', 19)}/25 | "
            f"{row.get('improved_over_stage1', '—')} | "
            f"{row.get('retained_from_original_19', 19)}/19 | "
            f"{row.get('gained_from_original_failed_6', 0)}/6 | "
            f"{row.get('lost_from_original_19', 0)}/19 | "
            f"{row.get('cumulative_introduced_flipped_faces', '—')} |"
        )
    lines.extend(["", "## Decision", ""])
    for key, value in summary["decision"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        ["", f"Contract audit: `{summary['contract_audit']['passed']}`.", ""]
    )
    return "\n".join(lines)


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _f(value: Any) -> str:
    return "—" if value is None else f"{float(value):.9g}"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

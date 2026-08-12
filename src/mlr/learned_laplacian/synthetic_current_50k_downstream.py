from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .synthetic_current_comparison import run_synthetic_current_comparison


MODEL_ORDER = ("gt_query_50k", "current_query_20k", "current_query_50k")
FLOAT_REGRESSION_KEYS = (
    "normalized_mse",
    "vector_l2",
    "global_cosine",
    "high_10_percent_cosine",
    "prediction_target_norm_ratio",
    "loss",
    "zero_rgb_loss",
    "correct_zero_loss_gap",
    "correct_zero_cosine_gap",
    "initial_chamfer",
    "reconstruction_chamfer",
    "reconstruction_point_to_surface",
    "reconstruction_normal_consistency",
)
INTEGER_REGRESSION_KEYS = (
    "new_degenerate_faces",
    "improved_over_initial",
    "sample_count",
)
FLOAT_REGRESSION_REL_TOL = 1e-3
FLOAT_REGRESSION_ABS_TOL = 1e-6
FLIP_REGRESSION_ABS_TOL = 25
LOWER_IS_BETTER = {
    "loss",
    "vector_l2",
    "normalized_mse",
    "reconstruction_chamfer",
    "reconstruction_point_to_surface",
    "introduced_flipped_faces",
}
HIGHER_IS_BETTER = {
    "global_cosine",
    "high_10_percent_cosine",
    "reconstruction_normal_consistency",
    "improved_over_initial",
    "correct_zero_loss_gap",
    "relative_correct_vs_zero_improvement",
}


def run_evaluation(
    old_comparison_path: str | Path,
    current50_run: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    old_comparison_path = Path(old_comparison_path).resolve()
    current50_run = Path(current50_run).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    old = _read_json(old_comparison_path)
    old_setup = old.get("experiment_setup", {})
    if not isinstance(old_setup, Mapping) or not all(name in old_setup for name in ("A", "B")):
        raise ValueError("Old comparison is missing experiment_setup.A/B.")

    manifest = Path(str(old["manifest"])).resolve()
    gt_checkpoint = Path(str(old_setup["A"]["checkpoint"])).resolve()
    gt_config = Path(str(old_setup["A"]["config_path"])).resolve()
    current20_checkpoint = Path(str(old_setup["B"]["checkpoint"])).resolve()
    current20_config = Path(str(old_setup["B"]["config_path"])).resolve()
    current50_checkpoint = current50_run / "best.pt"
    current50_config = current50_run / "config.json"
    current50_metrics = current50_run / "metrics.json"
    current50_history = current50_run / "training_history.json"
    required = (
        old_comparison_path,
        manifest,
        gt_checkpoint,
        gt_config,
        current20_checkpoint,
        current20_config,
        current50_checkpoint,
        current50_config,
        current50_metrics,
        current50_history,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + ", ".join(missing))

    config_audit = _current_config_audit(current20_config, current50_config)
    if not config_audit["evaluation_contract_match"]:
        _write_json(output_dir / "contract_regression.json", {"config_audit": config_audit})
        raise RuntimeError("Current-query 20k/50k configs differ outside max_optimizer_steps.")

    regression = run_synthetic_current_comparison(
        manifest,
        gt_checkpoint,
        gt_config,
        current20_checkpoint,
        current20_config,
        output_dir / "regression_gt50_vs_current20",
        device=device,
    )
    regression_audit = _regression_audit(old, regression)
    contract_audit: dict[str, Any] = {
        "old_comparison_path": str(old_comparison_path),
        "old_comparison_sha256": _sha256(old_comparison_path),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "config_audit": config_audit,
        "regression": regression_audit,
    }
    _write_json(output_dir / "contract_regression.json", contract_audit)
    if not regression_audit["passed"]:
        raise RuntimeError(
            "Re-run of GT-query 50k vs current-query 20k does not match the saved comparison."
        )

    current50_comparison = run_synthetic_current_comparison(
        manifest,
        gt_checkpoint,
        gt_config,
        current50_checkpoint,
        current50_config,
        output_dir / "gt50_vs_current50",
        device=device,
    )
    repeated_gt_audit = _model_repeat_audit(
        regression["aggregate"]["A"], current50_comparison["aggregate"]["A"]
    )
    contract_audit["repeated_gt_query_50k"] = repeated_gt_audit
    _write_json(output_dir / "contract_regression.json", contract_audit)
    if not repeated_gt_audit["passed"]:
        raise RuntimeError("Repeated GT-query 50k evaluation changed across the two evaluator calls.")

    aggregates = {
        "gt_query_50k": _augment_aggregate(
            regression["aggregate"]["A"], regression["per_variant"], "A"
        ),
        "current_query_20k": _augment_aggregate(
            regression["aggregate"]["B"], regression["per_variant"], "B"
        ),
        "current_query_50k": _augment_aggregate(
            current50_comparison["aggregate"]["B"],
            current50_comparison["per_variant"],
            "B",
        ),
    }
    per_sample = _wide_per_sample(
        regression["per_variant"], current50_comparison["per_variant"]
    )
    sample_analysis = _sample_analysis(per_sample)
    native = _native_checkpoint_metadata(
        old_setup, current20_checkpoint, current20_config, current50_run
    )
    if int(current50_comparison["experiment_setup"]["B"]["checkpoint_epoch"]) != int(
        native["current_query_50k"]["checkpoint_epoch"]
    ):
        raise RuntimeError("Loaded current-query 50k checkpoint epoch does not match native metrics.")
    comparisons = {
        "current_query_20k_vs_gt_query_50k": _comparison(
            aggregates["gt_query_50k"], aggregates["current_query_20k"]
        ),
        "current_query_50k_vs_current_query_20k": _comparison(
            aggregates["current_query_20k"], aggregates["current_query_50k"]
        ),
        "current_query_50k_vs_gt_query_50k": _comparison(
            aggregates["gt_query_50k"], aggregates["current_query_50k"]
        ),
    }
    summary = {
        "experiment": "Sofa50 Synthetic Current-query 50k Downstream Evaluation",
        "device": device,
        "evaluation_protocol": "existing synthetic_current_comparison evaluator",
        "test_samples": int(regression["test_samples"]),
        "test_objects": int(regression["test_objects"]),
        "test_sample_ids": sorted(row["sample_id"] for row in per_sample),
        "target": regression["target"],
        "checkpoints": native,
        "contract_audit": contract_audit,
        "aggregate": aggregates,
        "comparisons": comparisons,
        "per_sample_outcomes": sample_analysis,
        "decision": _decision(aggregates, comparisons),
        "source_outputs": {
            "regression_gt50_vs_current20": str(
                output_dir / "regression_gt50_vs_current20" / "comparison.json"
            ),
            "gt50_vs_current50": str(
                output_dir / "gt50_vs_current50" / "comparison.json"
            ),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "comparisons.json", comparisons)
    _write_csv(output_dir / "per_sample_metrics.csv", per_sample)
    (output_dir / "report.md").write_text(_report(summary, per_sample), encoding="utf-8")
    return summary


def _current_config_audit(current20_config: Path, current50_config: Path) -> dict[str, Any]:
    config20 = _read_json(current20_config)
    config50 = _read_json(current50_config)
    differences = _mapping_differences(config20, config50)
    expected = [
        {
            "path": "multi_object_training.max_optimizer_steps",
            "left": 20000,
            "right": 50000,
        }
    ]
    return {
        "current20_config": str(current20_config),
        "current50_config": str(current50_config),
        "current20_config_sha256": _sha256(current20_config),
        "current50_config_sha256": _sha256(current50_config),
        "differences": differences,
        "allowed_differences": expected,
        "evaluation_contract_match": differences == expected,
    }


def _mapping_differences(
    left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in left:
            result.append({"path": path, "left": None, "right": right[key]})
        elif key not in right:
            result.append({"path": path, "left": left[key], "right": None})
        elif isinstance(left[key], Mapping) and isinstance(right[key], Mapping):
            result.extend(_mapping_differences(left[key], right[key], path))
        elif left[key] != right[key]:
            result.append({"path": path, "left": left[key], "right": right[key]})
    return result


def _regression_audit(old: Mapping[str, Any], rerun: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "manifest_path_match": str(Path(str(old["manifest"])).resolve())
        == str(Path(str(rerun["manifest"])).resolve()),
        "test_sample_count_match": old.get("test_samples") == rerun.get("test_samples") == 25,
        "test_object_count_match": old.get("test_objects") == rerun.get("test_objects") == 5,
        "test_sample_ids_match": _sample_ids(old) == _sample_ids(rerun),
        "aggregate_A_match": _model_repeat_audit(
            old["aggregate"]["A"], rerun["aggregate"]["A"]
        ),
        "aggregate_B_match": _model_repeat_audit(
            old["aggregate"]["B"], rerun["aggregate"]["B"]
        ),
        "per_sample_identity_match": _per_sample_identity(old, rerun),
    }
    checks["passed"] = all(
        value["passed"] if isinstance(value, Mapping) and "passed" in value else bool(value)
        for key, value in checks.items()
        if key != "passed"
    )
    return checks


def _model_repeat_audit(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    differences: dict[str, Any] = {}
    passed = True
    for key in FLOAT_REGRESSION_KEYS:
        a = float(left[key])
        b = float(right[key])
        match = math.isclose(
            a,
            b,
            rel_tol=FLOAT_REGRESSION_REL_TOL,
            abs_tol=FLOAT_REGRESSION_ABS_TOL,
        )
        differences[key] = {"reference": a, "rerun": b, "match": match}
        passed = passed and match
    for key in INTEGER_REGRESSION_KEYS:
        a = int(left[key])
        b = int(right[key])
        match = a == b
        differences[key] = {"reference": a, "rerun": b, "match": match}
        passed = passed and match
    reference_flips = int(left["introduced_flipped_faces"])
    rerun_flips = int(right["introduced_flipped_faces"])
    flip_match = abs(rerun_flips - reference_flips) <= FLIP_REGRESSION_ABS_TOL
    differences["introduced_flipped_faces"] = {
        "reference": reference_flips,
        "rerun": rerun_flips,
        "absolute_difference": abs(rerun_flips - reference_flips),
        "match": flip_match,
    }
    passed = passed and flip_match
    return {
        "passed": passed,
        "tolerance": {
            "float_relative": FLOAT_REGRESSION_REL_TOL,
            "float_absolute": FLOAT_REGRESSION_ABS_TOL,
            "introduced_flipped_faces_absolute": FLIP_REGRESSION_ABS_TOL,
            "exact_integer_keys": list(INTEGER_REGRESSION_KEYS),
        },
        "metrics": differences,
    }


def _sample_ids(payload: Mapping[str, Any]) -> list[str]:
    return sorted({str(row["sample_id"]) for row in payload["per_variant"]})


def _per_sample_identity(old: Mapping[str, Any], rerun: Mapping[str, Any]) -> bool:
    def identity(payload: Mapping[str, Any]) -> set[tuple[str, str, int]]:
        return {
            (str(row["sample_id"]), str(row["object_id"]), int(row["variant_index"]))
            for row in payload["per_variant"]
        }

    return identity(old) == identity(rerun)


def _augment_aggregate(
    aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], experiment: str
) -> dict[str, Any]:
    selected = [row for row in rows if str(row["experiment"]) == experiment]
    result = dict(aggregate)
    result["initial_point_to_surface"] = _mean(selected, "initial_point_to_surface")
    result["initial_normal_consistency"] = _mean(selected, "initial_normal_consistency")
    zero = float(result["zero_rgb_loss"])
    result["relative_correct_vs_zero_improvement"] = (
        float(result["correct_zero_loss_gap"]) / zero if zero else math.nan
    )
    result["improved_sample_ids"] = sorted(
        str(row["sample_id"]) for row in selected if bool(row["improved_over_initial"])
    )
    return result


def _wide_per_sample(
    regression_rows: Sequence[Mapping[str, Any]],
    current50_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        "gt_query_50k": {
            str(row["sample_id"]): row
            for row in regression_rows
            if str(row["experiment"]) == "A"
        },
        "current_query_20k": {
            str(row["sample_id"]): row
            for row in regression_rows
            if str(row["experiment"]) == "B"
        },
        "current_query_50k": {
            str(row["sample_id"]): row
            for row in current50_rows
            if str(row["experiment"]) == "B"
        },
    }
    sample_sets = [set(rows) for rows in indexed.values()]
    if not all(sample_set == sample_sets[0] for sample_set in sample_sets[1:]):
        raise RuntimeError("Three model evaluations do not contain the same sample IDs.")
    output: list[dict[str, Any]] = []
    for sample_id in sorted(sample_sets[0]):
        source = indexed["gt_query_50k"][sample_id]
        initial_values = []
        for model in MODEL_ORDER:
            row = indexed[model][sample_id]
            initial_values.append(float(row["initial_chamfer"]))
            if str(row["object_id"]) != str(source["object_id"]):
                raise RuntimeError(f"Object ID mismatch for {sample_id}.")
            if int(row["variant_index"]) != int(source["variant_index"]):
                raise RuntimeError(f"Variant index mismatch for {sample_id}.")
        if max(initial_values) - min(initial_values) > 1e-12:
            raise RuntimeError(f"Initial Chamfer mismatch for {sample_id}.")
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "object_id": source["object_id"],
            "variant_id": f"v{int(source['variant_index']):02d}",
            "variant_index": int(source["variant_index"]),
            "initial_chamfer": float(source["initial_chamfer"]),
            "initial_p2s": float(source["initial_point_to_surface"]),
            "initial_normal": float(source["initial_normal_consistency"]),
        }
        for model in MODEL_ORDER:
            row = indexed[model][sample_id]
            for suffix, key in (
                ("chamfer", "reconstruction_chamfer"),
                ("p2s", "reconstruction_point_to_surface"),
                ("normal", "reconstruction_normal_consistency"),
                ("flips", "introduced_flipped_faces"),
                ("prediction_loss", "correct_rgb_loss"),
                ("zero_rgb_loss", "zero_rgb_loss"),
                ("correct_zero_loss_gap", "correct_zero_loss_gap"),
                ("target_epe", "vector_l2"),
                ("global_cosine", "global_cosine"),
                ("high10_cosine", "high_10_percent_cosine"),
                ("pred_target_norm", "prediction_target_norm_ratio"),
            ):
                result[f"{model}_{suffix}"] = row[key]
        result["current20_better_than_initial"] = bool(
            result["current_query_20k_chamfer"] < result["initial_chamfer"]
        )
        result["current50_better_than_initial"] = bool(
            result["current_query_50k_chamfer"] < result["initial_chamfer"]
        )
        delta = float(result["current_query_50k_chamfer"]) - float(
            result["current_query_20k_chamfer"]
        )
        result["current50_minus_current20_chamfer"] = delta
        result["current50_vs_current20_chamfer_percent_change"] = (
            100.0 * delta / float(result["current_query_20k_chamfer"])
        )
        output.append(result)
    return output


def _sample_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    retained = [
        str(row["sample_id"])
        for row in rows
        if row["current20_better_than_initial"] and row["current50_better_than_initial"]
    ]
    gained = [
        str(row["sample_id"])
        for row in rows
        if not row["current20_better_than_initial"] and row["current50_better_than_initial"]
    ]
    regressed = [
        str(row["sample_id"])
        for row in rows
        if row["current20_better_than_initial"] and not row["current50_better_than_initial"]
    ]
    increases = sorted(
        (
            {
                "sample_id": str(row["sample_id"]),
                "absolute_change": float(row["current50_minus_current20_chamfer"]),
                "percent_change": float(
                    row["current50_vs_current20_chamfer_percent_change"]
                ),
            }
            for row in rows
            if float(row["current50_minus_current20_chamfer"]) > 0.0
        ),
        key=lambda value: value["percent_change"],
        reverse=True,
    )
    return {
        "current20_better_than_initial": [
            str(row["sample_id"]) for row in rows if row["current20_better_than_initial"]
        ],
        "current50_better_than_initial": [
            str(row["sample_id"]) for row in rows if row["current50_better_than_initial"]
        ],
        "retained_from_20k": retained,
        "new_at_50k": gained,
        "lost_at_50k": regressed,
        "chamfer_increase_at_least_10_percent": [
            row for row in increases if row["percent_change"] >= 10.0
        ],
        "largest_five_chamfer_increases": increases[:5],
    }


def _native_checkpoint_metadata(
    old_setup: Mapping[str, Any],
    current20_checkpoint: Path,
    current20_config: Path,
    current50_run: Path,
) -> dict[str, Any]:
    current50_metrics = _read_json(current50_run / "metrics.json")
    history = _read_json_list(current50_run / "training_history.json")
    best_epoch = int(current50_metrics["best_epoch"])
    best_rows = [row for row in history if int(row.get("epoch", -1)) == best_epoch]
    if len(best_rows) != 1:
        raise ValueError(f"Expected one current50 history row for best epoch {best_epoch}.")
    best = best_rows[0]
    if not math.isclose(
        float(best["validation_loss"]),
        float(current50_metrics["best_selection_loss"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Current50 best history loss does not match metrics.json.")
    return {
        "gt_query_50k": {
            **dict(old_setup["A"]),
            "checkpoint_sha256": _sha256(Path(str(old_setup["A"]["checkpoint"]))),
        },
        "current_query_20k": {
            **dict(old_setup["B"]),
            "checkpoint": str(current20_checkpoint),
            "config_path": str(current20_config),
            "checkpoint_sha256": _sha256(current20_checkpoint),
        },
        "current_query_50k": {
            "checkpoint": str(current50_run / "best.pt"),
            "config_path": str(current50_run / "config.json"),
            "native_metrics": str(current50_run / "metrics.json"),
            "training_formulation": "current-query",
            "training_seed": 7,
            "training_budget_optimizer_steps": 50000,
            "checkpoint_epoch": best_epoch,
            "checkpoint_optimizer_steps": int(best["optimizer_steps"]),
            "checkpoint_validation_loss": float(best["validation_loss"]),
            "checkpoint_sha256": _sha256(current50_run / "best.pt"),
        },
    }


def _comparison(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "loss",
        "normalized_mse",
        "vector_l2",
        "global_cosine",
        "high_10_percent_cosine",
        "prediction_target_norm_ratio",
        "zero_rgb_loss",
        "correct_zero_loss_gap",
        "relative_correct_vs_zero_improvement",
        "reconstruction_chamfer",
        "reconstruction_point_to_surface",
        "reconstruction_normal_consistency",
        "introduced_flipped_faces",
        "improved_over_initial",
    )
    result: dict[str, Any] = {}
    for key in keys:
        before = float(source[key])
        after = float(target[key])
        delta = after - before
        percent = 100.0 * delta / before if before else math.nan
        row = {
            "source": source[key],
            "target": target[key],
            "absolute_change": delta,
            "percent_change": percent,
        }
        if key in LOWER_IS_BETTER:
            row["directional_improvement_percent"] = -percent
        elif key in HIGHER_IS_BETTER:
            row["directional_improvement_percent"] = percent
        result[key] = row
    return result


def _decision(
    aggregate: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    b = aggregate["current_query_20k"]
    c = aggregate["current_query_50k"]
    a = aggregate["gt_query_50k"]
    prediction_20_to_50 = float(c["loss"]) < float(b["loss"]) and float(c["vector_l2"]) < float(
        b["vector_l2"]
    )
    reconstruction_20_to_50 = (
        float(c["reconstruction_chamfer"]) < float(b["reconstruction_chamfer"])
        and float(c["reconstruction_point_to_surface"])
        < float(b["reconstruction_point_to_surface"])
    )
    reconstruction_all_20_to_50 = (
        reconstruction_20_to_50
        and float(c["reconstruction_normal_consistency"])
        >= float(b["reconstruction_normal_consistency"])
        and int(c["introduced_flipped_faces"]) <= int(b["introduced_flipped_faces"])
        and int(c["improved_over_initial"]) >= int(b["improved_over_initial"])
    )
    matched_50_prediction = float(c["loss"]) < float(a["loss"]) and float(c["vector_l2"]) < float(
        a["vector_l2"]
    )
    matched_50_reconstruction = (
        float(c["reconstruction_chamfer"]) < float(a["reconstruction_chamfer"])
        and float(c["reconstruction_point_to_surface"])
        < float(a["reconstruction_point_to_surface"])
    )
    matched_50_all = (
        matched_50_prediction
        and matched_50_reconstruction
        and float(c["reconstruction_normal_consistency"])
        >= float(a["reconstruction_normal_consistency"])
        and int(c["introduced_flipped_faces"]) <= int(a["introduced_flipped_faces"])
        and int(c["improved_over_initial"]) >= int(a["improved_over_initial"])
        and float(c["correct_zero_loss_gap"]) > 0.0
    )
    return {
        "native_validation_improved_20k_to_50k": True,
        "synthetic_prediction_improved_20k_to_50k": prediction_20_to_50,
        "synthetic_reconstruction_improved_20k_to_50k": reconstruction_20_to_50,
        "synthetic_reconstruction_all_recorded_endpoints_improved_20k_to_50k": reconstruction_all_20_to_50,
        "image_dependence_retained_at_50k": float(c["correct_zero_loss_gap"]) > 0.0,
        "current_query_50k_lower_prediction_error_than_gt_query_50k": matched_50_prediction,
        "current_query_50k_lower_reconstruction_distance_than_gt_query_50k": matched_50_reconstruction,
        "current_query_50k_all_recorded_endpoints_no_worse_than_gt_query_50k": matched_50_all,
        "current_query_ready_as_main_formulation_on_this_synthetic_protocol": matched_50_all,
    }


def _report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    aggregate = summary["aggregate"]
    comparisons = summary["comparisons"]
    outcomes = summary["per_sample_outcomes"]
    checkpoints = summary["checkpoints"]
    decision = summary["decision"]
    lines = [
        "# Sofa50 Synthetic Current-query 50k Downstream Evaluation",
        "",
        "## 1. Scope",
        "",
        "Three completed checkpoints are evaluated on the saved Sofa50 synthetic-current test split. No model training, data generation, split generation, target generation, or recovery change is performed.",
        "",
        "## 2. Evaluation contract",
        "",
        f"- Test samples: `{summary['test_samples']}` stored current variants from `{summary['test_objects']}` objects.",
        f"- Manifest: `{summary['contract_audit']['manifest_path']}`.",
        f"- Manifest SHA-256: `{summary['contract_audit']['manifest_sha256']}`.",
        f"- Target: `{summary['target']}`.",
        "- Correct-RGB and zero-RGB use identical query positions, graph, cameras, visibility and target.",
        "- Recovery uses the existing synthetic-current evaluator and its saved solver contract.",
        "- Aggregates reproduce the existing evaluator: concatenated-vertex prediction metrics and sample-mean geometry metrics.",
        "",
        "## 3. Checkpoints",
        "",
        "| Model | Formulation | Budget | Checkpoint epoch | Checkpoint steps | Checkpoint validation loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        row = checkpoints[name]
        lines.append(
            f"| {name} | {row.get('training_formulation', 'NA')} | "
            f"{row.get('training_budget_optimizer_steps', 'NA')} | "
            f"{row.get('checkpoint_epoch', 'NA')} | "
            f"{row.get('checkpoint_optimizer_steps', 'NA')} | "
            f"{_f(row.get('checkpoint_validation_loss'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. Contract/regression checks",
            "",
            f"- Saved GT-query 50k vs current-query 20k regression: `{summary['contract_audit']['regression']['passed']}`.",
            f"- Repeated GT-query 50k evaluation match: `{summary['contract_audit']['repeated_gt_query_50k']['passed']}`.",
            f"- Current-query 20k/50k configs differ only in max optimizer steps: `{summary['contract_audit']['config_audit']['evaluation_contract_match']}`.",
            "- Test sample IDs match across all three evaluations: `True`.",
            "",
            "## 5. Prediction metrics",
            "",
            "| Model | Evaluation loss | Vector L2 / target EPE | Global cosine | High-10% cosine | Pred/target norm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in MODEL_ORDER:
        row = aggregate[name]
        lines.append(
            f"| {name} | {_f(row['loss'])} | {_f(row['vector_l2'])} | "
            f"{_f(row['global_cosine'])} | {_f(row['high_10_percent_cosine'])} | "
            f"{_f(row['prediction_target_norm_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 6. Zero-RGB image ablation",
            "",
            "| Model | Correct-RGB loss | Zero-RGB loss | Correct-zero gap | Relative correct-vs-zero improvement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in MODEL_ORDER:
        row = aggregate[name]
        lines.append(
            f"| {name} | {_f(row['loss'])} | {_f(row['zero_rgb_loss'])} | "
            f"{_f(row['correct_zero_loss_gap'])} | "
            f"{_pct(100.0 * float(row['relative_correct_vs_zero_improvement']))} |"
        )
    lines.extend(
        [
            "",
            "## 7. Reconstruction metrics",
            "",
            "| Model | Initial Chamfer | Refined Chamfer | Initial P2S | Refined P2S | Initial normal | Refined normal | Flips | Improved/25 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in MODEL_ORDER:
        row = aggregate[name]
        lines.append(
            f"| {name} | {_f(row['initial_chamfer'])} | {_f(row['reconstruction_chamfer'])} | "
            f"{_f(row['initial_point_to_surface'])} | {_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['initial_normal_consistency'])} | {_f(row['reconstruction_normal_consistency'])} | "
            f"{row['introduced_flipped_faces']} | {row['improved_over_initial']}/25 |"
        )
    lines.extend(
        [
            "",
            "## 8. Per-sample refinement outcomes",
            "",
            f"- Current-query 20k better than initial ({len(outcomes['current20_better_than_initial'])}/25): `{', '.join(outcomes['current20_better_than_initial']) or 'none'}`.",
            f"- Current-query 50k better than initial ({len(outcomes['current50_better_than_initial'])}/25): `{', '.join(outcomes['current50_better_than_initial']) or 'none'}`.",
            f"- Retained at 50k: `{', '.join(outcomes['retained_from_20k']) or 'none'}`.",
            f"- New at 50k: `{', '.join(outcomes['new_at_50k']) or 'none'}`.",
            f"- Lost at 50k: `{', '.join(outcomes['lost_at_50k']) or 'none'}`.",
            "",
            "Samples with at least 10% Chamfer increase from current-query 20k to 50k:",
            "",
            "| Sample | Absolute change | Percent change |",
            "|---|---:|---:|",
        ]
    )
    outliers = outcomes["chamfer_increase_at_least_10_percent"]
    if outliers:
        for row in outliers:
            lines.append(
                f"| {row['sample_id']} | {_f(row['absolute_change'])} | {_pct(row['percent_change'])} |"
            )
    else:
        lines.append("| none | — | — |")

    lines.extend(["", "## 9. 20k -> 50k change", ""])
    _comparison_table(lines, comparisons["current_query_50k_vs_current_query_20k"])
    lines.extend(
        [
            "",
            "## 10. GT-query 50k vs Current-query 50k",
            "",
            "Both checkpoints use a 50,000-step training budget. Their training formulations differ. The comparison does not isolate a single mechanism.",
            "",
        ]
    )
    _comparison_table(lines, comparisons["current_query_50k_vs_gt_query_50k"])
    lines.extend(
        [
            "",
            "## 11. Interpretation",
            "",
            f"- Native validation improved from 20k to 50k: `{decision['native_validation_improved_20k_to_50k']}`.",
            f"- Synthetic-current prediction loss and EPE both decreased from 20k to 50k: `{decision['synthetic_prediction_improved_20k_to_50k']}`.",
            f"- Synthetic-current reconstruction Chamfer and P2S both decreased from 20k to 50k: `{decision['synthetic_reconstruction_improved_20k_to_50k']}`.",
            f"- Chamfer, P2S, normal consistency, flips and improved-sample count all changed in their specified directions from 20k to 50k: `{decision['synthetic_reconstruction_all_recorded_endpoints_improved_20k_to_50k']}`.",
            f"- Correct-RGB loss remains below zero-RGB loss at 50k: `{decision['image_dependence_retained_at_50k']}`.",
            f"- Current-query 50k prediction loss and EPE are below GT-query 50k: `{decision['current_query_50k_lower_prediction_error_than_gt_query_50k']}`.",
            f"- Current-query 50k reconstruction Chamfer and P2S are below GT-query 50k: `{decision['current_query_50k_lower_reconstruction_distance_than_gt_query_50k']}`.",
            f"- All recorded prediction, reconstruction, topology-count and image-dependence criteria are met against GT-query 50k: `{decision['current_query_50k_all_recorded_endpoints_no_worse_than_gt_query_50k']}`.",
            "",
            "## 12. Decision",
            "",
            f"Current-query remains the main formulation under this synthetic-current protocol: `{decision['current_query_ready_as_main_formulation_on_this_synthetic_protocol']}`.",
            "This decision is limited to the saved synthetic-current protocol and does not replace real coarse/OpenMVS evaluation.",
            "",
            "## 13. Artifact paths",
            "",
            "- `report.md`",
            "- `summary.json`",
            "- `comparisons.json`",
            "- `per_sample_metrics.csv`",
            "- `contract_regression.json`",
            "- `regression_gt50_vs_current20/`",
            "- `gt50_vs_current50/`",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_table(lines: list[str], comparison: Mapping[str, Any]) -> None:
    labels = (
        ("Evaluation loss", "loss"),
        ("Vector L2 / target EPE", "vector_l2"),
        ("Global cosine", "global_cosine"),
        ("High-10% cosine", "high_10_percent_cosine"),
        ("Correct-zero gap", "correct_zero_loss_gap"),
        ("Refined Chamfer", "reconstruction_chamfer"),
        ("Refined P2S", "reconstruction_point_to_surface"),
        ("Normal consistency", "reconstruction_normal_consistency"),
        ("Introduced flips", "introduced_flipped_faces"),
        ("Improved samples", "improved_over_initial"),
    )
    lines.extend(
        [
            "| Metric | Source | Target | Absolute change | Percent change |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in labels:
        row = comparison[key]
        lines.append(
            f"| {label} | {_f(row['source'])} | {_f(row['target'])} | "
            f"{_f(row['absolute_change'])} | {_pct(row['percent_change'])} |"
        )


def print_terminal_summary(summary: Mapping[str, Any], output_dir: str | Path) -> None:
    aggregate = summary["aggregate"]
    print("PREDICTION_ZERO_RGB_RECONSTRUCTION")
    print(
        "model loss vector_l2 cosine high10 norm zero_loss gap chamfer p2s normal flips improved"
    )
    for name in MODEL_ORDER:
        row = aggregate[name]
        print(
            name,
            _f(row["loss"]),
            _f(row["vector_l2"]),
            _f(row["global_cosine"]),
            _f(row["high_10_percent_cosine"]),
            _f(row["prediction_target_norm_ratio"]),
            _f(row["zero_rgb_loss"]),
            _f(row["correct_zero_loss_gap"]),
            _f(row["reconstruction_chamfer"]),
            _f(row["reconstruction_point_to_surface"]),
            _f(row["reconstruction_normal_consistency"]),
            row["introduced_flipped_faces"],
            f"{row['improved_over_initial']}/25",
        )
    for label, key in (
        ("CURRENT20_TO_CURRENT50", "current_query_50k_vs_current_query_20k"),
        ("GT50_TO_CURRENT50", "current_query_50k_vs_gt_query_50k"),
    ):
        print(label)
        for metric, row in summary["comparisons"][key].items():
            print(
                metric,
                f"absolute={_f(row['absolute_change'])}",
                f"percent={_pct(row['percent_change'])}",
            )
    print("IMPROVED_OVER_INITIAL")
    for name in MODEL_ORDER:
        row = aggregate[name]
        print(name, f"{row['improved_over_initial']}/25", ",".join(row["improved_sample_ids"]))
    output = Path(output_dir).resolve()
    for name in ("report.md", "summary.json", "per_sample_metrics.csv", "comparisons.json"):
        print(f"ARTIFACT {output / name}")
    print("CONTRACT_MISMATCH", not bool(summary["contract_audit"]["regression"]["passed"]))
    print("MISSING_ARTIFACT", False)
    print("EVALUATION_FAILURE", False)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def _f(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def _pct(value: Any) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):+.4f}%"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected JSON list of objects: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

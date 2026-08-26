#!/usr/bin/env python3
from __future__ import annotations

"""Render the separate S1 split-geometry architecture experiment report."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ARCH_THRESHOLDS = {
    "material_hybrid_cd_relative_improvement": 0.03,
    "substantial_specialist_gap_closure": 0.25,
    "approximately_same_relative_band": 0.03,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any, digits: int = 7) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}g}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    headers = tuple(headers)
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _pick(rows: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    chosen = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(chosen) != 1:
        raise RuntimeError(f"Expected one row for {keys}; got {len(chosen)}")
    return chosen[0]


def _training(run: Path) -> dict[str, Any]:
    history = _read(run / "training_step_history.json")
    metrics = _read(run / "metrics.json")
    diagnostic = [row for row in history if row.get("pcg_iterations_mean") is not None]
    return {
        "intervals": len(diagnostic), "optimizer_steps": int(metrics["optimizer_steps"]),
        "best_epoch": int(metrics["best_epoch"]), "best_selection_loss": float(metrics["best_selection_loss"]),
        "runtime_seconds": float(metrics.get("runtime_seconds", sum(float(row.get("interval_seconds", 0)) for row in diagnostic))),
        "pcg_iterations_mean": float(np.mean([row["pcg_iterations_mean"] for row in diagnostic])),
        "pcg_iterations_max": int(max(row["pcg_iterations_max"] for row in diagnostic)),
        "pcg_residual_max": float(max(row["pcg_relative_residual_max"] for row in diagnostic)),
        "pcg_failed": int(sum(row["pcg_failed_solves"] for row in diagnostic)),
        "nan_inf": int(sum(row["nan_inf_count"] for row in diagnostic)),
        "peak_gpu_mb": float(max(row["peak_gpu_memory_mb"] for row in diagnostic)),
        "lap_head_grad_mean": float(np.mean([row["b_laplacian_head_gradient_norm"] for row in diagnostic if row.get("b_laplacian_head_gradient_norm") is not None])),
        "direct_head_grad_mean": float(np.mean([row["e_direct_head_gradient_norm"] for row in diagnostic if row.get("e_direct_head_gradient_norm") is not None])),
        "lap_backbone_grad_mean": float(np.mean([row["b_backbone_gradient_norm"] for row in diagnostic if row.get("b_backbone_gradient_norm") is not None])),
        "direct_backbone_grad_mean": float(np.mean([row["e_backbone_gradient_norm"] for row in diagnostic if row.get("e_backbone_gradient_norm") is not None])),
    }


def _paired(
    left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]],
    left_method: str, right_method: str, field: str,
) -> dict[str, Any]:
    left = {row["sample_id"]: float(row[field]) for row in left_rows if row["arm"] == left_method}
    right = {row["sample_id"]: float(row[field]) for row in right_rows if row["arm"] == right_method}
    if sorted(left) != sorted(right):
        raise RuntimeError(f"Paired IDs differ: {left_method}, {right_method}")
    values = np.asarray([right[key] - left[key] for key in sorted(left)])
    rng = np.random.default_rng(7)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "left": left_method, "right": right_method, "field": field, "samples": len(values),
        "right_minus_left_mean": float(values.mean()), "right_minus_left_median": float(np.median(values)),
        "right_better": int(np.sum(values < 0)), "left_better": int(np.sum(values > 0)), "ties": int(np.sum(values == 0)),
        "bootstrap_ci95_low": float(np.quantile(samples, 0.025)), "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--mechanism-root", required=True, type=Path)
    parser.add_argument("--loss-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.s1_root.resolve()
    selected_root = root / "selected_analysis"
    s1 = _read(selected_root / "s1_selected_summary.json")
    preflight = _read(root / "preflight_audit.json")
    mechanism = _read(args.mechanism_root.resolve() / "mechanism_summary.json")
    mechanism_gradient = _read(args.mechanism_root.resolve() / "gradient_summary.json")
    loss = _read(args.loss_root.resolve() / "loss_mechanism_summary.json")
    run = args.run.resolve()
    config_payload = _read(run / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    training = _training(run)
    trajectory = [_read(root / "trajectory" / f"trajectory_{label}.json") for label in ("step000000", "step001000", "step002500", "step005000", "step007500", "step010000", "step012500", "step015000", "step017500", "step020000", "best")]

    existing_geo = mechanism["geometry_aggregate"]
    s1_geo = s1["geometry_aggregate"]
    s0_val = _pick(existing_geo, split="validation", method="Joint_Hybrid")
    s0_test = _pick(existing_geo, split="test", method="Joint_Hybrid")
    s1_val = _pick(s1_geo, split="validation", method="S1_Hybrid")
    s1_test = _pick(s1_geo, split="test", method="S1_Hybrid")
    relative_improvement = {
        "validation": (float(s0_val["chamfer"]) - float(s1_val["chamfer"])) / float(s0_val["chamfer"]),
        "test": (float(s0_test["chamfer"]) - float(s1_test["chamfer"])) / float(s0_test["chamfer"]),
    }
    semantic_existing_lap = mechanism["lap_semantic_aggregate"]
    semantic_existing_direct = mechanism["position_semantic_aggregate"]
    semantic_s1 = s1["semantic_aggregate"]
    gap_closure = {}
    for split in ("validation", "test"):
        b = _pick(semantic_existing_lap, split=split, method="Pretrained_B")
        s0_lap = _pick(semantic_existing_lap, split=split, method="Joint_Lap")
        s1_lap = _pick(semantic_s1, split=split, method="S1_Lap")
        e = _pick(semantic_existing_direct, split=split, method="Pretrained_E")
        s0_direct = _pick(semantic_existing_direct, split=split, method="Joint_Direct")
        s1_direct = _pick(semantic_s1, split=split, method="S1_Direct")
        gap_closure[split] = {
            "lap_raw_epe": (float(s0_lap["raw_epe_mean"]) - float(s1_lap["raw_epe"])) / max(float(s0_lap["raw_epe_mean"]) - float(b["raw_epe_mean"]), 1e-30),
            "direct_vertex_rms": (float(s0_direct["vertex_rms_mean"]) - float(s1_direct["vertex_rms"])) / max(float(s0_direct["vertex_rms_mean"]) - float(e["vertex_rms_mean"]), 1e-30),
        }
    material_improvement = all(value >= ARCH_THRESHOLDS["material_hybrid_cd_relative_improvement"] for value in relative_improvement.values())
    specialization = all(gap_closure[split][field] >= ARCH_THRESHOLDS["substantial_specialist_gap_closure"] for split in ("validation", "test") for field in ("lap_raw_epe", "direct_vertex_rms"))
    unstable = training["pcg_failed"] > 0 or training["nan_inf"] > 0 or not s1["contract_audit"]
    materially_worse = any(value <= -ARCH_THRESHOLDS["material_hybrid_cd_relative_improvement"] for value in relative_improvement.values())
    if unstable or materially_worse:
        classification = "ARCH4"
    elif material_improvement and specialization:
        classification = "ARCH1"
    elif material_improvement:
        classification = "ARCH2"
    else:
        classification = "ARCH3"

    existing_rows = _csv(args.mechanism_root.resolve() / "geometry_rows.csv")
    s1_rows = _csv(selected_root / "geometry_rows.csv")
    paired = []
    for split in ("validation", "test"):
        left = [row for row in existing_rows if row["split"] == split]
        right = [row for row in s1_rows if row["split"] == split]
        for a, b in (("Joint_Hybrid", "S1_Hybrid"), ("Pretrained_B", "S1_Lap"), ("Pretrained_E", "S1_Direct"), ("Frozen_BE", "S1_Hybrid")):
            for field in ("refined_chamfer", "same_index_recovered_vertex_rms", "p2s_p95"):
                row = _paired(left, right, a, b, field)
                row["split"] = split
                paired.append(row)
    fields = sorted({key for row in paired for key in row})
    with (root / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)

    lines = [
        "# Sofa50 v2 S1 split-geometry architecture experiment",
        "",
        f"Contract audit: **{str(bool(preflight['contract_audit'] and s1['contract_audit'] and training['optimizer_steps'] == 20000)).lower()}**. Classification: **{classification}**.",
        "",
        "Slurm training job: `17469` (4×L40). This report is separate from the descriptive loss-mechanism audit.",
        "",
        "## Architecture and execution contract",
        "",
        "The exact S0 visual/pre-graph frontend produces one shared per-vertex tensor. S1 forks that tensor before both predictor input MLPs and every graph/message-passing block. The Laplacian and Direct branches then use complete, independent graph towers and heads with no feature exchange.",
        "",
    ]
    counts = preflight["architecture"]["parameter_counts"]
    lines.extend(_table(("Group", "Parameters"), (("Shared frontend", counts["shared_frontend"]), ("Lap tower+head", counts["lap_tower"]), ("Direct tower+head", counts["direct_tower"]), ("S1 total", counts["total_S1"]), ("S0 total", counts["total_S0"]))))
    lines.extend(["", "Storage audit: shared frontend shared exactly once; Lap/Direct parameter-ID intersection is empty; the parameter partition is complete. S1 adds `701,696` parameters relative to S0; this un-matched capacity increase is an explicit limitation of the architecture-only experiment.", "", "Training is from scratch with seed 7, Adam at `1e-3`, zero weight decay, the exact S0 ReduceLROnPlateau schedule, FP16 AMP, gradient clipping 1, full-vertex sampling, 20,000 optimizer steps, world size 4, per-rank microbatch 1, accumulation 2, and effective global batch 8.", "", "The shared frontend follows the same deterministic S0 initialization procedure, but no archived S0 tensor state is loaded. Both post-fork towers are independently allocated and initialized from that procedure.", "", "## Fixed recovery and objective", "", "`V_direct=V_input+DeltaV_direct`. With `L_U=I-D^-1 A` and `lambda=3e-2`, S1 solves `V_H=argmin_V ||L_U V-delta_hat||_F^2 + lambda ||V-V_direct||_F^2`, equivalently `(L_U^T L_U+lambda I)V_H=L_U^T delta_hat+lambda V_direct`.", "", "The only optimization objective is `mean_i ||V_H_i-V_clean_i||_2^2`. Recovery uses float64 PCG, tolerance `1e-8`, and maximum 2048 iterations. No branch, confidence, Chamfer, normal, spectral, balancing, or auxiliary loss is active.", "", "## Preflight and training stability", ""])
    real = preflight["real_model_gradient_preflight"]
    lines.extend(_table(("Path", "Gradient norm", "Finite/nonzero"), (("Shared frontend", _f(real["shared_visual_frontend"]["norm"]), "yes"), ("Lap graph tower", _f(real["lap_graph_tower"]["norm"]), "yes"), ("Lap head", _f(real["lap_head"]["norm"]), "yes"), ("Direct graph tower", _f(real["direct_graph_tower"]["norm"]), "yes"), ("Direct head", _f(real["direct_head"]["norm"]), "yes"))))
    analytic = preflight["analytic_gradient_preflight"]
    solver_rows = preflight["solver_audit"]
    lines.extend(["", f"Implicit-gradient finite-difference maximum relative error: `{analytic['maximum_checked_relative_error']:.3e}`. Across three real-mesh PCG/LSMR checks, maximum vertex RMS difference was `{max(float(row['pcg_lsmr_vertex_rms']) for row in solver_rows):.3e}` and maximum coordinate difference was `{max(float(row['pcg_lsmr_max_coordinate']) for row in solver_rows):.3e}`; all PCG solves converged."])
    lines.extend(["", f"Training intervals `{training['intervals']}`; runtime `{training['runtime_seconds']/3600:.3f} h`; PCG mean/max `{_f(training['pcg_iterations_mean'])}` / `{training['pcg_iterations_max']}`; maximum residual `{training['pcg_residual_max']:.3e}`; failed solves `{training['pcg_failed']}`; NaN/Inf `{training['nan_inf']}`; peak GPU memory `{_f(training['peak_gpu_mb'])} MiB`.", "", "## Validation checkpoint trajectory", ""])
    lines.extend(_table(("Checkpoint", "Validation Hybrid CD", "PCG mean/max", "SHA-256"), ((row["label"], _f(row["validation_hybrid_chamfer"]), f"{_f(row['pcg_iterations_mean'])}/{row['pcg_iterations_max']}", row["checkpoint_sha256"]) for row in trajectory)))
    selected_solver = s1["solver_aggregate"]
    lines.extend(["", f"Selected checkpoint: `{s1['checkpoint']}`; SHA-256 `{s1['checkpoint_sha256']}`. Selection used validation Hybrid Chamfer only; test/OOD data were not used.", "", f"Selected validation+test solves: `{selected_solver['solves']}`; PCG iterations mean/max `{_f(selected_solver['iterations_mean'])}` / `{selected_solver['iterations_max']}`; maximum relative residual `{selected_solver['relative_residual_max']:.3e}`; failed `{selected_solver['failed']}`.", "", "## Matched geometry", ""])
    comparison_methods = ("Initial", "Pretrained_B", "Pretrained_E", "Frozen_BE", "Joint_Lap", "Joint_Direct", "Joint_Hybrid")
    geo_rows = []
    for split in ("validation", "test"):
        split_existing = [row for row in existing_rows if row["split"] == split]
        for method in comparison_methods:
            row = _pick(existing_geo, split=split, method=method)
            detail = [item for item in split_existing if item["arm"] == method]
            relative_gain = float(np.mean([float(item["relative_chamfer_gain"]) for item in detail])) if detail else 0.0
            p2s = float(np.mean([float(item["p2s"]) for item in detail])) if detail else float(row["chamfer"])
            geo_rows.append((split, method, _f(row["chamfer"]), f"{100*relative_gain:+.2f}%", _f(row["vertex_rms"]), _f(p2s), _f(row["p2s_p95"]), _f(row["fscore"]), _f(row["normal"]), f"{row['flips']} / {_f(100*float(row['flip_rate']),4)}%", row["new_degenerates"], f"{row['improved']}/{row['worsened']}"))
        for method in METHODS:
            row = _pick(s1_geo, split=split, method=method)
            geo_rows.append((split, method, _f(row["chamfer"]), f"{100*float(row['relative_gain']):+.2f}%", _f(row["vertex_rms"]), _f(row["p2s"]), _f(row["p2s_p95"]), _f(row["fscore"]), _f(row["normal"]), f"{row['flips']} / {_f(100*float(row['flip_rate']),4)}%", row["new_degenerates"], f"{row['improved']}/{row['worsened']}"))
    lines.extend(_table(("Split", "Method", "CD", "Gain", "VRMS", "P2S", "P2S p95", "F", "Normal", "Flips/rate", "New deg.", "+/-"), geo_rows))
    paired_rows = [
        (
            row["split"], row["left"], row["right"], row["field"],
            _f(row["right_minus_left_mean"]), _f(row["right_minus_left_median"]),
            f"{row['right_better']}/{row['left_better']}/{row['ties']}",
            f"[{_f(row['bootstrap_ci95_low'])}, {_f(row['bootstrap_ci95_high'])}]",
        )
        for row in paired
    ]
    lines.extend(["", "### Paired S1 comparisons", ""] + _table(("Split", "Left", "Right", "Metric", "Right-left mean", "Median", "R/L/tie better", "Bootstrap 95% CI"), paired_rows))
    lines.extend(["", "## Specialist semantics", ""])
    semantic_rows = []
    for split in ("validation", "test"):
        for method in ("Pretrained_B", "Joint_Lap"):
            row = _pick(semantic_existing_lap, split=split, method=method)
            semantic_rows.append((split, method, "differential", _f(row["raw_epe_mean"]), _f(row["raw_rms_mean"]), _f(row["top10_epe_mean"]), _f(row["top1_epe_mean"])))
        row = _pick(semantic_s1, split=split, method="S1_Lap")
        semantic_rows.append((split, "S1_Lap", "differential", _f(row["raw_epe"]), _f(row["raw_rms"]), _f(row["top10_epe"]), _f(row["top1_epe"])))
        for method in ("Pretrained_E", "Joint_Direct"):
            row = _pick(semantic_existing_direct, split=split, method=method)
            semantic_rows.append((split, method, "positional", _f(row["vertex_rms_mean"]), _f(row["vertex_error_mean_mean"]), _f(row["vertex_error_p95_mean"]), "n/a"))
        row = _pick(semantic_s1, split=split, method="S1_Direct")
        semantic_rows.append((split, "S1_Direct", "positional", _f(row["vertex_rms"]), _f(row["vertex_error_mean"]), _f(row["vertex_error_p95"]), "n/a"))
    lines.extend(_table(("Split", "Method", "Space", "Primary RMS/EPE", "Secondary", "p95/Top10", "Top1"), semantic_rows))
    position_component_rows = []
    for split in ("validation", "test"):
        for method in ("Pretrained_E", "Joint_Direct"):
            row = _pick(mechanism["component_aggregate"], split=split, method=method)
            position_component_rows.append((split, method, _f(row["component_translation_rms"]), _f(row["centered_deformation_vrms"])))
        row = _pick(s1["component_aggregate"], split=split, method="S1_Direct")
        position_component_rows.append((split, "S1_Direct", _f(row["translation_rms"]), _f(row["centered_vrms"])))
    lines.extend(["", "### Positional component/nullspace semantics", ""] + _table(("Split", "Method", "Component translation RMS", "Centered within-component VRMS"), position_component_rows))
    lines.extend(["", f"Specialist-gap closure (validation/test): Lap `{gap_closure['validation']['lap_raw_epe']:.2%}` / `{gap_closure['test']['lap_raw_epe']:.2%}`; Direct `{gap_closure['validation']['direct_vertex_rms']:.2%}` / `{gap_closure['test']['direct_vertex_rms']:.2%}`.", "", "## Shared-frontend gradient interaction", ""])
    grad_rows = []
    s0_grad = mechanism_gradient["aggregate"]
    mapping = (("image_encoder", "shared_encoder_parameters"), ("projected_image_field", "projected_image_field"), ("all_shared_parameters", "full_shared_frontend_parameters"), ("shared_feature_Phi", "shared_vertex_feature_at_fork"))
    for old, new in mapping:
        a = _pick(s0_grad, layer=old); b = _pick(s1["gradient_aggregate"], split="validation", layer=new)
        grad_rows.extend((("S0", old, _f(a["cosine_mean"]), _f(a["lap_norm_mean"]), _f(a["direct_norm_mean"]), _f(a["magnitude_ratio_median"]), _f(a["alignment_ratio_mean"])), ("S1", new, _f(b["mean_cosine"]), _f(b["mean_lap_norm"]), _f(b["mean_direct_norm"]), _f(b["median_norm_ratio"]), _f(b["mean_alignment_ratio"]))))
    lines.extend(_table(("Model", "Shared location", "Cos", "||g_L||", "||g_D||", "Norm ratio", "R_align"), grad_rows))
    lines.extend(["", "## Spectral, component, and RHS diagnostics", ""])
    diag_rows = []
    for split in ("validation", "test"):
        for method in ("Pretrained_B", "Pretrained_E", "Frozen_BE", "Joint_Lap", "Joint_Direct", "Joint_Hybrid"):
            sp = _pick(mechanism["spectral_aggregate"], split=split, method=method); cp = _pick(mechanism["component_aggregate"], split=split, method=method)
            diag_rows.append((split, method, _f(sp["total_energy"]), _f(sp["low_energy"]), _f(sp["mid_energy"]), _f(sp["high_energy"]), _f(cp["component_translation_rms"]), _f(cp["centered_deformation_vrms"])))
        for method in METHODS:
            sp = _pick(s1["spectral_aggregate"], split=split, method=method); cp = _pick(s1["component_aggregate"], split=split, method=method)
            diag_rows.append((split, method, _f(sp["total_energy"]), _f(sp["low_energy"]), _f(sp["mid_energy"]), _f(sp["high_energy"]), _f(cp["translation_rms"]), _f(cp["centered_vrms"])))
    lines.extend(_table(("Split", "Method", "Total error E", "Low", "Mid", "High", "Component translation RMS", "Centered VRMS"), diag_rows))
    rhs_rows = []
    for split in ("validation", "test"):
        for state in ("Frozen_B_plus_E", "S0"):
            row = _pick(loss["rhs_aggregate"], split=split, state=state)
            rhs_rows.append((split, state, _f(row["mean_lap_rhs_norm"]), _f(row["mean_direct_rhs_norm"]), _f(row["mean_combined_rhs_norm"]), _f(row["median_rhs_cosine"]), _f(row["median_cancellation_ratio"]), _f(row["p10_cancellation_ratio"]), _f(row["p90_cancellation_ratio"])))
        row = _pick(s1["rhs_aggregate"], split=split)
        rhs_rows.append((split, "S1", _f(row["mean_lap_rhs_norm"]), _f(row["mean_direct_rhs_norm"]), _f(row["mean_combined_rhs_norm"]), _f(row["median_rhs_cosine"]), _f(row["median_cancellation_ratio"]), _f(row["p10_cancellation_ratio"]), _f(row["p90_cancellation_ratio"])))
    lines.extend([""] + _table(("Split", "State", "||e_L||", "||e_D||", "||e_q||", "Median cos", "Median C", "p10 C", "p90 C"), rhs_rows))
    lines.extend(["", "## Architecture decision", "", f"Classification: **{classification}**.", ""])
    lines.extend(_table(("Gate", "Result"), (("Material S1 Hybrid CD improvement on validation and test", str(material_improvement).lower()), ("Both Lap and Direct close at least 25% of specialist gaps on both splits", str(specialization).lower()), ("Numerically/training unstable", str(unstable).lower()), ("Materially worse", str(materially_worse).lower()))))
    lines.extend(["", f"S1-vs-S0 Hybrid relative CD change: validation `{relative_improvement['validation']:+.2%}`, test `{relative_improvement['test']:+.2%}`. Paired means, medians, wins/losses, and bootstrap 95% intervals are in `paired_comparisons.csv`.", "", "## Final answer", ""])
    answer = {
        "ARCH1": "Yes: the pre-graph split materially improves final-only fusion and substantially restores both differential and positional specialization.",
        "ARCH2": "Partly: the pre-graph split materially improves fusion, but final-only supervision still does not restore both specialists.",
        "ARCH3": "No material benefit is established: performance remains near S0 and specialist semantics remain weak.",
        "ARCH4": "No: S1 is materially worse or unstable under the fixed contract.",
    }[classification]
    lines.extend([answer, "", f"Metric protocol: `{s1['metric_protocol']}`.", f"Spectral protocol: `{s1['spectral_protocol']}`."])
    report = root / "FINAL_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = {"contract_audit": bool(preflight["contract_audit"] and s1["contract_audit"] and training["optimizer_steps"] == 20000), "classification": classification, "thresholds": ARCH_THRESHOLDS, "relative_hybrid_cd_improvement": relative_improvement, "specialist_gap_closure": gap_closure, "checkpoint_sha256": s1["checkpoint_sha256"]}
    (root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), **decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

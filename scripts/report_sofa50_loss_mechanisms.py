#!/usr/bin/env python3
from __future__ import annotations

"""Render the separate Sofa50 v2 loss-mechanism audit report."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:.{digits}g}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    headers = tuple(headers)
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _pick(rows: list[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {keys}, got {len(selected)}")
    return selected[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--mechanism-report", required=True, type=Path)
    args = parser.parse_args()
    root = args.audit_dir.resolve()
    summary = _read(root / "loss_mechanism_summary.json")
    transfer = _read(root / "transfer_profile_summary.json")
    evolution = [
        _read(root / "evolution" / f"loss_evolution_{label}.json")
        for label in ("step005000", "step010000", "step015000", "step020000", "best")
    ]
    established = _read(args.mechanism_report.resolve() / "mechanism_summary.json")
    gradient = summary["gradient_aggregate"]
    same = summary["same_state_aggregate"]
    positional_same = summary["positional_same_state_aggregate"]
    correlations = summary["correlation_aggregate"]
    rhs = summary["rhs_aggregate"]
    thresholds = summary["decision_thresholds"]

    b_same = _pick(same, split="validation", state="B_state")
    band_shift = max(
        abs(float(b_same[f"{band}_fraction_direct"]) - float(b_same[f"{band}_fraction_recovery"]))
        for band in ("low", "mid", "high")
    )
    direct_corr = _pick(
        correlations, split="validation", path="g_B_lap_delta",
        feature="final_recovered_geometry_error",
    )
    recovery_corr = _pick(
        correlations, split="validation", path="g_B_vertex_delta",
        feature="final_recovered_geometry_error",
    )
    loss1 = (
        float(b_same["mean_cosine"]) <= thresholds["material_gradient_cosine"]
        and band_shift >= thresholds["material_band_fraction_points"]
        and float(recovery_corr["mean_spearman"]) - float(direct_corr["mean_spearman"])
        >= thresholds["geometry_correlation_advantage"]
    )
    s0_rhs = [_pick(rhs, split=split, state="S0") for split in ("validation", "test")]
    loss2 = all(
        float(row["median_rhs_cosine"]) <= thresholds["strong_cancellation_cosine"]
        and float(row["median_cancellation_ratio"]) <= thresholds["strong_cancellation_ratio"]
        for row in s0_rhs
    )
    s0_norm_ratios = []
    for split in ("validation", "test"):
        a = _pick(gradient, split=split, path="g_S0_lap_delta")
        b = _pick(gradient, split=split, path="g_S0_direct_V")
        s0_norm_ratios.append(max(float(a["gradient_norm"]), float(b["gradient_norm"])) / max(min(float(a["gradient_norm"]), float(b["gradient_norm"])), 1e-30))
    loss3 = all(value >= thresholds["strong_path_norm_ratio"] for value in s0_norm_ratios)
    active = sum((loss1, loss2, loss3))
    if active >= 2:
        classification = "LOSS4"
    elif loss1:
        classification = "LOSS1"
    elif loss2:
        classification = "LOSS2"
    elif loss3:
        classification = "LOSS3"
    else:
        classification = "LOSS5"

    lines = [
        "# Sofa50 v2 loss-mechanism audit",
        "",
        f"Contract audit: **{str(bool(summary['contract_audit'] and transfer['contract_audit'] and all(item['all_finite'] for item in evolution))).lower()}**. Classification: **{classification}**.",
        "",
        "This is a strictly read-only output-space and exact-operator audit of existing A/B/E/S0 checkpoints and archives. It trained no model and did not use S1 results. S1 remains a separate architecture experiment.",
        "",
        "## Historical objectives reproduced",
        "",
        "- A: unit-weight Huber raw-Laplacian loss, delta `0.01`.",
        "- B: the same Huber term plus `1e-2 * L_vertex`, with historical `lambda=1e-2` recovery.",
        "- E: `mean_i ||V_direct_i-V_clean_i||_2^2`.",
        "- S0: only `mean_i ||V_H_i-V_clean_i||_2^2`, with `lambda=3e-2` hybrid recovery.",
        "",
        "Every reported gradient is with respect to an output field at the same frozen state; cross-method parameter gradients are not compared as though they occupied the same parameterization.",
        "",
        "## Exact checkpoint identities",
        "",
    ]
    lines.extend(_table(("Method", "Checkpoint", "SHA-256"), ((row["method"], row["checkpoint"], row["checkpoint_sha256"]) for row in summary["checkpoint_identities"])))
    lines.extend([
        "", "## Exact output-space derivatives", "",
        "For A, `g_A_delta` is autograd through the archived unit-weight Huber loss (`delta=0.01`) using the archived float32 target tensor. For B, `g_B_lap_delta=dL_lap/ddelta`, `g_B_vertex_delta=d(beta L_vertex)/ddelta`, and `g_B_total_delta` is their exact sum.",
        "",
        "For `A=L_U^T L_U+lambda I`, output geometry gradient `g_V`, and `z=A^-1 g_V`, implicit differentiation gives `dL/ddelta=L_U z` and `dL/dV_anchor=lambda z`. E uses `g_E_V=2(V_direct-V_clean)/N`. S0 applies the two implicit formulas separately at `lambda=3e-2`. Main gradients use autograd/custom implicit solves; centered finite differences are restricted to a verification subset.",
        "", "## Output-gradient absolute energy and graph-frequency allocation", "",
    ])
    grad_rows = []
    for split in ("validation", "test"):
        for path in ("g_A_delta", "g_B_lap_delta", "g_B_vertex_delta", "g_B_total_delta", "g_E_V", "g_S0_lap_delta", "g_S0_direct_V"):
            row = _pick(gradient, split=split, path=path)
            grad_rows.append((split, path, _f(row["total_energy"]), _f(row["low_energy"]), _f(row["mid_energy"]), _f(row["high_energy"]), _f(row["low_fraction"]), _f(row["mid_fraction"]), _f(row["high_fraction"])))
    lines.extend(_table(("Split", "Path", "Total", "Low", "Mid", "High", "Low frac.", "Mid frac.", "High frac."), grad_rows))
    lines.extend(["", "Absolute energy and norm bootstrap intervals are in `gradient_aggregate.csv`; normalized fractions are secondary.", "", "## Same-state direct versus recovery supervision", ""])
    same_rows = []
    for row in same:
        same_rows.append((row["split"], row["state"], _f(row["mean_cosine"]), _f(row["median_cosine"]), _f(row["mean_direct_norm"]), _f(row["mean_recovery_norm"]), _f(row["mean_norm_ratio_recovery_over_direct"]), _f(row["low_fraction_direct"]), _f(row["low_fraction_recovery"]), _f(row["high_fraction_direct"]), _f(row["high_fraction_recovery"])))
    lines.extend(_table(("Split", "Frozen state", "Mean cos", "Median cos", "||g_direct||", "||g_recovery||", "Recovery/direct norm", "Direct low", "Recovery low", "Direct high", "Recovery high"), same_rows))
    lines.extend(["", "This counterfactual holds `delta_hat` fixed and changes only the scalar supervision: direct Huber versus the B-style recovered-vertex objective.", "", "## Same-state direct vertex versus hybrid positional supervision", ""])
    positional_rows = []
    for row in positional_same:
        positional_rows.append((row["split"], row["state"], _f(row["mean_cosine"]), _f(row["mean_norm_ratio_hybrid_over_direct"]), _f(row["mean_direct_vertex_gradient_mean"]), _f(row["mean_hybrid_vertex_gradient_mean"]), _f(row["mean_direct_vertex_gradient_median"]), _f(row["mean_hybrid_vertex_gradient_median"]), _f(row["mean_direct_vertex_gradient_p95"]), _f(row["mean_hybrid_vertex_gradient_p95"])))
    lines.extend(_table(("Split", "State", "Mean cos", "Hybrid/direct norm", "Direct mean", "Hybrid mean", "Direct median", "Hybrid median", "Direct p95", "Hybrid p95"), positional_rows))
    lines.extend(["", "Each row holds the exact same positional tensor fixed while changing only direct MSE versus solver-mediated final geometry supervision.", "", "## Gradient localization", ""])
    corr_rows = []
    for split in ("validation", "test"):
        for path in ("g_A_delta", "g_B_lap_delta", "g_B_vertex_delta", "g_B_total_delta", "g_E_V", "g_S0_lap_delta", "g_S0_direct_V"):
            row = _pick(correlations, split=split, path=path, feature="final_recovered_geometry_error")
            corr_rows.append((split, path, _f(row["mean_pearson"]), _f(row["mean_spearman"]), _f(row["mean_top10_gradient_energy_fraction"]), _f(row["mean_top1_gradient_energy_fraction"])))
    lines.extend(_table(("Split", "Path", "Pearson", "Spearman", "Top-error 10% grad energy", "Top-error 1% grad energy"), corr_rows))
    top_rows = []
    for split in ("validation", "test"):
        for path in ("g_A_delta", "g_B_lap_delta", "g_B_vertex_delta", "g_B_total_delta", "g_E_V", "g_S0_lap_delta", "g_S0_direct_V"):
            for feature in ("raw_laplacian_error", "same_index_vertex_error", "gt_differential_magnitude", "final_recovered_geometry_error"):
                row = _pick(correlations, split=split, path=path, feature=feature)
                top_rows.append((split, path, feature, _f(row["mean_feature_all_vertices"]), _f(row["mean_feature_on_top10_gradient_vertices"]), _f(row["mean_feature_on_top1_gradient_vertices"])))
    lines.extend(["", "### Feature magnitude on top-gradient vertices", ""] + _table(("Split", "Path", "Feature", "All vertices", "Top-grad 10%", "Top-grad 1%"), top_rows))
    lines.extend(["", "Correlations are descriptive and are not interpreted causally.", "", "## Hybrid RHS interaction", ""])
    rhs_rows = []
    for row in rhs:
        rhs_rows.append((row["split"], row["state"], _f(row["mean_lap_rhs_norm"]), _f(row["mean_direct_rhs_norm"]), _f(row["mean_combined_rhs_norm"]), _f(row["mean_rhs_cosine"]), _f(row["median_rhs_cosine"]), _f(row["mean_cancellation_ratio"]), _f(row["median_cancellation_ratio"]), _f(row["p10_cancellation_ratio"]), _f(row["p90_cancellation_ratio"])))
    lines.extend(_table(("Split", "State", "||e_L||", "||e_D||", "||e_q||", "Mean cos", "Median cos", "Mean C", "Median C", "p10 C", "p90 C"), rhs_rows))
    lines.extend(["", "Here `e_L=L_U^T(delta_hat-delta*)`, `e_D=lambda(V_direct-V_clean)`, and `C_cancel=||e_L+e_D||/(||e_L||+||e_D||)`. GT is used only for this diagnostic.", "", "## Empirical exact-operator transfer", ""])
    transfer_rows = [(row["branch"], row["band"], row["measurements"], _f(row["mean_gain"]), _f(row["standard_deviation_gain"]), _f(row["median_gain"]), f"[{_f(row['minimum_gain'])}, {_f(row['maximum_gain'])}]") for row in transfer["aggregate"]]
    lines.extend(_table(("Map", "Band", "n", "Mean gain", "Std", "Median gain", "Range"), transfer_rows))
    lines.extend(["", "The probes use the exact non-symmetric random-walk operator and its true transpose. `S_delta=(L_U^T L_U+lambda I)^-1 L_U^T`; `S_direct=lambda(L_U^T L_U+lambda I)^-1`. No symmetry shortcut is used.", "", "## S0 checkpoint evolution", ""])
    evolution_rows = []
    for item in evolution:
        lap = _pick(item["spectral_aggregate"], path="g_S0_lap_delta")
        direct = _pick(item["spectral_aggregate"], path="g_S0_direct_V")
        shared = _pick(item["shared_gradient_aggregate"], layer="all_shared_parameters")
        rhs_item = item["rhs_aggregate"]
        evolution_rows.append((item["label"], _f(lap["mean_gradient_norm"]), _f(direct["mean_gradient_norm"]), _f(shared["mean_cosine"]), _f(shared["median_magnitude_ratio"]), _f(rhs_item["mean_lap_rhs_norm"]), _f(rhs_item["mean_direct_rhs_norm"]), _f(rhs_item["mean_combined_rhs_norm"]), _f(rhs_item["median_rhs_cosine"]), _f(rhs_item["median_cancellation_ratio"])))
    lines.extend(_table(("Checkpoint", "Lap grad", "Direct grad", "Shared cos", "Norm ratio", "||e_L||", "||e_D||", "||e_q||", "RHS median cos", "Median C"), evolution_rows))
    evolution_spectral_rows = []
    for item in evolution:
        for path in ("g_S0_lap_delta", "g_S0_direct_V"):
            row = _pick(item["spectral_aggregate"], path=path)
            evolution_spectral_rows.append((item["label"], path, _f(row["mean_total_energy"]), _f(row["mean_low_energy"]), _f(row["mean_mid_energy"]), _f(row["mean_high_energy"])))
    lines.extend(["", "### S0 checkpoint output-gradient absolute energy", ""] + _table(("Checkpoint", "Path", "Total", "Low", "Mid", "High"), evolution_spectral_rows))
    mechanism_evolution = {
        path.stem.removeprefix("evolution_"): _read(path)
        for path in args.mechanism_report.resolve().glob("evolution_*.json")
    }
    semantic_rows = []
    for label in ("step005000", "step010000", "step015000", "step020000", "best"):
        item = mechanism_evolution[label]
        lap_semantic = item["lap_semantic_aggregate"]
        direct_semantic = item["direct_semantic_aggregate"]
        hybrid = _pick(item["geometry_aggregate"], method="Joint_Hybrid")
        semantic_rows.append((label, _f(lap_semantic["raw_epe"]), _f(direct_semantic["vertex_rms"]), _f(hybrid["chamfer"])))
    lines.extend([""] + _table(("Checkpoint", "Lap raw EPE", "Direct VRMS", "Hybrid CD"), semantic_rows))
    lines.extend(["", "All evolution rows use validation indices `0,5,...,45`; no missing checkpoint was reconstructed.", "", "## Paired bootstrap statistics", ""])
    paired_rows = [
        (
            row["split"], row["left"], row["right"], row["field"],
            _f(row["mean_left_minus_right"]), _f(row["median_left_minus_right"]),
            f"{row['left_lower']}/{row['right_lower']}/{row['ties']}",
            f"[{_f(row['bootstrap_ci95_low'])}, {_f(row['bootstrap_ci95_high'])}]",
        )
        for row in summary["paired_gradient_statistics"]
    ]
    lines.extend(_table(("Split", "Left", "Right", "Field", "Mean L-R", "Median L-R", "L/R/tie lower", "Bootstrap 95% CI"), paired_rows))
    lines.extend(["", "## Decision", "", f"Classification: **{classification}**.", ""])
    decision_rows = [
        ("LOSS1: direct/recovery supervision geometry differs", str(loss1).lower()),
        ("LOSS2: destructive hybrid cancellation", str(loss2).lower()),
        ("LOSS3: one hybrid path dominates", str(loss3).lower()),
    ]
    lines.extend(_table(("Predeclared gate", "Result"), decision_rows))
    answer = {
        "LOSS1": "Direct prediction-space and recovery-aware supervision emphasize materially different update directions and graph-frequency allocations.",
        "LOSS2": "The strongest isolated mechanism is compensatory final-only hybrid coding: Lap and Direct RHS errors oppose one another and reduce the combined residual.",
        "LOSS3": "The strongest isolated mechanism is persistent pathway-gradient magnitude imbalance.",
        "LOSS4": "Multiple mechanisms coexist: loss geometry, compensatory RHS coding, and/or pathway magnitude imbalance jointly explain the observed final-only behavior.",
        "LOSS5": "The measured loss directions and hybrid interactions do not isolate one strong mechanism under the predeclared gates.",
    }[classification]
    lines.extend(["", "`LOSS4` is assigned when at least two of LOSS1--LOSS3 are simultaneously supported; `LOSS5` means none pass. Thresholds and all gate inputs are stored in `loss_mechanism_summary.json`.", "", f"Finite-difference maximum relative error: `{summary['maximum_finite_difference_relative_error']:.6g}`.", "", "## Final answer", "", answer, "", "This loss audit is descriptive and read-only. It does not convert its mechanism associations into an architecture-causal claim.", ""])
    target = root / "FINAL_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    decision = {
        "contract_audit": bool(summary["contract_audit"] and transfer["contract_audit"] and all(item["all_finite"] for item in evolution)),
        "classification": classification,
        "gates": {"LOSS1": loss1, "LOSS2": loss2, "LOSS3": loss3},
        "thresholds": thresholds,
        "established_mechanism_contract_audit": established["contract_audit"],
    }
    (root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target), **decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

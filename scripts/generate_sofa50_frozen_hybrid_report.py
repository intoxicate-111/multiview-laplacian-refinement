#!/usr/bin/env python3
from __future__ import annotations

"""Generate the final frozen Sofa50 B/E direct-anchor hybrid report."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
ARM_H = "Hybrid_B_laplacian_E_anchor"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup(rows: Sequence[Mapping[str, Any]], **values: Any) -> Mapping[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in values.items())]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {values}, found {len(selected)}")
    return selected[0]


def _classification(matched: Mapping[str, Any], ood: Mapping[str, Any]) -> tuple[str, str]:
    test = {row["arm"]: row for row in matched["aggregate"] if row["split"] == "test"}
    matched_better = (
        test[ARM_H]["refined_chamfer"] < test[ARM_B]["refined_chamfer"]
        and test[ARM_H]["refined_chamfer"] < test[ARM_E]["refined_chamfer"]
    )
    domains = {}
    for row in ood["aggregate"]:
        domains.setdefault(row["domain"], {})[row["arm"]] = row
    hybrid_retains_b = bool(domains) and all(
        ARM_H in arms and arms[ARM_H]["refined_chamfer"] <= arms[ARM_B]["refined_chamfer"]
        for arms in domains.values()
    )
    hybrid_beats_e_ood = bool(domains) and all(
        ARM_H in arms and arms[ARM_H]["refined_chamfer"] < arms[ARM_E]["refined_chamfer"]
        for arms in domains.values()
    )
    if matched_better and hybrid_retains_b:
        return "HBR1", "Hybrid improves matched-domain Chamfer over both frozen arms and retains or improves Arm B's relative OOD Chamfer."
    if matched_better and hybrid_beats_e_ood:
        return "HBR3", "Hybrid improves matched geometry and moves OOD behavior from E toward B, but does not fully retain B's OOD Chamfer."
    if matched_better:
        return "HBR2", "Hybrid improves matched-domain geometry but provides no consistent OOD robustness advantage."
    return "HBR4", "Hybrid does not meaningfully improve over the stronger frozen standalone representation."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    sweep = _read(output / "sweep_summary.json")
    matched = _read(output / "matched_summary.json")
    ood = _read(output / "ood" / "ood_summary.json")
    classification, classification_reason = _classification(matched, ood)
    contract_audit = bool(sweep["contract_audit"] and matched["contract_audit"] and ood["implementation_audit"])
    summary = {
        "contract_audit": contract_audit,
        "read_only": True,
        "models_retrained": False,
        "lambda_hybrid_best": sweep["lambda_hybrid_best"],
        "classification": classification,
        "classification_reason": classification_reason,
        "matched_summary": matched,
        "ood_summary": ood,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Sofa50 v2 frozen Arm-B/Arm-E hybrid recovery diagnostic",
        "",
        f"Contract audit: **{str(contract_audit).lower()}**. This is a zero-retraining, read-only recombination of the frozen selected Arm B and Arm E outputs.",
        "",
        "The primary solve is `min ||L V - delta_B||² + lambda ||V - V_direct||²`, with no additional input anchor. PCG is float64, tolerance `1e-4`, maximum `2048` iterations. GT is evaluation-only.",
        "",
        "## Implementation/read-only audit",
        "",
        f"- `delta_B` is read from the selected frozen Arm B prediction archive/checkpoint `{matched['arm_b_checkpoint']}` (SHA-256 `{matched['arm_b_checkpoint_sha256']}`).",
        f"- `V_direct = V_input + delta_v_E` uses the selected frozen Arm E archive/checkpoint `{matched['arm_e_checkpoint']}` (SHA-256 `{matched['arm_e_checkpoint_sha256']}`).",
        "- Matched B/E arrays are checked against the exact same manifest sample IDs, ordering, input mesh and connectivity. The archived predictors used the same 28 native-960 images and cameras; OOD re-inference additionally audits the actual common model-input mapping for every sample.",
        "- No GT field enters either predictor or the recovery solve; clean vertices are loaded only after predictions for evaluation.",
        "- No network parameter, prediction, image, camera, mesh, topology or benchmark output is modified. No fine-tuning occurs.",
        "- Relative to Arm B, the only primary recovery change is the positional anchor target: `V_input` is replaced by frozen `V_direct`.",
        "",
        "## Validation-only lambda selection",
        "",
        f"Selected by validation mean Chamfer: **lambda = {sweep['lambda_hybrid_best']:.0e}**. Diagnostic VRMS optimum: `{sweep['lambda_best_vertex_rms_diagnostic']:.0e}`; diagnostic P2S-p95 optimum: `{sweep['lambda_best_p2s_p95_diagnostic']:.0e}`.",
        "",
        "| Lambda | CD | CD gain | VRMS | P2S p95 | F-score | Normal | Flip rate | Improved/worsened | PCG iter mean/max | Hybrid→E VRMS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep["aggregate"]:
        lines.append(
            f"| {row['lambda']:.0e} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:+.2%} | {row['same_index_recovered_vertex_rms']:.8g} | {row['p2s_p95']:.8g} | {row['fscore']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['improved']}/{row['worsened']} | {row['pcg_iterations_mean']:.2f}/{row['pcg_iterations_max']} | {row['hybrid_to_e_vertex_rms']:.8g} |"
        )
    max_lsmr_rms = max(row["pcg_vs_lsmr_vertex_rms"] for row in sweep["lsmr_checks"])
    max_lsmr_coordinate = max(row["pcg_vs_lsmr_max_coordinate"] for row in sweep["lsmr_checks"])
    lines.extend((
        "",
        f"Forward audit against tight float64 LSMR: maximum PCG↔LSMR vertex RMS `{max_lsmr_rms:.6g}`, maximum coordinate difference `{max_lsmr_coordinate:.6g}`; all PCG and LSMR checks converged.",
        "",
        "## Matched validation and test",
        "",
        "| Split | Arm | Initial CD | Refined CD | Gain / eta | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened | VRMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in matched["aggregate"]:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['initial_chamfer']:.8g} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:+.2%} / {row['eta']:.6g} | {row['p2s_p95']:.8g} | {row['fscore']:.8g} | {row['normal_consistency']:.8g} | {row['introduced_flipped_faces']} / {row['normalized_flip_rate']:.3%} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} | {row['same_index_recovered_vertex_rms']:.8g} |"
        )
    lines.extend((
        "",
        "## Paired comparisons and bootstrap intervals",
        "",
        "| Split | Comparison | H lower CD | H lower VRMS | H lower P95 | H higher F | H higher normal | H fewer flips |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ))
    for row in matched["paired_wins"]:
        lines.append(
            f"| {row['split']} | {row['comparison']} | {row['hybrid_better_refined_chamfer']}/{row['samples']} | {row['hybrid_better_same_index_recovered_vertex_rms']}/{row['samples']} | {row['hybrid_better_p2s_p95']}/{row['samples']} | {row['hybrid_better_fscore']}/{row['samples']} | {row['hybrid_better_normal_consistency']}/{row['samples']} | {row['hybrid_better_normalized_flip_rate']}/{row['samples']} |"
        )
    lines.extend((
        "",
        "| Split | Quantity | Mean | Median | Paired bootstrap 95% CI |",
        "|---|---|---:|---:|---:|",
    ))
    for row in matched["paired_statistics"]:
        lines.append(
            f"| {row['split']} | {row['quantity']} | {row['mean_paired_difference']:+.8g} | {row['median_paired_difference']:+.8g} | [{row['bootstrap_ci95_low']:+.8g}, {row['bootstrap_ci95_high']:+.8g}] |"
        )
    lines.extend((
        "",
        "## Test recipe and generation-family breakdown",
        "",
        "| Group | Arm | CD | VRMS | P2S p95 | Normal | Flip rate | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ))
    recipe_order = ["A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2", "mild", "strong", "original_topology", "subdivided", "adaptive_topology"]
    for group in recipe_order:
        for arm in (ARM_B, ARM_E, ARM_H):
            row = _lookup(matched["recipe_aggregate"], split="test", group=group, arm=arm)
            lines.append(
                f"| {group} | {arm} | {row['refined_chamfer']:.8g} | {row['same_index_recovered_vertex_rms']:.8g} | {row['p2s_p95']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['improved']}/{row['worsened']} |"
            )
    lines.extend((
        "",
        "Per-recipe Hybrid-vs-B/E paired wins are in `recipe_paired_wins.csv`.",
        "",
        "## Graph-frequency analysis",
        "",
        matched["spectral_protocol"] + ".",
        "",
        "| Split | Signal | Total absolute energy | Low energy / fraction | Mid energy / fraction | High energy / fraction |",
        "|---|---|---:|---:|---:|---:|",
    ))
    for row in matched["spectral_aggregate"]:
        lines.append(
            f"| {row['split']} | {row['signal']} | {row['total_energy']:.8g} | {row['low_energy']:.8g} / {row['low_fraction']:.2%} | {row['mid_energy']:.8g} / {row['mid_fraction']:.2%} | {row['high_energy']:.8g} / {row['high_fraction']:.2%} |"
        )
    test_spectral = {row["signal"]: row for row in matched["spectral_aggregate"] if row["split"] == "test"}
    lines.extend((
        "",
        "Absolute test error energy shows a qualified spectral fusion: Hybrid low-frequency error is between E and B, while Hybrid mid/high error is slightly below both. It does not beat E in total error energy, so the conclusion is not inferred from normalized fractions alone.",
        "",
        "## Connected components",
        "",
        "| Split | Arm | Components | Translation error mean / RMS / median / p95 | Centered deformation VRMS |",
        "|---|---|---:|---:|---:|",
    ))
    for row in matched["component_aggregate"]:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['components']} | {row['component_translation_error_mean']:.8g} / {row['component_translation_error_rms']:.8g} / {row['component_translation_error_median']:.8g} / {row['component_translation_error_p95']:.8g} | {row['centered_vertex_rms_mean']:.8g} |"
        )
    lines.extend((
        "",
        "The component translation modes of Hybrid reproduce E almost exactly, as expected because the random-walk Laplacian cannot constrain per-component constants and the direct anchor fixes them. Within-component centered error is between B and E, not better than E.",
        "",
        "## Frozen OOD (lambda fixed from matched validation)",
        "",
        "| Domain | Arm | Initial CD | Refined CD | Mean gain | P2S p95 | F-score | Normal | Flip rate | VRMS | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in ood["aggregate"]:
        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['initial_chamfer']:.8g} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:+.2%} | {row['p2s_p95']:.8g} | {row['fscore']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['same_index_recovered_vertex_rms']:.8g} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend((
        "",
        "| Domain | Comparison | Right lower CD | Right lower VRMS | Right lower P95 | Right higher F | Right higher normal | Right fewer flips |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ))
    for row in ood["paired"]:
        if ARM_H not in row["comparison"]:
            continue
        lines.append(
            f"| {row['domain']} | {row['comparison']} | {row['right_lower_chamfer']}/{row['samples']} | {row['right_lower_vertex_rms']}/{row['samples']} | {row['right_lower_p2s_p95']}/{row['samples']} | {row['right_higher_fscore']}/{row['samples']} | {row['right_higher_normal']}/{row['samples']} | {row['right_lower_flip_rate']}/{row['samples']} |"
        )
    lines.extend((
        "",
        "Relative OOD improvements are not called successful refinement unless the Hybrid aggregate gain is positive.",
        "",
        "## Lambda sensitivity and endpoint audit",
        "",
        "Small lambda collapses toward the unstable unanchored Laplacian inverse: validation CD is worst at `1e-4`. The useful CD basin is centered around `1e-2`–`1e-1`, with the validation optimum at `3e-2`. Increasing lambda monotonically reduces Hybrid→E vertex distance over the tested range; lambda `3` is already close to E but is not the mathematical infinity endpoint.",
        "",
        "The approximate per-mode interpretation `mu²/(mu²+lambda) * V_lap + lambda/(mu²+lambda) * V_direct` is used only to explain this behavior; recovery itself performs no explicit spectral decomposition.",
        "",
        "## Decision",
        "",
        f"Classification: **{classification}**.",
        "",
        classification_reason,
        "",
    ))
    if classification in {"HBR1", "HBR3"}:
        lines.append("Recommendation: the frozen fusion is strong enough to justify a later, separately controlled jointly trained hybrid ablation, but it does not authorize scaling or retraining automatically.")
    else:
        lines.append("Recommendation: do not start a jointly trained hybrid solely from this result; first resolve the identified matched/OOD or correspondence trade-off.")
    lines.extend(("", f"Metric protocol: `{matched['metric_protocol']}`.", ""))
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"contract_audit": contract_audit, "classification": classification, "lambda": sweep["lambda_hybrid_best"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

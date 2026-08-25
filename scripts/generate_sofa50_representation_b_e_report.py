#!/usr/bin/env python3
from __future__ import annotations

"""Combine frozen matched/OOD/spectral diagnostics into the requested report."""

import argparse
import json
from pathlib import Path
from typing import Any


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
VARIANTS = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup(rows, **keys):
    return next(row for row in rows if all(row[key] == value for key, value in keys.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", required=True, type=Path)
    parser.add_argument("--ood-dir", required=True, type=Path)
    parser.add_argument("--visual-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    matched = _read(args.matched_dir.resolve() / "summary.json")
    ood = _read(args.ood_dir.resolve() / "ood_summary.json")
    visuals = _read(args.visual_dir.resolve() / "comparison_manifest.json")
    audit = bool(
        matched["implementation_audit"]
        and ood["implementation_audit"]
        and visuals["read_only"]
        and visuals["objective_selection"]
    )
    aggregate = matched["aggregate"]
    paired_wins = matched["paired_wins"]
    all_b = _lookup(aggregate, group="all", arm=ARM_B)
    all_e = _lookup(aggregate, group="all", arm=ARM_E)
    e_recipe_cd_wins = sum(
        _lookup(aggregate, group=variant, arm=ARM_E)["refined_chamfer"]
        < _lookup(aggregate, group=variant, arm=ARM_B)["refined_chamfer"]
        for variant in VARIANTS
    )
    severity = matched["severity_aggregate"]
    e_severity_cd_wins = sum(
        _lookup(severity, group=label, arm=ARM_E)["refined_chamfer"]
        < _lookup(severity, group=label, arm=ARM_B)["refined_chamfer"]
        for label in ("low", "medium", "high")
    )
    spectral = {row["signal"]: row for row in matched["spectral_aggregate"]}
    b_error, e_error = spectral["b_error"], spectral["e_error"]
    frequency_winners = {}
    for band in ("low", "mid", "high"):
        b_value, e_value = float(b_error[f"{band}_energy"]), float(e_error[f"{band}_energy"])
        relative = (e_value - b_value) / max(min(e_value, b_value), 1e-30)
        frequency_winners[band] = "tie" if abs(relative) <= 0.01 else (ARM_E if e_value < b_value else ARM_B)
    meaningful = {value for value in frequency_winners.values() if value != "tie"}
    frequency_crossover = meaningful == {ARM_B, ARM_E}

    ood_available = [row for row in ood["domains"] if row["available"]]
    ood_aggregate = ood["aggregate"]
    ood_paired = {row["domain"]: row for row in ood["paired"]}
    ood_winners = {}
    robust_b_domains = 0
    for domain in ood_available:
        name = domain["domain"]
        b = _lookup(ood_aggregate, domain=name, arm=ARM_B)
        e = _lookup(ood_aggregate, domain=name, arm=ARM_E)
        winner = ARM_E if e["refined_chamfer"] < b["refined_chamfer"] else ARM_B
        ood_winners[name] = winner
        if winner == ARM_B and b["refined_chamfer"] <= 0.98 * e["refined_chamfer"]:
            robust_b_domains += 1

    e_matched = all_e["refined_chamfer"] < all_b["refined_chamfer"]
    if frequency_crossover:
        classification = "R3"
        title = "Frequency complementarity"
        recommendation = "Investigate a hybrid representation that combines direct low/global correction with the representation that wins the complementary spectral band; do not scale either representation alone from this result."
    elif e_matched and robust_b_domains > 0:
        classification = "R2"
        title = "Matched-domain E, OOD Laplacian"
        recommendation = "Keep the Laplacian path as an OOD inductive-bias candidate while testing a direct-residual matched-domain head; a hybrid or domain-aware gate is justified."
    elif (
        e_matched
        and e_recipe_cd_wins >= 8
        and sum(value == ARM_E for value in ood_winners.values()) >= max(1, len(ood_winners) - 1)
        and all(value in (ARM_E, "tie") for value in frequency_winners.values())
    ):
        classification = "R1"
        title = "Direct residual broadly dominates"
        recommendation = "Use direct vertex residual as the next primary representation; current evidence does not support Laplacian prediction as superior."
    else:
        classification = "R4"
        title = "Regime complementarity"
        recommendation = "Investigate an input-dependent representation or hybrid gate; neither representation is uniformly preferable across coarse regimes."

    lines = [
        "# Sofa50 v2 frozen Arm B vs Arm E representation diagnostic",
        "",
        f"Implementation/read-only audit: **{str(audit).lower()}**. No checkpoint was trained, fine-tuned, or modified.",
        "",
        "## 1. Implementation and inference audit",
        "",
        f"Arm B checkpoint SHA-256: `{matched['arm_b_checkpoint_sha256']}`. Arm E checkpoint SHA-256: `{matched['arm_e_checkpoint_sha256']}`. Both contain `{matched['parameter_count']}` parameters. OOD inference passes the exact same prepared RGB/camera/mesh mapping to both models; GT/targets are absent from that mapping and used only after inference for evaluation.",
        "",
        "## 2. A1-D2 B-vs-E breakdown",
        "",
        "| Recipe | B CD | E CD | B gain | E gain | B VRMS | E VRMS | B P95 | E P95 | B/E flip rate | E CD wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        b = _lookup(aggregate, group=variant, arm=ARM_B)
        e = _lookup(aggregate, group=variant, arm=ARM_E)
        win = _lookup(paired_wins, group=variant, metric="refined_chamfer")
        lines.append(
            f"| {variant} | {b['refined_chamfer']:.8g} | {e['refined_chamfer']:.8g} | {b['relative_chamfer_gain']:.2%} | {e['relative_chamfer_gain']:.2%} | {b['same_index_recovered_vertex_rms']:.8g} | {e['same_index_recovered_vertex_rms']:.8g} | {b['p2s_p95']:.8g} | {e['p2s_p95']:.8g} | {b['normalized_flip_rate']:.3%} / {e['normalized_flip_rate']:.3%} | {win['e_wins']}/{win['samples']} |"
        )
    lines.extend((
        "",
        f"Arm E has lower mean recipe-level CD in **{e_recipe_cd_wins}/10** recipes. Full requested metrics and per-metric B/E/tie counts are in `recipe_and_group_aggregate.csv` and `recipe_and_group_paired_wins.csv`.",
        "",
        "## 3. Mild/strong and topology-family summary",
        "",
        "| Group | B CD | E CD | B VRMS | E VRMS | B normal | E normal | B/E flip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for group in ("mild", "strong", "original_topology", "global_midpoint", "adaptive_topology", "all"):
        b = _lookup(aggregate, group=group, arm=ARM_B)
        e = _lookup(aggregate, group=group, arm=ARM_E)
        lines.append(
            f"| {group} | {b['refined_chamfer']:.8g} | {e['refined_chamfer']:.8g} | {b['same_index_recovered_vertex_rms']:.8g} | {e['same_index_recovered_vertex_rms']:.8g} | {b['normal_consistency']:.8g} | {e['normal_consistency']:.8g} | {b['normalized_flip_rate']:.3%} / {e['normalized_flip_rate']:.3%} |"
        )
    lines.extend((
        "",
        "## 4. Correction-severity analysis",
        "",
        "Severity is the GT displacement RMS rank split into equal-count low/medium/high bins. It is diagnostic-only and unavailable at deployment.",
        "",
        "| Severity | Samples | B CD | E CD | B VRMS | E VRMS | B P95 | E P95 | B/E normal | E CD wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for label in ("low", "medium", "high"):
        b = _lookup(severity, group=label, arm=ARM_B)
        e = _lookup(severity, group=label, arm=ARM_E)
        win = _lookup(matched["severity_paired_wins"], group=label, metric="refined_chamfer")
        lines.append(
            f"| {label} | {b['samples']} | {b['refined_chamfer']:.8g} | {e['refined_chamfer']:.8g} | {b['same_index_recovered_vertex_rms']:.8g} | {e['same_index_recovered_vertex_rms']:.8g} | {b['p2s_p95']:.8g} | {e['p2s_p95']:.8g} | {b['normal_consistency']:.8g} / {e['normal_consistency']:.8g} | {win['e_wins']}/{win['samples']} |"
        )
    gt_cd_corr = _lookup(matched["correlations"], severity_or_proxy="gt_displacement_rms", outcome="cd_e_minus_b")
    gt_vrms_corr = _lookup(matched["correlations"], severity_or_proxy="gt_displacement_rms", outcome="vrms_e_minus_b")
    lines.extend((
        "",
        f"GT displacement RMS correlation with `CD_E-CD_B`: Spearman `{gt_cd_corr['spearman']:.4f}`; with `VRMS_E-VRMS_B`: `{gt_vrms_corr['spearman']:.4f}`. E has lower bin-mean CD in `{e_severity_cd_wins}/3` severity bins. GT-free proxy results are in `severity_correlations.csv`.",
        "",
        "## 5. Frozen OOD/generalization",
        "",
        "| Domain | Arm | Valid | Initial CD | Refined CD | Gain | Improved/worsened | P95 | Normal | Flip rate | Vertex RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in ood_aggregate:
        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['valid_samples']} | {row['initial_chamfer']:.8g} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:.2%} | {row['improved']}/{row['worsened']} | {row['p2s_p95']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['same_index_recovered_vertex_rms']:.8g} |"
        )
    lines.extend((
        "",
        "OOD results were not used for tuning. Any incompatible candidate is explicitly marked unavailable in `ood_summary.json`.",
        "",
        "## 6. Paired statistics on matched Sofa50 test",
        "",
        "| Quantity | Mean E-B | Median E-B | Bootstrap 95% CI | Wilcoxon p |",
        "|---|---:|---:|---:|---:|",
    ))
    for row in matched["paired_statistics"]:
        lines.append(
            f"| {row['quantity']} | {row['mean_paired_difference']:+.8g} | {row['median_paired_difference']:+.8g} | [{row['bootstrap_ci95_low']:+.8g}, {row['bootstrap_ci95_high']:+.8g}] | {row['wilcoxon_p']:.6g} |"
        )
    lines.extend((
        "",
        "## 7. Graph-spectrum definition",
        "",
        matched["spectral_protocol"] + ". Bands are identical for B and E on each matched input graph; xyz coefficient energy is summed. This full-spectrum approximation avoids a topology-size-dependent fixed eigenvector cutoff.",
        "",
        "## 8. Low/mid/high frequency error",
        "",
        "| Signal | Total energy | Mean/vertex | Low | Mid | High |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in matched["spectral_aggregate"]:
        lines.append(
            f"| {row['signal']} | {row['total_energy']:.8g} | {row['mean_energy_per_vertex']:.8g} | {row['low_fraction']:.2%} | {row['mid_fraction']:.2%} | {row['high_fraction']:.2%} |"
        )
    lines.extend((
        "",
        "Band winners by absolute error energy: " + ", ".join(f"{band}=`{winner}`" for band, winner in frequency_winners.items()) + ". A winner requires more than a 1% aggregate-energy difference; otherwise the band is a tie.",
        "",
        "## 9. Representative matched visualizations",
        "",
        "Five cases are selected by objective rules: strongest E CD win, strongest B CD win, nearest tie, and the largest E win within mild and strong samples. Each contains matched full geometry plus low/mid/high displacement and error reconstructions.",
        "",
    ))
    for row in visuals["records"]:
        lines.append(f"- `{row['selection_rule']}`: `{row['sample_id']}` → `{row['full_panel']}`")
    lines.extend((
        "",
        "## 10. Final classification",
        "",
        f"**Case {classification} — {title}.**",
        "",
        "The classification follows the predeclared hierarchy: a meaningful cross-band winner change gives R3; otherwise matched E plus materially stronger Laplacian OOD gives R2; broad E consistency gives R1; remaining input-regime dependence gives R4.",
        "",
        "## 11. Recommendation",
        "",
        recommendation,
        "",
    ))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "contract_audit": audit,
        "classification": classification,
        "classification_title": title,
        "recommendation": recommendation,
        "frequency_winners": frequency_winners,
        "ood_winners": ood_winners,
        "e_recipe_cd_wins": e_recipe_cd_wins,
        "e_severity_cd_wins": e_severity_cd_wins,
    }
    output.with_name("FINAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

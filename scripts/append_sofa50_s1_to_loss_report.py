#!/usr/bin/env python3
from __future__ import annotations

"""Append the completed, separately trained S1 diagnostics to the loss audit."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MARKER = "<!-- S1_LOSS_APPENDIX -->"
LABELS = (
    "step000000", "step001000", "step002500", "step005000", "step007500",
    "step010000", "step012500", "step015000", "step017500", "step020000", "best",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(rows: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {keys}; got {len(selected)}")
    return selected[0]


def _f(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    headers = tuple(headers)
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-root", required=True, type=Path)
    parser.add_argument("--s1-root", required=True, type=Path)
    args = parser.parse_args()
    loss_root = args.loss_root.resolve()
    s1_root = args.s1_root.resolve()
    report = loss_root / "FINAL_REPORT.md"
    decision = _read(loss_root / "decision.json")
    selected = _read(s1_root / "selected_analysis" / "s1_selected_summary.json")
    evolution = [_read(loss_root / "s1_evolution" / f"s1_loss_evolution_{label}.json") for label in LABELS]
    checks = {
        "base_loss_report_contract": bool(decision["contract_audit"]),
        "s1_selected_contract": bool(selected["contract_audit"]),
        "all_s1_evolution_read_only": all(row["read_only"] for row in evolution),
        "all_s1_evolution_finite": all(row["all_finite"] for row in evolution),
        "all_s1_evolution_pcg_converged": all(row["all_pcg_converged"] for row in evolution),
        "exact_checkpoint_sequence": tuple(row["label"] for row in evolution) == LABELS,
    }
    appendix = [
        MARKER,
        "",
        "## S1 post-training loss-mechanism appendix",
        "",
        f"Appendix contract audit: **{str(all(checks.values())).lower()}**. S1 remained a separately trained architecture experiment; these rows were added only after its checkpoint was frozen and do not alter the LOSS1--LOSS5 decision above.",
        "",
        "### Selected S1 output gradients",
        "",
    ]
    output_rows = []
    for split in ("validation", "test"):
        for path in ("g_S1_delta", "g_S1_direct"):
            row = _pick(selected["output_gradient_aggregate"], split=split, path=path)
            output_rows.append((split, path, _f(row["total_energy"]), _f(row["low_energy"]), _f(row["mid_energy"]), _f(row["high_energy"]), _f(row["mean_low_fraction"]), _f(row["mean_mid_fraction"]), _f(row["mean_high_fraction"])))
    appendix.extend(_table(("Split", "Path", "Total", "Low", "Mid", "High", "Low frac.", "Mid frac.", "High frac."), output_rows))
    counterfactual_rows = []
    for row in selected["same_state_aggregate"]:
        counterfactual_rows.append((row["split"], _f(row["mean_cosine"]), _f(row["mean_direct_norm"]), _f(row["mean_recovery_norm"]), _f(row["mean_norm_ratio_recovery_over_direct"]), _f(row["low_fraction_direct"]), _f(row["low_fraction_recovery"]), _f(row["high_fraction_direct"]), _f(row["high_fraction_recovery"])))
    appendix.extend(["", "### S1 same-state direct-Lap versus recovery supervision", ""] + _table(("Split", "Mean cos", "||g_direct||", "||g_recovery||", "Recovery/direct", "Direct low", "Recovery low", "Direct high", "Recovery high"), counterfactual_rows))
    positional_rows = []
    for row in selected["positional_same_state_aggregate"]:
        positional_rows.append((row["split"], _f(row["mean_cosine"]), _f(row["mean_norm_ratio_hybrid_over_direct"]), _f(row["mean_direct_vertex_gradient_mean"]), _f(row["mean_hybrid_vertex_gradient_mean"]), _f(row["mean_direct_vertex_gradient_p95"]), _f(row["mean_hybrid_vertex_gradient_p95"])))
    appendix.extend(["", "### S1 same-state direct-MSE versus hybrid positional supervision", ""] + _table(("Split", "Mean cos", "Hybrid/direct", "Direct mean", "Hybrid mean", "Direct p95", "Hybrid p95"), positional_rows))
    appendix.extend(["", "### Selected S1 localization and RHS interaction", ""])
    localization_rows = []
    for split in ("validation", "test"):
        for path in ("g_S1_delta", "g_S1_direct"):
            row = _pick(selected["output_correlation_aggregate"], split=split, path=path, feature="final_recovered_geometry_error")
            localization_rows.append((split, path, _f(row["mean_pearson"]), _f(row["mean_spearman"]), _f(row["mean_top10_gradient_energy_fraction"]), _f(row["mean_top1_gradient_energy_fraction"])))
    appendix.extend(_table(("Split", "Path", "Pearson", "Spearman", "Top-error 10% grad E", "Top-error 1% grad E"), localization_rows))
    top_rows = []
    for split in ("validation", "test"):
        for path in ("g_S1_delta", "g_S1_direct"):
            for feature in ("raw_laplacian_error", "same_index_vertex_error", "gt_differential_magnitude", "final_recovered_geometry_error"):
                row = _pick(selected["output_correlation_aggregate"], split=split, path=path, feature=feature)
                top_rows.append((split, path, feature, _f(row["mean_feature_all_vertices"]), _f(row["mean_feature_on_top10_gradient_vertices"]), _f(row["mean_feature_on_top1_gradient_vertices"])))
    appendix.extend([""] + _table(("Split", "Path", "Feature", "All vertices", "Top-grad 10%", "Top-grad 1%"), top_rows))
    rhs_rows = []
    for split in ("validation", "test"):
        row = _pick(selected["rhs_aggregate"], split=split)
        rhs_rows.append((split, _f(row["mean_lap_rhs_norm"]), _f(row["mean_direct_rhs_norm"]), _f(row["mean_combined_rhs_norm"]), _f(row["median_rhs_cosine"]), _f(row["median_cancellation_ratio"]), _f(row["p10_cancellation_ratio"]), _f(row["p90_cancellation_ratio"])))
    appendix.extend([""] + _table(("Split", "||e_L||", "||e_D||", "||e_q||", "Median cos", "Median C", "p10 C", "p90 C"), rhs_rows))
    appendix.extend(["", "### S1 checkpoint evolution (fixed validation indices 0,5,...,45)", ""])
    evolution_rows = []
    for item in evolution:
        lap = _pick(item["spectral_aggregate"], path="g_S1_delta")
        direct = _pick(item["spectral_aggregate"], path="g_S1_direct")
        shared = _pick(item["shared_gradient_aggregate"], layer="full_shared_frontend_parameters")
        rhs = item["rhs_aggregate"]
        semantic = item["semantic_aggregate"]
        evolution_rows.append((item["label"], _f(lap["mean_gradient_norm"]), _f(direct["mean_gradient_norm"]), _f(shared["mean_cosine"]), _f(shared["median_magnitude_ratio"]), _f(rhs["mean_lap_rhs_norm"]), _f(rhs["mean_direct_rhs_norm"]), _f(rhs["mean_combined_rhs_norm"]), _f(rhs["median_cancellation_ratio"]), _f(semantic["lap_raw_epe"]), _f(semantic["direct_vertex_rms"]), _f(semantic["hybrid_chamfer"])))
    appendix.extend(_table(("Checkpoint", "Lap grad", "Direct grad", "Shared cos", "Norm ratio", "||e_L||", "||e_D||", "||e_q||", "Median C", "Lap EPE", "Direct VRMS", "Hybrid CD"), evolution_rows))
    spectral_evolution_rows = []
    for item in evolution:
        for path in ("g_S1_delta", "g_S1_direct"):
            row = _pick(item["spectral_aggregate"], path=path)
            spectral_evolution_rows.append((item["label"], path, _f(row["mean_total_energy"]), _f(row["mean_low_energy"]), _f(row["mean_mid_energy"]), _f(row["mean_high_energy"])))
    appendix.extend(["", "### S1 checkpoint output-gradient absolute energy", ""] + _table(("Checkpoint", "Path", "Total", "Low", "Mid", "High"), spectral_evolution_rows))
    appendix.extend(["", "S1 per-sample absolute energies, shared-front-end VJPs, RHS terms, PCG audits, and checkpoint SHA-256 identities are stored in `s1_evolution/`.", ""])
    existing = report.read_text(encoding="utf-8")
    if MARKER in existing:
        existing = existing.split(MARKER, 1)[0].rstrip()
    report.write_text(existing + "\n\n" + "\n".join(appendix), encoding="utf-8")
    payload = {"contract_audit": all(checks.values()), "contract_checks": checks, "selected_checkpoint_sha256": selected["checkpoint_sha256"], "evolution_labels": list(LABELS)}
    (loss_root / "s1_appendix.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

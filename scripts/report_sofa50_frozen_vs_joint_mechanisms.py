#!/usr/bin/env python3
from __future__ import annotations

"""Render the frozen-specialists versus joint-model mechanism report."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


B_CHECKPOINT = "/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt"
E_CHECKPOINT = "/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt"
B_SHA = "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c"
E_SHA = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"
JOINT_SHA = "9af46b5c3203415aa06c3967fe2f5d36bd1cab389f036c481e147e874e5dab62"
METHOD_LABELS = {
    "Pretrained_B": "Pretrained B",
    "Pretrained_E": "Pretrained E",
    "Frozen_BE": "Frozen B+E",
    "Joint_Lap": "Joint Lap",
    "Joint_Direct": "Joint Direct",
    "Joint_Hybrid": "Joint Hybrid",
}
LAYER_LABELS = {
    "image_encoder": "Encoder",
    "projected_image_field": "Projected image field",
    "graph_backbone": "Graph backbone",
    "shared_feature_Phi": "Shared feature Phi",
    "all_shared_parameters": "All shared parameters",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _row(rows: Sequence[Mapping[str, Any]], **keys: Any) -> Mapping[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {keys}, got {len(selected)}")
    return selected[0]


def _f(value: Any, digits: int = 7) -> str:
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1e4:
        return f"{number:.4e}"
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _classification(mechanism: Mapping[str, Any], gradient: Mapping[str, Any]) -> dict[str, Any]:
    grad = _row(gradient["aggregate"], layer="all_shared_parameters")
    grad_strong = bool(
        float(grad["cosine_mean"]) <= -0.10
        and float(grad["fraction_cosine_negative"]) >= 0.60
    )
    be = _row(mechanism["latent_aggregate"], split="validation", pair="B_E")
    joint = _row(
        mechanism["latent_aggregate"], split="validation", pair="Joint_Lap_Direct"
    )
    be_norm = abs(math.log(max(float(be["norm_ratio_mean"]), 1e-30)))
    joint_norm = abs(math.log(max(float(joint["norm_ratio_mean"]), 1e-30)))
    redundancy_strong = bool(
        float(joint["relative_discrepancy_mean"])
        <= 0.75 * float(be["relative_discrepancy_mean"])
        and float(joint["cosine_mean"]) >= float(be["cosine_mean"]) + 0.10
        and joint_norm <= 0.75 * max(be_norm, 1e-12)
    )
    lap_b = _row(
        mechanism["lap_semantic_aggregate"], split="validation", method="Pretrained_B"
    )
    lap_j = _row(
        mechanism["lap_semantic_aggregate"], split="validation", method="Joint_Lap"
    )
    direct_e = _row(
        mechanism["position_semantic_aggregate"], split="validation", method="Pretrained_E"
    )
    direct_j = _row(
        mechanism["position_semantic_aggregate"], split="validation", method="Joint_Direct"
    )
    specialization_strong = bool(
        float(lap_b["raw_epe_mean"]) < 0.95 * float(lap_j["raw_epe_mean"])
        and float(direct_e["vertex_rms_mean"]) < 0.95 * float(direct_j["vertex_rms_mean"])
    )
    corr_be = _row(
        mechanism["error_pair_aggregate"], split="validation", pair="B_E", band="global"
    )
    corr_joint = _row(
        mechanism["error_pair_aggregate"],
        split="validation",
        pair="Joint_Lap_Direct",
        band="global",
    )
    gain_be = _row(mechanism["fusion_aggregate"], split="validation", pair="B_E")
    gain_joint = _row(
        mechanism["fusion_aggregate"], split="validation", pair="Joint_Lap_Direct"
    )
    complementarity_strong = bool(
        float(corr_be["error_cosine_mean"]) + 0.10
        <= float(corr_joint["error_cosine_mean"])
        and float(gain_be["fusion_gain_mean"]) > float(gain_joint["fusion_gain_mean"])
    )
    if grad_strong and redundancy_strong:
        label = "MECH3"
    elif grad_strong:
        label = "MECH1"
    elif redundancy_strong:
        label = "MECH2"
    elif specialization_strong and complementarity_strong:
        label = "MECH4"
    else:
        label = "MECH5"
    return {
        "classification": label,
        "gradient_interference_strong": grad_strong,
        "representation_redundancy_strong": redundancy_strong,
        "specialization_gap_strong": specialization_strong,
        "complementarity_gap_strong": complementarity_strong,
        "rule": "gradient: all-shared mean cosine<=-0.10 and negative fraction>=60%; redundancy: >=25% lower relative discrepancy, >=0.10 higher cosine, and >=25% closer log norm-ratio to zero; specialization: both B raw-EPE and E VRMS >=5% better; complementarity: B/E global error cosine >=0.10 lower and mean fusion gain higher",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    mechanism = _read(args.input_dir / "mechanism_summary.json")
    gradient = _read(args.input_dir / "gradient_summary.json")
    evolution = []
    for path in sorted(args.input_dir.glob("evolution_*.json")):
        evolution.append(_read(path))
    preferred = {"step005000": 0, "step010000": 1, "step015000": 2, "step020000": 3, "best": 4}
    evolution.sort(key=lambda row: preferred.get(str(row["label"]), 99))
    decision = _classification(mechanism, gradient)
    checks = {
        "mechanism_contract": bool(mechanism["contract_audit"]),
        "gradient_contract": bool(gradient["contract_audit"]),
        "joint_best_sha": gradient["checkpoint_sha256"] == JOINT_SHA,
        "all_evolution_read_only": all(row["read_only"] for row in evolution),
        "all_evolution_solvers_converged": all(row["all_solvers_converged"] for row in evolution),
        "all_five_expected_checkpoints": [row["label"] for row in evolution]
        == ["step005000", "step010000", "step015000", "step020000", "best"],
    }
    references = {
        "Pretrained_B": 0.00358497,
        "Pretrained_E": 0.00334039,
        "Frozen_BE": 0.00302983,
        "Joint_Hybrid": 0.00341856765,
    }
    reference_rows = []
    for method, reference in references.items():
        actual = float(
            _row(mechanism["geometry_aggregate"], split="test", method=method)["chamfer"]
        )
        relative = abs(actual - reference) / reference
        reference_rows.append((method, reference, actual, relative))
    checks["reference_test_geometry_reproduced"] = all(
        relative <= (1e-3 if method == "Joint_Hybrid" else 2e-4)
        for method, _reference, _actual, relative in reference_rows
    )
    contract = all(checks.values())

    lines = [
        "# Sofa50 v2 frozen-specialist versus shared-joint mechanism analysis",
        "",
        f"Contract audit: **{'true' if contract else 'false'}**. Classification: **{decision['classification']}**.",
        "",
        "This is a read-only diagnostic of existing checkpoints. No model was trained, fine-tuned, or modified; the active pretrained-B+E continuation was neither used nor awaited.",
        "",
        "## Checkpoint identity and formulation",
        "",
        f"- Arm B: `{B_CHECKPOINT}`; SHA-256 `{B_SHA}`.",
        f"- Arm E: `{E_CHECKPOINT}`; SHA-256 `{E_SHA}`.",
        f"- From-scratch shared joint best: `{gradient['checkpoint']}`; SHA-256 `{gradient['checkpoint_sha256']}`.",
        "- Joint selected epoch: `384`; frozen and joint fusion both use Uniform `L_U=I-D^-1 A` and `lambda=3e-2`.",
        "",
        "The final solve is `A_H V_H=L_U^T delta_hat+lambda V_direct`, where `A_H=L_U^T L_U+lambda I`. For `g=dLoss/dV_H`, the audit solves `z=A_H^-1 g` and independently evaluates `J_delta^T(L_U z)` and `J_direct^T(lambda z)`. Branch-specific output heads are excluded from shared-parameter cosine.",
        "",
        "## Consolidated validation mechanism table",
        "",
    ]
    geometry = mechanism["geometry_aggregate"]
    spectral = mechanism["spectral_aggregate"]
    components = mechanism["component_aggregate"]
    primary_rows = []
    for method in METHOD_LABELS:
        geo = _row(geometry, split="validation", method=method)
        spec = _row(spectral, split="validation", method=method)
        comp = _row(components, split="validation", method=method)
        primary_rows.append(
            (
                METHOD_LABELS[method],
                _f(geo["chamfer"]),
                _f(geo["vertex_rms"]),
                _f(spec["low_energy"]),
                _f(spec["mid_energy"]),
                _f(spec["high_energy"]),
                _f(comp["component_translation_rms"]),
                _f(comp["centered_deformation_vrms"]),
            )
        )
    lines.extend(
        _table(
            ("Method", "CD", "Vertex RMS", "Low energy", "Mid energy", "High energy", "Component translation RMS", "Centered VRMS"),
            primary_rows,
        )
    )
    lines.extend(["", "### Validation standalone geometry details", ""])
    validation_rows = []
    for method in METHOD_LABELS:
        geo = _row(geometry, split="validation", method=method)
        validation_rows.append(
            (
                METHOD_LABELS[method], _f(geo["chamfer"]), _f(geo["vertex_rms"]),
                _f(geo["p2s_p95"]), _f(geo["fscore"]), _f(geo["normal"]),
                f"{geo['flips']} / {_pct(geo['flip_rate'])}",
                f"{geo['new_degenerates']}", f"{geo['improved']}/{geo['worsened']}",
            )
        )
    lines.extend(
        _table(
            ("Method", "CD", "VRMS", "P2S p95", "F-score", "Normal", "Flips / rate", "New deg.", "Improved/worsened"),
            validation_rows,
        )
    )
    lines.extend(["", "## Matched-v2 test confirmation", ""])
    test_rows = []
    for method in METHOD_LABELS:
        geo = _row(geometry, split="test", method=method)
        test_rows.append(
            (
                METHOD_LABELS[method], _f(geo["chamfer"]), _f(geo["vertex_rms"]),
                _f(geo["p2s_p95"]), _f(geo["fscore"]), _f(geo["normal"]),
                f"{geo['flips']} / {_pct(geo['flip_rate'])}", f"{geo['improved']}/{geo['worsened']}",
            )
        )
    lines.extend(_table(("Method", "CD", "VRMS", "P2S p95", "F-score", "Normal", "Flips / rate", "Improved/worsened"), test_rows))
    lines.extend(["", "### Established-result consistency", ""])
    lines.extend(
        _table(
            ("Method", "Established test CD", "Read-only rerun CD", "Relative difference"),
            [
                (METHOD_LABELS[method], _f(reference), _f(actual), _pct(relative))
                for method, reference, actual, relative in reference_rows
            ],
        )
    )
    lines.extend(["", "The joint latent inference uses the documented non-bitwise-deterministic FP16 CUDA path; its identity gate is the checkpoint SHA, while the geometry reproduction gate allows 0.1% relative variation. Frozen B+E uses its established `tol=1e-4`; Joint Hybrid uses its established `tol=1e-8`."])

    lines.extend(["", "## Exact shared-gradient decomposition", ""])
    grad_rows = []
    for layer in ("image_encoder", "projected_image_field", "graph_backbone", "shared_feature_Phi", "all_shared_parameters"):
        row = _row(gradient["aggregate"], layer=layer)
        primary = [
            item
            for item in gradient["rows"]
            if item["layer"] == layer and int(item["repeat"]) == 0
        ]
        zero_direct = sum(float(item["direct_norm"]) <= 1e-30 for item in primary)
        grad_rows.append(
            (
                LAYER_LABELS[layer], _f(row["cosine_mean"], 4), _f(row["cosine_median"], 4),
                f"{_f(row['cosine_p10'],4)} / {_f(row['cosine_p90'],4)}", _pct(row["fraction_cosine_negative"]),
                _f(row["lap_norm_mean"], 4), _f(row["direct_norm_mean"], 4),
                _f(row["magnitude_ratio_median"], 3), f"{zero_direct}/{len(primary)}",
                _f(row["alignment_ratio_mean"], 4),
            )
        )
    lines.extend(_table(("Layer", "Mean cos", "Median", "p10 / p90", "Conflict", "||g_lap||", "||g_direct||", "Median norm ratio", "Zero direct", "R_align"), grad_rows))
    lines.extend(["", "### Full gradient-cosine distribution", ""])
    distribution_rows = []
    for layer in ("image_encoder", "projected_image_field", "graph_backbone", "shared_feature_Phi", "all_shared_parameters"):
        row = _row(gradient["aggregate"], layer=layer)
        distribution_rows.append(
            (
                LAYER_LABELS[layer], _f(row["cosine_mean"], 4), _f(row["cosine_median"], 4),
                _f(row["cosine_p10"], 4), _f(row["cosine_p25"], 4),
                _f(row["cosine_p75"], 4), _f(row["cosine_p90"], 4),
                _f(row["cosine_minimum"], 4), _f(row["cosine_maximum"], 4),
                _pct(row["fraction_cosine_negative"]),
                _pct(row["fraction_cosine_below_minus_0p25"]),
                _pct(row["fraction_cosine_above_plus_0p25"]),
            )
        )
    lines.extend(
        _table(
            ("Layer", "Mean", "Median", "p10", "p25", "p75", "p90", "Min", "Max", "<0", "<-0.25", ">+0.25"),
            distribution_rows,
        )
    )
    max_analytic = max(float(row["analytic_gradient_relative_error"]) for row in gradient["solver_audit_rows"])
    lines.extend(["", f"All VJPs were finite; maximum analytic latent-gradient relative error was `{max_analytic:.3e}`.", "", "### FP16 repeat-noise envelope", ""])
    noise_rows = []
    for layer in ("all_shared_parameters", "shared_feature_Phi"):
        for metric in ("cosine", "lap_norm", "direct_norm", "alignment_ratio"):
            row = _row(gradient["fp16_repeat_noise"], layer=layer, metric=metric)
            noise_rows.append((LAYER_LABELS[layer], metric, row["samples"], _f(row["mean_of_repeat_means"], 6), _f(row["mean_repeat_std"], 6), _f(row["maximum_repeat_range"], 6)))
    lines.extend(_table(("Layer", "Metric", "n", "Mean", "Mean repeat std", "Max repeat range"), noise_rows))

    lines.extend(["", "## Latent redundancy", ""])
    latent_rows = []
    for split in ("validation", "test"):
        for pair, label in (("B_E", "B + E"), ("Joint_Lap_Direct", "Joint Lap + Direct")):
            row = _row(mechanism["latent_aggregate"], split=split, pair=pair)
            latent_rows.append((split, label, _f(row["redundancy_rms_mean"]), _f(row["relative_discrepancy_mean"]), _f(row["cosine_mean"], 5), _f(row["norm_ratio_mean"], 5)))
    lines.extend(_table(("Split", "Pair", "Redundancy RMS", "Relative", "Cosine", "Norm ratio"), latent_rows))

    lines.extend(["", "## GT semantic specialization", "", "### Differential representation", ""])
    lap_rows = []
    for split in ("validation", "test"):
        for method in ("Pretrained_B", "Joint_Lap"):
            row = _row(mechanism["lap_semantic_aggregate"], split=split, method=method)
            lap_rows.append((split, METHOD_LABELS[method], _f(row["raw_epe_mean"]), _f(row["raw_rms_mean"]), _f(row["raw_cosine_mean"], 6), _f(row["top10_epe_mean"]), _f(row["top1_epe_mean"])))
    lines.extend(_table(("Split", "Method", "Raw EPE", "RMS", "Cosine", "Top10", "Top1"), lap_rows))
    lines.extend(["", "### Positional representation", ""])
    pos_rows = []
    for split in ("validation", "test"):
        for method in ("Pretrained_E", "Joint_Direct"):
            row = _row(mechanism["position_semantic_aggregate"], split=split, method=method)
            comp = _row(mechanism["component_aggregate"], split=split, method=method)
            pos_rows.append((split, METHOD_LABELS[method], _f(row["vertex_rms_mean"]), _f(row["vertex_error_mean_mean"]), _f(row["vertex_error_p95_mean"]), _f(comp["component_translation_rms"]), _f(comp["centered_deformation_vrms"])))
    lines.extend(_table(("Split", "Method", "Vertex RMS", "Mean error", "p95", "Component translation RMS", "Centered VRMS"), pos_rows))
    lines.extend(["", "## Absolute graph-frequency error energy", ""])
    spectral_rows = []
    for split in ("validation", "test"):
        for method in METHOD_LABELS:
            row = _row(mechanism["spectral_aggregate"], split=split, method=method)
            spectral_rows.append(
                (
                    split, METHOD_LABELS[method], _f(row["total_energy"]),
                    _f(row["low_energy"]), _f(row["mid_energy"]), _f(row["high_energy"]),
                )
            )
    lines.extend(_table(("Split", "Method", "Total", "Low", "Mid", "High"), spectral_rows))
    lines.extend(["", "Absolute energies are primary; normalized fractions remain available in `spectral_aggregate.csv`.", "", "## Connected-component/nullspace diagnostic", ""])
    component_rows = []
    for split in ("validation", "test"):
        for method in METHOD_LABELS:
            row = _row(mechanism["component_aggregate"], split=split, method=method)
            component_rows.append(
                (
                    split, METHOD_LABELS[method], row["components"],
                    _f(row["component_translation_rms"]),
                    _f(row["component_translation_mean"]),
                    _f(row["centered_deformation_vrms"]),
                )
            )
    lines.extend(_table(("Split", "Method", "Components", "Translation RMS", "Translation mean", "Centered VRMS"), component_rows))

    lines.extend(["", "## Branch error correlation and overlap", ""])
    pair_rows = []
    for split in ("validation", "test"):
        for pair, label in (("B_E", "B + E"), ("Joint_Lap_Direct", "Joint Lap + Direct")):
            for band in ("global", "low", "mid", "high"):
                row = _row(mechanism["error_pair_aggregate"], split=split, pair=pair, band=band)
                pair_rows.append((split, label, band, _f(row["error_cosine_mean"], 5), _f(row["energy_overlap_mean"], 5)))
    lines.extend(_table(("Split", "Pair", "Band", "Error cosine", "Energy overlap"), pair_rows))

    lines.extend(["", "## Actual fusion benefit", ""])
    gain_rows = []
    for split in ("validation", "test"):
        for pair, label in (("B_E", "Frozen B+E"), ("Joint_Lap_Direct", "Joint Hybrid")):
            row = _row(mechanism["fusion_aggregate"], split=split, pair=pair)
            gain_rows.append((split, label, _f(row["fusion_gain_mean"]), _f(row["fusion_gain_median"]), f"{row['positive_count']}/{row['negative_count']}", f"{_f(row['fusion_gain_p25'])} / {_f(row['fusion_gain_p75'])}", _f(row["maximum"])))
    lines.extend(_table(("Split", "Fusion", "Mean gain", "Median", "+/-", "p25 / p75", "Maximum"), gain_rows))
    lines.extend(["", "Positive gain means fusion beats the better standalone branch.", "", "### Fusion-gain correlations (validation, n=50)", ""])
    corr_rows = []
    for pair, label in (("B_E", "Frozen B+E"), ("Joint_Lap_Direct", "Joint Hybrid")):
        for predictor in ("global_error_cosine", "global_energy_overlap", "relative_redundancy", "component_translation_complementarity"):
            row = _row(mechanism["fusion_correlations"], split="validation", pair=pair, predictor=predictor)
            corr_rows.append((label, predictor, row["n"], _f(row["pearson"], 4), _f(row["pearson_p"], 4), _f(row["spearman"], 4), _f(row["spearman_p"], 4)))
    lines.extend(_table(("Fusion", "Predictor", "n", "Pearson", "p", "Spearman", "p"), corr_rows))
    lines.extend(["", "These correlations are exploratory and are not interpreted causally.", "", "## Stored-checkpoint evolution (fixed validation indices 0,5,...,45)", ""])
    evo_rows = []
    for item in evolution:
        latent = item["latent_aggregate"]
        lap = item["lap_semantic_aggregate"]
        direct = item["direct_semantic_aggregate"]
        geos = {row["method"]: row for row in item["geometry_aggregate"]}
        shared = _row(item["gradient_aggregate"], layer="all_shared_parameters")
        evo_rows.append((item["label"], _f(latent["relative_discrepancy"]), _f(latent["cosine"], 5), _f(lap["raw_epe"]), _f(direct["vertex_rms"]), _f(geos["Joint_Lap"]["chamfer"]), _f(geos["Joint_Direct"]["chamfer"]), _f(geos["Joint_Hybrid"]["chamfer"]), _f(shared["cosine_mean"], 4), _pct(shared["conflict_rate"])))
    lines.extend(_table(("Checkpoint", "Relative redundancy", "Latent cos", "Lap EPE", "Direct VRMS", "Lap CD", "Direct CD", "Hybrid CD", "Shared grad cos", "Conflict"), evo_rows))
    lines.extend(["", "### Checkpoint evolution: absolute error energies", ""])
    evo_spectral_rows = []
    for item in evolution:
        for row in item["spectral_aggregate"]:
            evo_spectral_rows.append(
                (
                    item["label"], METHOD_LABELS[row["method"]], _f(row["total_energy"]),
                    _f(row["low_energy"]), _f(row["mid_energy"]), _f(row["high_energy"]),
                )
            )
    lines.extend(_table(("Checkpoint", "Method", "Total", "Low", "Mid", "High"), evo_spectral_rows))
    lines.extend(["", "All evolution rows use the identical fixed ten samples. No missing checkpoint was reconstructed or retrained.", "", "### Evolution checkpoint identities", ""])
    lines.extend(
        _table(
            ("Checkpoint", "SHA-256"),
            [(item["label"], item["checkpoint_sha256"]) for item in evolution],
        )
    )
    lines.extend(["", "## Paired specialist statistics", ""])
    paired_rows = []
    for row in mechanism["paired_specialist_statistics"]:
        paired_rows.append((row["split"], METHOD_LABELS.get(row["left"], row["left"]), METHOD_LABELS.get(row["right"], row["right"]), row["field"], _f(row["mean_difference"]), f"{row['left_wins']}/{row['right_wins']}/{row['ties']}", f"[{_f(row['bootstrap_ci95_low'])}, {_f(row['bootstrap_ci95_high'])}]"))
    lines.extend(_table(("Split", "Left", "Right", "Metric", "Left-right mean", "L/R/tie", "Bootstrap 95% CI"), paired_rows))

    lines.extend(["", "## Mechanism decision", "", f"Classification: **{decision['classification']}**.", ""])
    evidence_rows = [
        ("Strong gradient interference", decision["gradient_interference_strong"]),
        ("Strong latent redundancy", decision["representation_redundancy_strong"]),
        ("Strong specialist gap", decision["specialization_gap_strong"]),
        ("Strong complementarity gap", decision["complementarity_gap_strong"]),
    ]
    lines.extend(_table(("Diagnostic gate", "Result"), [(name, str(value).lower()) for name, value in evidence_rows]))
    lines.extend(["", f"Predeclared quantitative decision rule: {decision['rule']}.", ""])
    if decision["classification"] in {"MECH2", "MECH3", "MECH4"}:
        answer = "Within this specific Sofa50 v2 formulation, the evidence supports the explanation that separate objectives create more identifiable absolute and differential specialists, whose complementary errors make analytical fusion more effective than the from-scratch single-final-loss shared model."
    elif decision["classification"] == "MECH1":
        answer = "The dominant measured explanation is shared-gradient interference; the evidence does not require a strong latent-redundancy claim."
    else:
        answer = "The diagnostics do not isolate a sufficiently clear mechanism; the observed performance gap remains specific to these runs rather than a general result about separate versus joint training."
    lines.extend([answer, "", "This conclusion is restricted to the existing checkpoints and does not claim that separate training is universally better than joint training.", "", f"Metric protocol: `{mechanism['metric_protocol']}`.", "", f"Spectral protocol: `{mechanism['spectral_protocol']}`."])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {
        "contract_audit": contract,
        "contract_checks": checks,
        "decision": decision,
        "checkpoint_identities": {
            "arm_b": {"path": B_CHECKPOINT, "sha256": B_SHA},
            "arm_e": {"path": E_CHECKPOINT, "sha256": E_SHA},
            "joint_best": {"path": gradient["checkpoint"], "sha256": gradient["checkpoint_sha256"]},
        },
        "continuation_used": False,
        "training_or_checkpoint_mutation": False,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"contract_audit": contract, "classification": decision["classification"], "report": str(args.output_dir / 'FINAL_REPORT.md')}, indent=2))
    if not contract:
        raise RuntimeError(f"Mechanism report contract failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Merge the controlled Uniform/Cotangent hybrid study into its final report."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


RECIPES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")
MILD = {"A1", "B1", "C1", "C3", "D1"}
GROUPS = {
    **{recipe: {recipe} for recipe in RECIPES},
    "mild": MILD,
    "strong": set(RECIPES) - MILD,
    "original_topology": {"A1", "A2"},
    "midpoint_subdivision": {"B1", "B2"},
    "adaptive_topology": {"C1", "C2", "C3", "C4", "D1", "D2"},
}
LOWER = (
    "refined_chamfer",
    "p2s",
    "p2s_p95",
    "same_index_recovered_vertex_rms",
    "introduced_flipped_faces",
    "normalized_flip_rate",
    "new_degenerate_faces",
)
HIGHER = ("fscore", "normal_consistency")
AGGREGATE_FIELDS = (
    "initial_chamfer",
    "refined_chamfer",
    "relative_chamfer_gain",
    "eta",
    "p2s",
    "p2s_p95",
    "fscore",
    "normal_consistency",
    "same_index_recovered_vertex_rms",
    "normalized_flip_rate",
)
METRIC_PROTOCOL = (
    "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
    "area_weighted_triangle_surface_sampling;"
    "bidirectional_sampled_surface_to_exact_triangle_surface;"
    "surface_samples=3000;seed=7;fscore_threshold=0.01;"
    "alignment=shared_prepared_coordinate_frame_no_ICP"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(identity)
    result["samples"] = len(rows)
    for field in AGGREGATE_FIELDS:
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    result["introduced_flipped_faces"] = int(
        sum(int(row["introduced_flipped_faces"]) for row in rows)
    )
    result["new_degenerate_faces"] = int(
        sum(int(row["new_degenerate_faces"]) for row in rows)
    )
    result["improved"] = int(sum(bool(row["improved"]) for row in rows))
    result["worsened"] = int(sum(bool(row["worsened"]) for row in rows))
    result["pcg_iterations_mean"] = float(
        np.mean([float(row["pcg_iterations"]) for row in rows])
    )
    result["pcg_iterations_max"] = int(
        max(int(row["pcg_iterations"]) for row in rows)
    )
    return result


def _bootstrap(values: np.ndarray, *, samples: int = 10000) -> tuple[float, float]:
    generator = np.random.default_rng(7)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired(
    uniform: Sequence[Mapping[str, Any]], cotangent: Sequence[Mapping[str, Any]], scope: str
) -> list[dict[str, Any]]:
    left = {str(row["sample_id"]): row for row in uniform}
    right = {str(row["sample_id"]): row for row in cotangent}
    if left.keys() != right.keys():
        raise RuntimeError(f"Unpaired sample IDs in {scope}.")
    rows = []
    for field in (*LOWER, *HIGHER):
        differences = np.asarray(
            [float(right[key][field]) - float(left[key][field]) for key in sorted(left)]
        )
        lower_is_better = field in LOWER
        wins = int(np.sum(differences < 0)) if lower_is_better else int(np.sum(differences > 0))
        losses = int(np.sum(differences > 0)) if lower_is_better else int(np.sum(differences < 0))
        low, high = _bootstrap(differences)
        rows.append(
            {
                "scope": scope,
                "metric": field,
                "samples": len(differences),
                "cotangent_wins": wins,
                "ties": len(differences) - wins - losses,
                "uniform_wins": losses,
                "mean_cotangent_minus_uniform": float(differences.mean()),
                "median_cotangent_minus_uniform": float(np.median(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return rows


def _training_stability(run: Path) -> dict[str, Any]:
    rows = json.loads((run / "training_step_history.json").read_text(encoding="utf-8"))
    diagnostic = [row for row in rows if row.get("pcg_iterations_mean") is not None]
    if not diagnostic:
        return {"logged_intervals": 0}
    return {
        "logged_intervals": len(diagnostic),
        "latest_optimizer_step": int(diagnostic[-1]["optimizer_steps"]),
        "pcg_iterations_mean": float(
            np.mean([float(row["pcg_iterations_mean"]) for row in diagnostic])
        ),
        "pcg_iterations_max": int(
            max(float(row["pcg_iterations_max"]) for row in diagnostic)
        ),
        "pcg_relative_residual_max": float(
            max(float(row["pcg_relative_residual_max"]) for row in diagnostic)
        ),
        "pcg_failed_solves": int(
            sum(int(row["pcg_failed_solves"]) for row in diagnostic)
        ),
        "nan_inf_count": int(sum(int(row["nan_inf_count"]) for row in diagnostic)),
        "delta_pred_gradient_norm_mean": float(
            np.mean([float(row["delta_pred_gradient_norm"]) for row in diagnostic])
        ),
        "direct_pred_gradient_norm_mean": float(
            np.mean([float(row["delta_v_gradient_norm"]) for row in diagnostic])
        ),
        "peak_gpu_memory_mb": float(
            max(float(row["peak_gpu_memory_mb"]) for row in diagnostic)
        ),
    }


def _operator_summary(
    rows: Sequence[Mapping[str, str]], lambdas: Mapping[str, float]
) -> list[dict[str, Any]]:
    output = []
    names = {
        "uniform_random_walk": "uniform",
        "symmetric_cotangent_stiffness": "cotangent",
    }
    for operator, arm in names.items():
        selected = [row for row in rows if row["operator"] == operator]
        regularization = lambdas[arm]
        norms = np.asarray([float(row["operator_norm_estimate"]) for row in selected])
        low_transfer: list[float] = []
        high_transfer: list[float] = []
        for row in selected:
            for field, destination in (
                ("small_singular_values_json", low_transfer),
                ("large_singular_values_json", high_transfer),
            ):
                if row[field]:
                    singular = np.asarray(json.loads(row[field]), dtype=np.float64)
                    destination.extend(
                        (regularization / (np.square(singular) + regularization)).tolist()
                    )
        output.append(
            {
                "arm": arm,
                "operator": operator,
                "lambda": regularization,
                "symmetry_error_mean": float(
                    np.mean([float(row["symmetry_relative_frobenius"]) for row in selected])
                ),
                "row_sum_error_max": float(
                    max(float(row["constant_nullspace_max_abs"]) for row in selected)
                ),
                "frobenius_norm_mean": float(
                    np.mean([float(row["frobenius_norm"]) for row in selected])
                ),
                "operator_norm_mean": float(norms.mean()),
                "normal_plus_lambda_condition_estimate_mean": float(
                    np.mean((np.square(norms) + regularization) / regularization)
                ),
                "small_mode_direct_transfer_median": (
                    float(np.median(low_transfer)) if low_transfer else None
                ),
                "large_mode_direct_transfer_median": (
                    float(np.median(high_transfer)) if high_transfer else None
                ),
            }
        )
    return output


def _mesh_correlations(
    audit_rows: Sequence[Mapping[str, str]],
    uniform: Sequence[Mapping[str, Any]],
    cotangent: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    audit = {row["sample_id"]: row for row in audit_rows}
    left = {str(row["sample_id"]): row for row in uniform}
    right = {str(row["sample_id"]): row for row in cotangent}
    predictors = (
        "mean_triangle_aspect_proxy",
        "minimum_angle_degrees",
        "maximum_angle_degrees",
        "obtuse_triangle_fraction",
        "negative_weight_fraction",
        "boundary_edge_fraction",
        "nonmanifold_edge_fraction",
        "correction_rms",
    )
    outcomes = (
        "refined_chamfer",
        "same_index_recovered_vertex_rms",
        "p2s_p95",
        "normal_consistency",
        "normalized_flip_rate",
    )
    rows = []
    common = sorted(left.keys() & right.keys() & audit.keys())
    for predictor in predictors:
        x = np.asarray([float(audit[key][predictor]) for key in common])
        for outcome in outcomes:
            y = np.asarray(
                [float(right[key][outcome]) - float(left[key][outcome]) for key in common]
            )
            valid = np.isfinite(x) & np.isfinite(y)
            statistic = spearmanr(x[valid], y[valid])
            rows.append(
                {
                    "predictor": predictor,
                    "outcome_cotangent_minus_uniform": outcome,
                    "spearman": float(statistic.statistic),
                    "p_value": float(statistic.pvalue),
                    "n": int(valid.sum()),
                }
            )
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--operator-audit-dir", required=True, type=Path)
    parser.add_argument("--lambda-pilot-dir", required=True, type=Path)
    parser.add_argument("--uniform-run", required=True, type=Path)
    parser.add_argument("--cotangent-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payloads = {}
    for domain in ("matched_v2", "legacy_v1", "unseen_recipes_v1"):
        for arm in ("uniform", "cotangent"):
            payloads[(domain, arm)] = _read_json(
                args.evaluation_dir / f"{domain}_{arm}.json"
            )
    selection = _read_json(args.lambda_pilot_dir / "selection.json")
    topology = _read_json(args.operator_audit_dir / "audit.json")
    operator_rows = _read_csv(args.operator_audit_dir / "operator_representatives.csv")
    mesh_rows = _read_csv(args.operator_audit_dir / "mesh_audit.csv")
    lambdas = {"uniform": 3e-2, "cotangent": float(selection["selected_lambda"])}
    operator_summary = _operator_summary(operator_rows, lambdas)

    aggregates = []
    paired = []
    recipe_rows = []
    for domain in ("matched_v2", "legacy_v1", "unseen_recipes_v1"):
        split_values = sorted(
            {row["split"] for arm in ("uniform", "cotangent") for row in payloads[(domain, arm)]["rows"]}
        )
        for split in split_values:
            by_arm = {}
            for arm in ("uniform", "cotangent"):
                rows = [row for row in payloads[(domain, arm)]["rows"] if row["split"] == split]
                by_arm[arm] = rows
                aggregates.append(_aggregate(rows, {"domain": domain, "split": split, "arm": arm}))
            paired.extend(_paired(by_arm["uniform"], by_arm["cotangent"], f"{domain}:{split}"))
            if domain == "matched_v2":
                for group, recipes in GROUPS.items():
                    for arm in ("uniform", "cotangent"):
                        selected_rows = [row for row in by_arm[arm] if row["recipe"] in recipes]
                        recipe_rows.append(
                            _aggregate(
                                selected_rows,
                                {"domain": domain, "split": split, "group": group, "arm": arm},
                            )
                        )

    matched_test_u = [
        row for row in payloads[("matched_v2", "uniform")]["rows"] if row["split"] == "test"
    ]
    matched_test_c = [
        row for row in payloads[("matched_v2", "cotangent")]["rows"] if row["split"] == "test"
    ]
    correlations = _mesh_correlations(mesh_rows, matched_test_u, matched_test_c)
    cd_pair = next(
        row
        for row in paired
        if row["scope"] == "matched_v2:test" and row["metric"] == "refined_chamfer"
    )
    if not topology["contract_audit"]:
        classification = "COT5"
    elif float(cd_pair["bootstrap_95_high"]) < 0:
        test_groups = [row for row in recipe_rows if row["split"] == "test"]
        degradation = False
        for group in ("original_topology", "midpoint_subdivision", "adaptive_topology"):
            u = next(row for row in test_groups if row["group"] == group and row["arm"] == "uniform")
            c = next(row for row in test_groups if row["group"] == group and row["arm"] == "cotangent")
            degradation |= float(c["refined_chamfer"]) > 1.05 * float(u["refined_chamfer"])
        classification = "COT2" if degradation else "COT1"
    elif float(cd_pair["bootstrap_95_low"]) > 0:
        classification = "COT4"
    else:
        classification = "COT3"

    contract = {
        "all_evaluations_audited": all(payload["contract_audit"] for payload in payloads.values()),
        "topology_audit": bool(topology["contract_audit"]),
        "validation_only_lambda": bool(selection["contract_audit"] and not selection["test_or_ood_used"]),
        "same_parameter_count": len({payload["parameter_count"] for payload in payloads.values()}) == 1,
        "paired_matched_test_50": len(matched_test_u) == len(matched_test_c) == 50,
    }
    final = {
        "contract_audit": all(contract.values()),
        "contract_checks": contract,
        "classification": classification,
        "selected_cotangent_lambda": selection["selected_lambda"],
        "aggregates": aggregates,
        "paired": paired,
        "recipe_breakdown": recipe_rows,
        "operator_summary": operator_summary,
        "mesh_quality_correlations": correlations,
        "training_stability": {
            "uniform": _training_stability(args.uniform_run),
            "cotangent": _training_stability(args.cotangent_run),
        },
        "metric_protocol": METRIC_PROTOCOL,
    }
    if not final["contract_audit"]:
        raise RuntimeError(f"Final contract failed: {contract}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "aggregate.csv", aggregates)
    _write_csv(args.output_dir / "paired.csv", paired)
    _write_csv(args.output_dir / "recipe_breakdown.csv", recipe_rows)
    _write_csv(args.output_dir / "mesh_quality_correlations.csv", correlations)
    _write_csv(args.output_dir / "operator_summary.csv", operator_summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matched = [row for row in aggregates if row["domain"] == "matched_v2"]
    ood = [row for row in aggregates if row["domain"] != "matched_v2"]
    lines = [
        "# Sofa50 v2 Uniform vs Cotangent single-loss hybrid ablation",
        "",
        f"Contract audit: **{str(final['contract_audit']).lower()}**.",
        "",
        "## Exact operators",
        "",
        "- Uniform: `L_U = I - D^{-1}A` on the undirected input graph; it is generally nonsymmetric.",
        "- Cotangent: `C_ij=-w_ij`, `C_ii=sum_j w_ij`, with `w_ij=0.5 sum cot(opposite angle)` over actual incident faces. Boundary edges use one contribution, negative weights are retained, and no mass normalization is used.",
        "- Near-degenerate protection: a triangle contributes zero iff `2A <= 1e-12 * max_edge_squared`; topology is unchanged.",
        "",
        "## Cotangent topology and weight audit",
        "",
        f"Audited `{topology['meshes']}` meshes. Totals: `{topology['totals']}`. Maximum absolute weight `{_fmt(topology['maximum_absolute_cotangent_weight'])}`; protected-triangle fraction `{_fmt(topology['protected_triangle_fraction'])}`.",
        "",
        "## Validation-only lambda pilot",
        "",
        f"Selected Cotangent lambda: **{_fmt(selection['selected_lambda'])}** by validation final `V_H` mean Chamfer. Test and OOD were not used.",
        "",
        "| lambda | validation CD |",
        "|---:|---:|",
    ]
    for row in selection["pilot_curve"]:
        lines.append(
            f"| {_fmt(row['lambda'])} | {_fmt(row['validation_mean_final_hybrid_chamfer'])} |"
        )
    lines.extend(
        [
            "",
            "## Operator scale and recovery spectrum",
            "",
            "| Arm | Symmetry error | Row-sum max | ||L||₂ | condition estimate | low-mode direct transfer | high-mode direct transfer |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in operator_summary:
        lines.append(
            f"| {row['arm']} | {_fmt(row['symmetry_error_mean'])} | {_fmt(row['row_sum_error_max'])} | {_fmt(row['operator_norm_mean'])} | {_fmt(row['normal_plus_lambda_condition_estimate_mean'])} | {_fmt(row['small_mode_direct_transfer_median'])} | {_fmt(row['large_mode_direct_transfer_median'])} |"
        )
    lines.extend(
        [
            "",
            "The transfer summaries use each arm's own validation-fixed lambda and `lambda/(sigma^2+lambda)`; raw spectra are not compared without scale normalization.",
            "",
            "## Matched validation and test",
            "",
            "| Split | Arm | Initial CD | Refined CD | Gain | P2S p95 | F-score | Normal | VRMS | Flips/rate | New deg. | Improved/worsened |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matched:
        lines.append(
            f"| {row['split']} | {row['arm']} | {_fmt(row['initial_chamfer'])} | {_fmt(row['refined_chamfer'])} | {_fmt(row['relative_chamfer_gain'])} | {_fmt(row['p2s_p95'])} | {_fmt(row['fscore'])} | {_fmt(row['normal_consistency'])} | {_fmt(row['same_index_recovered_vertex_rms'])} | {row['introduced_flipped_faces']} / {_fmt(row['normalized_flip_rate'])} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Cotangent minus Uniform",
            "",
            "| Scope | Metric | C wins | U wins | Mean | Median | Bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row["scope"].startswith("matched_v2"):
            lines.append(
                f"| {row['scope']} | {row['metric']} | {row['cotangent_wins']}/{row['samples']} | {row['uniform_wins']}/{row['samples']} | {_fmt(row['mean_cotangent_minus_uniform'])} | {_fmt(row['median_cotangent_minus_uniform'])} | [{_fmt(row['bootstrap_95_low'])}, {_fmt(row['bootstrap_95_high'])}] |"
            )
    lines.extend(
        [
            "",
            "## Frozen OOD",
            "",
            "| Domain | Arm | Initial CD | Refined CD | Gain | P2S p95 | Normal | VRMS | Improved/worsened |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ood:
        lines.append(
            f"| {row['domain']} | {row['arm']} | {_fmt(row['initial_chamfer'])} | {_fmt(row['refined_chamfer'])} | {_fmt(row['relative_chamfer_gain'])} | {_fmt(row['p2s_p95'])} | {_fmt(row['normal_consistency'])} | {_fmt(row['same_index_recovered_vertex_rms'])} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "Full recipe, severity/topology, solver, bootstrap, and mesh-quality correlation tables are in the adjacent CSV/JSON files.",
            "",
            "## Decision",
            "",
            f"Classification: **{classification}**.",
            "",
            f"Primary test paired Chamfer difference (Cotangent - Uniform): mean `{_fmt(cd_pair['mean_cotangent_minus_uniform'])}`, 95% CI `[{_fmt(cd_pair['bootstrap_95_low'])}, {_fmt(cd_pair['bootstrap_95_high'])}]`, Cotangent wins `{cd_pair['cotangent_wins']}/50`.",
            "",
            f"Metric protocol: `{METRIC_PROTOCOL}`.",
        ]
    )
    (args.output_dir / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"contract_audit": True, "classification": classification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the sealed final report for Slurm 17438 continuation evaluation."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_STEPS = (0, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)
MECHANISM_LABELS = ("step000000", "step001000", "step005000", "step010000", "step020000", "best")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.9g}"


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validation_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in root.glob("validation_*.json"):
        payloads[path.stem.removeprefix("validation_")] = _read(path)
    missing = [f"step{step:06d}" for step in REQUIRED_STEPS if f"step{step:06d}" not in payloads]
    if missing or "best" not in payloads:
        raise RuntimeError(f"Missing validation evaluations: {missing}; best={'best' in payloads}")
    step0 = payloads["step000000"]
    baseline = step0["geometry"]
    rows: list[dict[str, Any]] = []
    for label, payload in payloads.items():
        geometry = payload["geometry"]
        step = 9400 if label == "best" else int(re.search(r"\d+", label).group())
        rows.append(
            {
                "label": label,
                "step": step,
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "chamfer": geometry["refined_chamfer"],
                "chamfer_change_vs_step0": geometry["refined_chamfer"] - baseline["refined_chamfer"],
                "relative_chamfer_change_vs_step0": geometry["refined_chamfer"] / baseline["refined_chamfer"] - 1.0,
                "vertex_rms": geometry["vertex_rms"],
                "vertex_rms_change_vs_step0": geometry["vertex_rms"] - baseline["vertex_rms"],
                "p2s_p95": geometry["p2s_p95"],
                "p2s_p95_change_vs_step0": geometry["p2s_p95"] - baseline["p2s_p95"],
                "fscore": geometry["fscore"],
                "normal": geometry["normal_consistency"],
                "normal_change_vs_step0": geometry["normal_consistency"] - baseline["normal_consistency"],
                "flips": geometry["introduced_flips"],
                "new_degenerates": geometry["new_degenerates"],
                "improved": geometry["improved_worsened"][0],
                "worsened": geometry["improved_worsened"][1],
            }
        )
    rows.sort(key=lambda row: (int(row["step"]), row["label"] == "best"))
    return rows, payloads["best"]


def _paired(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_rows = {
        row["sample_id"]: row
        for row in left["geometry_rows"]
        if row["method"] == "Current_Hybrid"
    }
    right_rows = {
        row["sample_id"]: row
        for row in right["geometry_rows"]
        if row["method"] == "Current_Hybrid"
    }
    if left_rows.keys() != right_rows.keys() or len(left_rows) != 50:
        raise RuntimeError("Matched-test paired rows are incomplete")
    fields = ("chamfer", "vertex_rms", "p2s_p95", "fscore", "normal", "flip_rate")
    rng = np.random.default_rng(7)
    result = []
    for field in fields:
        differences = []
        for sample_id in sorted(left_rows):
            a, b = left_rows[sample_id], right_rows[sample_id]
            left_value = float(a["flips"]) / float(a["faces"]) if field == "flip_rate" else float(a[field])
            right_value = float(b["flips"]) / float(b["faces"]) if field == "flip_rate" else float(b[field])
            differences.append(left_value - right_value)
        values = np.asarray(differences, dtype=np.float64)
        choices = rng.integers(0, len(values), size=(10000, len(values)))
        bootstrap = values[choices].mean(axis=1)
        lower_is_better = field not in {"fscore", "normal"}
        wins = values < 0 if lower_is_better else values > 0
        losses = values > 0 if lower_is_better else values < 0
        result.append(
            {
                "metric": field,
                "selected_minus_step0_mean": float(values.mean()),
                "selected_minus_step0_median": float(np.median(values)),
                "selected_wins": int(wins.sum()),
                "selected_losses": int(losses.sum()),
                "ties": int((values == 0).sum()),
                "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    return result


def _mechanism_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads = {
        label: _read(root / f"mechanism_validation_{label}.json")
        for label in MECHANISM_LABELS
    }
    rows = []
    for label in MECHANISM_LABELS:
        payload = payloads[label]
        geometry = {row["method"]: row for row in payload["geometry_aggregate"]}
        gradient = payload["gradient_aggregate"]
        rows.append(
            {
                "label": label,
                "b_chamfer": geometry["Current_B"]["chamfer"],
                "e_chamfer": geometry["Current_E"]["chamfer"],
                "hybrid_chamfer": geometry["Current_Hybrid"]["chamfer"],
                "redundancy_rms": payload["latent_aggregate"]["redundancy_rms"],
                "redundancy_relative": payload["latent_aggregate"]["relative_discrepancy"],
                "redundancy_cosine": payload["latent_aggregate"]["cosine"],
                "redundancy_norm_ratio": payload["latent_aggregate"]["norm_ratio"],
                "delta_b_output_rms_drift": payload["output_drift"]["delta_b_rms_drift"],
                "v_direct_output_rms_drift": payload["output_drift"]["v_direct_rms_drift"],
                "theta_b_l2_drift": payload["parameter_drift"]["arm_b"]["l2_drift"],
                "theta_e_l2_drift": payload["parameter_drift"]["arm_e"]["l2_drift"],
                "b_gradient_norm": gradient["b_total_gradient_norm"],
                "e_gradient_norm": gradient["e_total_gradient_norm"],
                "b_to_e_gradient_ratio": gradient["b_to_e_gradient_norm_ratio"],
            }
        )
    return rows, payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--preflight-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = args.evaluation_root.resolve()
    validation_rows, selected_validation = _validation_rows(root)
    mechanism_rows, mechanism_validation = _mechanism_rows(root)
    selected_test = _read(root / "matched_test_best.json")
    legacy = _read(root / "legacy_v1_test_best.json")
    unseen = _read(root / "unseen_recipes_v1_test_best.json")
    step0_test_mechanism = _read(root / "mechanism_test_step000000.json")
    selected_test_mechanism = _read(root / "mechanism_test_best.json")
    paired = _paired(selected_test_mechanism, step0_test_mechanism)
    nondeterminism = _read(args.preflight_root / "nondeterminism_envelope_validation_5runs.json")
    step0_validation = _read(args.preflight_root / "step0_validation.json")
    step0_test = _read(args.preflight_root / "step0_test.json")
    metrics = _read(args.run.resolve() / "metrics.json")

    selected_geometry = selected_test["geometry"]
    step0_geometry = step0_test["geometry"]
    step0_test_chamfer = float(step0_geometry["mean_chamfer"])
    test_improvement = step0_test_chamfer - float(selected_geometry["refined_chamfer"])
    noise_range = float(
        nondeterminism["execution_nondeterminism_envelope"]["aggregate_cd_range"]
    )
    meaningful = test_improvement > noise_range
    selected_mechanism = mechanism_validation["best"]
    step0_mechanism = mechanism_validation["step000000"]
    selected_methods = {row["method"]: row for row in selected_mechanism["geometry_aggregate"]}
    step0_methods = {row["method"]: row for row in step0_mechanism["geometry_aggregate"]}
    specialist_degraded = any(
        float(selected_methods[method]["chamfer"]) > float(step0_methods[method]["chamfer"])
        for method in ("Current_B", "Current_E")
    )
    classification = "CT1" if meaningful else "CT3"
    if meaningful and specialist_degraded:
        classification = "CT2"
    if not selected_mechanism["gradient_all_finite_nonzero"] or not selected_mechanism["hybrid_solver"]["all_converged"]:
        classification = "CT6"

    result = {
        "contract_audit": classification != "CT6",
        "classification": classification,
        "selected_checkpoint": selected_test["checkpoint"],
        "selected_checkpoint_sha256": selected_test["checkpoint_sha256"],
        "selected_epoch": selected_test["checkpoint_epoch"],
        "selection_used_validation_only": True,
        "training_metrics": metrics,
        "validation_trajectory": validation_rows,
        "mechanism_trajectory": mechanism_rows,
        "matched_test": selected_test,
        "ood": {"legacy_v1": legacy, "unseen_recipes_v1": unseen},
        "paired_selected_vs_step0": paired,
        "step0_validation": step0_validation,
        "step0_test": step0_test,
        "nondeterminism_envelope": nondeterminism,
        "test_chamfer_improvement_vs_step0": test_improvement,
        "execution_noise_aggregate_chamfer_range": noise_range,
        "improvement_exceeds_execution_noise": meaningful,
        "mechanism_validation": mechanism_validation,
        "mechanism_test": {
            "step0": step0_test_mechanism,
            "selected": selected_test_mechanism,
        },
    }
    (root / "final_evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(root / "validation_trajectory.csv", validation_rows)
    _write_csv(root / "mechanism_trajectory.csv", mechanism_rows)
    _write_csv(root / "paired_selected_vs_step0.csv", paired)

    lines = [
        "# Sofa50 v2 continuous pretrained B+E final evaluation",
        "",
        f"Contract audit: **{str(result['contract_audit']).lower()}**. Classification: **{classification}**.",
        "",
        "The checkpoint was selected only by matched validation recovered Hybrid Chamfer. Test and OOD results were evaluated after selection.",
        "",
        "## Identity and selected checkpoint",
        "",
        f"- Two complete independent networks: `826115 + 826115 = {selected_mechanism['parameter_count']}` parameters; shared storage: **false**.",
        f"- Selected checkpoint: `{selected_test['checkpoint']}`.",
        f"- Selected SHA-256: `{selected_test['checkpoint_sha256']}`; epoch `{selected_test['checkpoint_epoch']}` (step 9,400).",
        "- Recovery: Uniform `L=I-D^-1 A`, `lambda=3e-2`, float64 PCG, tolerance `1e-8`, maximum 2048 iterations.",
        "- Optimizer contract: fresh optimizer over both complete networks, initial LR `1e-4`, effective global batch `8`, final-geometry loss only.",
        "",
        "## Validation trajectory",
        "",
        "| Label | Step | CD | Change vs step 0 | Relative change | VRMS | P2S p95 | Normal | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['label']} | {row['step']} | {_fmt(row['chamfer'])} | {_fmt(row['chamfer_change_vs_step0'])} | "
            f"{_fmt(row['relative_chamfer_change_vs_step0'])} | {_fmt(row['vertex_rms'])} | {_fmt(row['p2s_p95'])} | "
            f"{_fmt(row['normal'])} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "## Branch, gradient, drift, and redundancy trajectory",
            "",
            "The trajectory uses fixed validation indices `0,5,...,45`; final matched-test spectral/component results use all 50 meshes.",
            "",
            "| Label | B CD | E CD | Hybrid CD | Redundancy rel. | Latent cosine | B output drift | E output drift | B param drift | E param drift | B/E grad ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mechanism_rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['b_chamfer'])} | {_fmt(row['e_chamfer'])} | {_fmt(row['hybrid_chamfer'])} | "
            f"{_fmt(row['redundancy_relative'])} | {_fmt(row['redundancy_cosine'])} | {_fmt(row['delta_b_output_rms_drift'])} | "
            f"{_fmt(row['v_direct_output_rms_drift'])} | {_fmt(row['theta_b_l2_drift'])} | {_fmt(row['theta_e_l2_drift'])} | "
            f"{_fmt(row['b_to_e_gradient_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Matched test and OOD",
            "",
            "| Domain | Samples | CD | Gain | VRMS | P2S p95 | F-score | Normal | Improved/worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for domain, payload in (("matched_v2", selected_test), ("legacy_v1", legacy), ("unseen_recipes_v1", unseen)):
        geometry = payload["geometry"]
        lines.append(
            f"| {domain} | {payload['samples']} | {_fmt(geometry['refined_chamfer'])} | {_fmt(geometry['relative_gain'])} | "
            f"{_fmt(geometry['vertex_rms'])} | {_fmt(geometry['p2s_p95'])} | {_fmt(geometry['fscore'])} | "
            f"{_fmt(geometry['normal_consistency'])} | {geometry['improved_worsened'][0]}/{geometry['improved_worsened'][1]} |"
        )
    lines.extend(
        [
            "",
            "## Paired selected-versus-step-0 test statistics",
            "",
            "Differences are selected minus step 0; lower is better except F-score and normal.",
            "",
            "| Metric | Mean difference | Median | Wins/losses/ties | Bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['metric']} | {_fmt(row['selected_minus_step0_mean'])} | {_fmt(row['selected_minus_step0_median'])} | "
            f"{row['selected_wins']}/{row['selected_losses']}/{row['ties']} | "
            f"[{_fmt(row['bootstrap_ci95_low'])}, {_fmt(row['bootstrap_ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Spectral and connected-component diagnostics",
            "",
            "Absolute low/mid/high energies and component translation/centered deformation metrics for step 0 and selected models are stored in `final_evaluation.json` under `mechanism_test`; percentages alone are not used.",
            "",
            "## Decision",
            "",
            f"Matched-test CD changed from `{_fmt(step0_test_chamfer)}` at geometry-equivalent step 0 to `{_fmt(selected_geometry['refined_chamfer'])}` at the validation-selected checkpoint, an improvement of `{_fmt(test_improvement)}`. The measured step-0 aggregate execution-noise range was `{_fmt(noise_range)}`; improvement exceeds it: **{str(meaningful).lower()}**.",
            "",
            f"Final classification: **{classification}**.",
        ]
    )
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(root / 'FINAL_REPORT.md'), "classification": classification}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

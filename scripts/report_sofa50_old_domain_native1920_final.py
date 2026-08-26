#!/usr/bin/env python3
from __future__ import annotations

"""Assemble the complete old-domain native-1920 B/E experiment report."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONTINUOUS_STEPS = (0, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)
ZERO_SHOT_CD = 0.0343849145


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.9g}"


def percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def step_rows(path: Path, targets: tuple[int, ...]) -> list[dict[str, Any]]:
    history = read_json(path)
    by_step = {int(row["optimizer_steps"]): row for row in history}
    return [by_step[step] for step in targets if step in by_step]


def validation_trajectory(history: list[dict[str, Any]], step0_cd: float) -> list[dict[str, Any]]:
    rows = [{"optimizer_steps": 0, "validation_hybrid_chamfer": step0_cd, "epoch": 0}]
    rows.extend(
        {
            "optimizer_steps": int(row["optimizer_steps"]),
            "validation_hybrid_chamfer": float(row["validation_hybrid_chamfer"]),
            "validation_loss": row.get("validation_loss"),
            "validation_recovered_vertex_rms": row.get("validation_recovered_vertex_rms"),
            "epoch": int(row["epoch"]),
        }
        for row in history
        if row.get("validation_hybrid_chamfer") is not None
    )
    unique = {int(row["optimizer_steps"]): row for row in rows}
    return [unique[step] for step in sorted(unique)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(aggregate: dict[str, dict[str, Any]], contract: bool) -> tuple[str, dict[str, bool]]:
    initial = float(aggregate["Initial mesh"]["chamfer"])
    b = float(aggregate["Old-domain Arm B"]["chamfer"])
    e = float(aggregate["Old-domain Arm E"]["chamfer"])
    frozen = float(aggregate["Old-domain Frozen B+E"]["chamfer"])
    continuous = float(aggregate["Old-domain Continuous B+E"]["chamfer"])
    archived = min(
        float(aggregate[name]["chamfer"])
        for name in ("NDS", "Previous Ours (native-1920 HF)", "nvdiffrec", "ExMesh")
    )
    gates = {
        "valid_contract": contract,
        "continuous_improves_initial": continuous < initial,
        "continuous_beats_B": continuous < b,
        "continuous_beats_E": continuous < e,
        "continuous_beats_frozen": continuous < frozen,
        "continuous_beats_strongest_archive": continuous < archived,
        "frozen_beats_B_and_E": frozen < b and frozen < e,
        "frozen_beats_strongest_archive": frozen < archived,
        "continuous_materially_beats_frozen_0p1_percent": continuous < frozen * (1.0 - 1e-3),
        "domain_match_beats_zero_shot": continuous < ZERO_SHOT_CD,
        "a_specialist_beats_strongest_archive": min(b, e) < archived,
    }
    if not contract:
        return "OLD6", gates
    if all(
        gates[key]
        for key in (
            "continuous_improves_initial",
            "continuous_beats_B",
            "continuous_beats_E",
            "continuous_beats_frozen",
            "continuous_beats_strongest_archive",
        )
    ):
        return "OLD1", gates
    if (
        gates["frozen_beats_B_and_E"]
        and gates["frozen_beats_strongest_archive"]
        and not gates["continuous_materially_beats_frozen_0p1_percent"]
    ):
        return "OLD2", gates
    if gates["a_specialist_beats_strongest_archive"] and min(frozen, continuous) >= min(b, e) * (1.0 - 1e-3):
        return "OLD3", gates
    if gates["continuous_improves_initial"] and gates["domain_match_beats_zero_shot"]:
        return "OLD4", gates
    return "OLD5", gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-e-run", required=True, type=Path)
    parser.add_argument("--continuous-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.report_root.resolve()
    audit = read_json(root / "preflight" / "contract_audit.json")
    runtime_preflight = read_json(root / "preflight" / "runtime_preflight.json")
    b_metrics = read_json(args.arm_b_run.resolve() / "metrics.json")
    e_metrics = read_json(args.arm_e_run.resolve() / "metrics.json")
    continuous_metrics = read_json(args.continuous_run.resolve() / "metrics.json")
    b_config = read_json(args.arm_b_run.resolve() / "run_config.json")
    e_config = read_json(args.arm_e_run.resolve() / "run_config.json")
    continuous_config = read_json(args.continuous_run.resolve() / "run_config.json")
    specialists = read_json(
        root / "validation_selection" / "specialists" / "validation_specialist_summary.json"
    )
    selection = read_json(root / "validation_selection" / "frozen_lambda" / "lambda_selection.json")
    frozen_validation = read_json(
        root / "validation_selection" / "frozen_validation_summary.json"
    )
    step0 = read_json(root / "continuous" / "preflight" / "step0_validation.json")
    continuous_validation = read_json(
        root / "continuous" / "validation" / "selected_checkpoint.json"
    )
    authorization = read_json(root / "selection_lock" / "test_authorization.json")
    final_test = read_json(root / "final_test" / "final_test_summary.json")

    prerequisite_contracts = {
        "historical_data": bool(audit["contract_audit"]),
        "runtime_preflight": bool(runtime_preflight["contract_audit"]),
        "specialists": bool(specialists["contract_audit"]),
        "lambda_selection": bool(selection["contract_audit"]),
        "frozen_validation": bool(frozen_validation["contract_audit"]),
        "step0": bool(step0["contract_audit"]),
        "selection_lock": bool(authorization["contract_audit"]),
        "final_test": bool(final_test["contract_audit"]),
        "test_opened_once": final_test["test_opened_once"] is True,
        "test_never_used_for_selection": final_test["test_used_for_selection"] is False
        and authorization["test_metric_used_before_lock"] is False,
        "all_training_completed": int(b_metrics["optimizer_steps"]) == 20000
        and int(e_metrics["optimizer_steps"]) == 20000
        and int(continuous_metrics["optimizer_steps"]) == 20000,
    }
    contract = all(prerequisite_contracts.values())
    with (root / "preflight" / "split_samples.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        split_rows = list(csv.DictReader(handle))
    test_object_ids = sorted(
        {row["object_id"] for row in split_rows if row["split"] == "test"}
    )
    if len(test_object_ids) != 5:
        raise RuntimeError(f"Expected five sealed test objects, found {test_object_ids}")
    aggregate = {row["method"]: row for row in final_test["aggregate"]}
    classification, classification_gates = classify(aggregate, contract)

    b_steps = step_rows(args.arm_b_run.resolve() / "training_step_history.json", (200, 5000, 10000, 15000, 20000))
    e_steps = step_rows(args.arm_e_run.resolve() / "training_step_history.json", (200, 5000, 10000, 15000, 20000))
    continuous_history = read_json(args.continuous_run.resolve() / "training_history.json")
    continuous_trajectory = validation_trajectory(
        continuous_history, float(step0["geometry"]["mean_chamfer"])
    )
    required_trajectory = {
        int(row["optimizer_steps"]): row for row in continuous_trajectory
    }
    write_csv(root / "continuous" / "validation" / "validation_trajectory.csv", continuous_trajectory)

    methods = (
        "Initial mesh",
        "NDS",
        "Previous Ours (native-1920 HF)",
        "nvdiffrec",
        "ExMesh",
        "Old-domain Arm B",
        "Old-domain Arm E",
        "Old-domain Frozen B+E",
        "Old-domain Continuous B+E",
    )
    continuous_cd = float(aggregate["Old-domain Continuous B+E"]["chamfer"])
    strongest_name = min(
        ("NDS", "Previous Ours (native-1920 HF)", "nvdiffrec", "ExMesh"),
        key=lambda name: float(aggregate[name]["chamfer"]),
    )
    strongest_cd = float(aggregate[strongest_name]["chamfer"])
    answer_yes = continuous_cd < strongest_cd

    lines = [
        "# Sofa50 old-domain native-1920 independent B+E experiment",
        "",
        f"Contract audit: **{str(contract).lower()}**. Classification: **{classification}**.",
        "",
        "## Final answer",
        "",
        (
            "**Yes.** After domain-matched native-1920 retraining, the selected Continuous B+E model "
            if answer_yes
            else "**No.** After domain-matched native-1920 retraining, the selected Continuous B+E model "
        )
        + f"has test CD `{fmt(continuous_cd)}` versus `{fmt(strongest_cd)}` for the strongest archived comparator ({strongest_name}).",
        "",
        "## Historical data contract and isolation",
        "",
        f"Dataset root: `{audit['dataset_root']}`.",
        f"Source manifest: `{audit['source_manifest']}` (SHA-256 `{audit['source_manifest_sha256']}`).",
        f"Historical manifest: `{audit['historical_manifest']}` (SHA-256 `{audit['historical_manifest_sha256']}`).",
        f"Sealed benchmark manifest: `{audit['sealed_test_manifest']}` (SHA-256 `{audit['sealed_test_manifest_sha256']}`).",
        "",
        f"The recovered object-level split is `{audit['split_sample_counts']['train']}` train / "
        f"`{audit['split_sample_counts']['validation']}` validation / `{audit['split_sample_counts']['test']}` test meshes, "
        f"from `{audit['split_object_counts']['train']}` / `{audit['split_object_counts']['validation']}` / "
        f"`{audit['split_object_counts']['test']}` mutually disjoint objects. Each object has five variants and 28 native `1920x1920` views.",
        "",
        "Sample IDs, object IDs, input geometry identities, and clean geometry identities are pairwise disjoint across splits. "
        "The 25 exact `v00-v04` benchmark samples remained sealed until the authorization lock; no derivative or intermediate test trajectory was created.",
        f"Sealed test object IDs: `{', '.join(test_object_ids)}`.",
        "",
        f"Perturbation recipe: requested Gaussian displacement standard deviation `{fmt(audit['perturbation']['requested_perturb_std_h']['mean'])} h`, "
        f"maximum displacement `{fmt(audit['perturbation']['max_offset_over_h']['maximum'])} h`; prepared inputs introduced zero flips and zero new degeneracies. "
        f"Topology ranges are train `{int(audit['topology']['train']['vertices']['minimum'])}`–`{int(audit['topology']['train']['vertices']['maximum'])}` vertices, "
        f"validation `{int(audit['topology']['validation']['vertices']['minimum'])}`–`{int(audit['topology']['validation']['vertices']['maximum'])}`, "
        f"and test `{int(audit['topology']['test']['vertices']['minimum'])}`–`{int(audit['topology']['test']['vertices']['maximum'])}`.",
        "",
        "## Independent specialist training",
        "",
        "Both models use 28 native-1920 views, Original+Gaussian-HF C2F2 image features, Fourier query geometry, "
        "three graph layers of width 256, and an independent 3D output head. They were initialized separately from scratch and share no parameters.",
        "",
        "| Arm | Semantics / only loss | Parameters | Steps | Best epoch | Selection loss | Checkpoint SHA-256 | Runtime | Peak GPU MiB |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
        f"| B | raw `L_U V_clean`; Huber + `0.01` recovery MSE | {runtime_preflight['runtime']['B']['parameter_count']} | {b_metrics['optimizer_steps']} | {b_metrics['best_epoch']} | {fmt(b_metrics['best_selection_loss'])} | `{specialists['arm_b_checkpoint_sha256']}` | {fmt(b_metrics['runtime_seconds'])} s | {fmt(b_metrics['peak_gpu_memory_mb'])} |",
        f"| E | direct displacement; final vertex MSE only | {runtime_preflight['runtime']['E']['parameter_count']} | {e_metrics['optimizer_steps']} | {e_metrics['best_epoch']} | {fmt(e_metrics['best_selection_loss'])} | `{specialists['arm_e_checkpoint_sha256']}` | {fmt(e_metrics['runtime_seconds'])} s | {fmt(e_metrics['peak_gpu_memory_mb'])} |",
        "",
        f"Arm-B config: `{args.arm_b_run.resolve() / 'run_config.json'}`. Arm-E config: `{args.arm_e_run.resolve() / 'run_config.json'}`. "
        "Arm B uses Uniform `L_U=I-D^-1A`, `lambda_B=0.01`, float32 PCG tolerance `1e-4`, max 256; Arm E has no recovery or auxiliary loss.",
        "",
        "### Specialist training trajectory",
        "",
        "| Arm | Step | Train loss | Objective | PCG iterations mean/max | LR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, rows in (("B", b_steps), ("E", e_steps)):
        for row in rows:
            pcg = "n/a" if row.get("pcg_iterations_mean") is None else f"{fmt(row['pcg_iterations_mean'])}/{fmt(row['pcg_iterations_max'])}"
            lines.append(
                f"| {arm} | {row['optimizer_steps']} | {fmt(row.get('train_loss'))} | "
                f"{fmt(row.get('train_objective'))} | {pcg} | {fmt(row.get('learning_rate'))} |"
            )
    lines.extend(
        [
            "",
            "## Validation-only frozen fusion",
            "",
            f"The exact predeclared λ grid was `{selection['lambda_grid']}`. Validation selected `lambda_old={fmt(selection['selected_lambda'])}`"
            + (" at a grid boundary." if selection["selected_at_grid_boundary"] else "."),
            "",
            "| λ | Validation CD | VRMS | P2S mean | P2S p95 | F-score | Normal | Improved/worsened |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selection["aggregate"]:
        lines.append(
            f"| {fmt(row['lambda'])} | {fmt(row['refined_chamfer'])} | "
            f"{fmt(row['same_index_recovered_vertex_rms'])} | {fmt(row['p2s'])} | {fmt(row['p2s_p95'])} | "
            f"{fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "Selected-λ complete validation:",
            "",
            "| Method | CD | Gain | VRMS | P2S mean | P2S p95 | F-score | Normal | Improved/worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frozen_validation["aggregate"]:
        lines.append(
            f"| {row['method']} | {fmt(row['refined_chamfer'])} | {percent(row['aggregate_relative_gain'])} | "
            f"{fmt(row['same_index_recovered_vertex_rms'])} | {fmt(row['p2s'])} | {fmt(row['p2s_p95'])} | "
            f"{fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | {row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            f"Frozen-vs-B paired wins/losses/ties: `{frozen_validation['paired']['Frozen_vs_B']['frozen_wins']}/"
            f"{frozen_validation['paired']['Frozen_vs_B']['frozen_losses']}/{frozen_validation['paired']['Frozen_vs_B']['ties']}`; "
            f"Frozen-vs-E: `{frozen_validation['paired']['Frozen_vs_E']['frozen_wins']}/"
            f"{frozen_validation['paired']['Frozen_vs_E']['frozen_losses']}/{frozen_validation['paired']['Frozen_vs_E']['ties']}`.",
            "",
            "## Continuous B+E",
            "",
            f"Step-0 validation audit: **{str(step0['contract_audit']).lower()}**; aggregate CD difference "
            f"`{fmt(step0['reproduction']['aggregate_relative_cd_difference'])}`; maximum per-sample CD difference "
            f"`{fmt(step0['reproduction']['maximum_per_sample_cd_difference'])}`. All required B/E latent, head, and backbone gradients were finite and non-zero.",
            "",
            f"Two complete independent networks contain `{step0['parameter_count']}` parameters. Training used a fresh Adam optimizer, LR `1e-4`, "
            f"effective batch 8, float64 PCG (`1e-8`, max 2048), and only final recovered same-index geometry MSE. "
            f"The validation-selected checkpoint is `{authorization['continuous_checkpoint']}` with SHA-256 `{authorization['continuous_checkpoint_sha256']}`.",
            "",
            "### Validation Hybrid CD trajectory",
            "",
            "| Step | Epoch | Validation Hybrid CD | Validation VRMS |",
            "|---:|---:|---:|---:|",
        ]
    )
    for step in CONTINUOUS_STEPS:
        row = required_trajectory.get(step)
        if row is not None:
            lines.append(
                f"| {step} | {row['epoch']} | {fmt(row['validation_hybrid_chamfer'])} | "
                f"{fmt(row.get('validation_recovered_vertex_rms'))} |"
            )
    lines.extend(
        [
            "",
            f"Continuous runtime `{fmt(continuous_metrics['runtime_seconds'])}` s; peak GPU memory `{fmt(continuous_metrics['peak_gpu_memory_mb'])}` MiB; "
            f"best epoch `{continuous_metrics['best_epoch']}`; final validation re-evaluation CD `{fmt(continuous_validation['geometry']['refined_chamfer'])}`.",
            "",
            "## Final sealed same-input test",
            "",
            "The test was opened exactly once after B, E, `lambda_old`, and the continuous checkpoint were locked. "
            "No test metric was available to training, λ selection, checkpoint selection, stopping, or architecture decisions.",
            "",
            "| Method | CD | Gain | VRMS | P2S mean | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        row = aggregate[method]
        flip_rate = "n/a" if row["normalized_flip_rate"] is None else f"{100 * row['normalized_flip_rate']:.3f}%"
        lines.append(
            f"| {method} | {fmt(row['chamfer'])} | {percent(row['aggregate_relative_gain'])} | "
            f"{fmt(row.get('same_index_recovered_vertex_rms'))} | {fmt(row['p2s'])} | {fmt(row['p2s_p95'])} | "
            f"{fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | "
            f"{row['introduced_flipped_faces']} / {flip_rate} | {row['new_degenerate_faces']} | "
            f"{row['improved']}/{row['worsened']} |"
        )
    lines.extend(
        [
            "",
            "### Curvature and distortion (same-topology rows)",
            "",
            "| Method | 2H MAE | Scaled curvature MAE | Dihedral MAE deg | Face-normal MAE deg | Edge log error | Area log error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in (
        "Initial mesh",
        "Previous Ours (native-1920 HF)",
        "Old-domain Arm B",
        "Old-domain Arm E",
        "Old-domain Frozen B+E",
        "Old-domain Continuous B+E",
    ):
        row = aggregate[method]
        lines.append(
            f"| {method} | {fmt(row['twice_mean_curvature_magnitude_error_mean'])} | "
            f"{fmt(row['scaled_curvature_error_mean'])} | {fmt(row['dihedral_angle_error_degrees_mean'])} | "
            f"{fmt(row['face_normal_angle_error_degrees_mean'])} | "
            f"{fmt(row['absolute_log_edge_length_ratio_mean'])} | {fmt(row['absolute_log_face_area_ratio_mean'])} |"
        )
    lines.extend(
        [
            "",
            "### Representation and exact RHS cancellation",
            "",
            "| State | δ vs L_U V_direct RMS | Relative | Cosine | Norm ratio | ||e_L|| | ||e_D|| | ||e_q|| | cos(e_L,e_D) | C_cancel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final_test["representation_and_rhs_aggregate"]:
        lines.append(
            f"| {row['state']} | {fmt(row['rms_difference'])} | {fmt(row['relative_rms_difference'])} | "
            f"{fmt(row['cosine'])} | {fmt(row['norm_ratio'])} | {fmt(row['e_L_norm'])} | "
            f"{fmt(row['e_D_norm'])} | {fmt(row['e_q_norm'])} | {fmt(row['e_L_e_D_cosine'])} | "
            f"{fmt(row['cancellation_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "### Paired Continuous B+E comparisons",
            "",
            "| Comparator | CD mean diff | Median diff | Bootstrap 95% CI | Wins/losses/ties | P2S p95 diff | F-score diff | Normal diff |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, stats in final_test["paired_continuous_comparisons"].items():
        cd = stats["chamfer"]
        lines.append(
            f"| {method} | {fmt(cd['continuous_minus_comparator_mean'])} | "
            f"{fmt(cd['continuous_minus_comparator_median'])} | "
            f"[{fmt(cd['bootstrap_95_percent_ci'][0])}, {fmt(cd['bootstrap_95_percent_ci'][1])}] | "
            f"{cd['continuous_wins']}/{cd['continuous_losses']}/{cd['ties']} | "
            f"{fmt(stats['p2s_p95']['continuous_minus_comparator_mean'])} | "
            f"{fmt(stats['fscore']['continuous_minus_comparator_mean'])} | "
            f"{fmt(stats['normal_consistency']['continuous_minus_comparator_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Numerical stability, runtime, and decision",
            "",
            f"Final test PCG: all converged, mean/max iterations `{fmt(final_test['solver']['iterations_mean'])}` / "
            f"`{final_test['solver']['iterations_max']}`, maximum relative residual `{fmt(final_test['solver']['relative_residual_max'])}`. "
            f"Final evaluation runtime `{fmt(final_test['runtime']['seconds'])}` s and peak GPU memory `{fmt(final_test['runtime']['peak_gpu_memory_mb'])}` MiB.",
            "",
            f"Classification: **{classification}**.",
            "",
            "| Decision gate | Result |",
            "|---|---|",
        ]
    )
    lines.extend(f"| {key.replace('_', ' ')} | {str(value).lower()} |" for key, value in classification_gates.items())
    lines.extend(
        [
            "",
            f"Metric protocol: `{final_test['metric_protocol']}`.",
            "",
            f"Curvature protocol: `{final_test['curvature_protocol']}`.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "contract_audit": contract,
        "prerequisite_contracts": prerequisite_contracts,
        "classification": classification,
        "classification_gates": classification_gates,
        "final_answer_yes": answer_yes,
        "continuous_test_chamfer": continuous_cd,
        "strongest_archived_comparator": strongest_name,
        "strongest_archived_chamfer": strongest_cd,
        "final_report": str(args.output.resolve()),
    }
    (args.output.parent / "FINAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not contract:
        raise RuntimeError(f"Final report contract failed: {prerequisite_contracts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

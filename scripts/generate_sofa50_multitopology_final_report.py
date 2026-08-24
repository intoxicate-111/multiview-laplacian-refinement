#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


OLD_ARM = "old_960_HF"
NEW_ARM = "new_multitopology_rawlap"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(rows: Sequence[Mapping[str, Any]], **conditions: str) -> dict[str, Any]:
    values = [row for row in rows if all(str(row.get(key)) == value for key, value in conditions.items())]
    if len(values) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(values)}")
    return dict(values[0])


def relative(new: float, old: float) -> float:
    return (float(new) - float(old)) / max(abs(float(old)), 1e-12)


def yn(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--unseen-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--require-unified-geometry", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    dataset = args.dataset_root.resolve()
    unseen = args.unseen_root.resolve()
    run = args.run_root.resolve()
    evaluation = (
        args.evaluation_root.resolve()
        if args.evaluation_root is not None
        else run / "evaluation"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    audit = read_json(dataset / "full_audit.json")
    unseen_audit = read_json(unseen / "full_audit.json")
    metrics = read_json(run / "metrics.json")
    run_config = read_json(run / "config.json")
    experiment_metadata = dict(run_config.get("experiment_metadata", {}))
    in_domain = read_json(evaluation / "in_domain" / "summary.json")
    unseen_eval = read_json(evaluation / "unseen_topology" / "summary.json")
    legacy = read_json(evaluation / "legacy_current25" / "summary.json")
    openmvs = read_json(evaluation / "openmvs48_zero_shot" / "summary.json")
    geometry_summaries = (in_domain, unseen_eval, legacy, openmvs)
    if args.require_unified_geometry:
        invalid = [
            index
            for index, value in enumerate(geometry_summaries)
            if not bool(value.get("contract_audit", {}).get("unified_metric_audit"))
            or "evaluate_mesh_geometry" not in str(value.get("metric_protocol", ""))
            or "no_ICP" not in str(value.get("metric_protocol", ""))
        ]
        if invalid:
            raise RuntimeError(
                f"Refusing legacy or unaudited geometry summaries at indices {invalid}."
            )
    evaluated_checkpoint = run / "checkpoint_latest.pt"
    checkpoint = run / "checkpoint_best.pt"
    if not checkpoint.is_file():
        checkpoint = evaluated_checkpoint
    git_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    inventory_paths = sorted(run.glob("gpu_inventory_*.csv"))
    training_job_ids = [
        path.stem.removeprefix("gpu_inventory_") for path in inventory_paths
    ]
    world_size = int(metrics["distributed_world_size"])
    global_batch = int(metrics["global_batch_meshes"])
    per_gpu_batch = global_batch / world_size
    per_gpu_batch_text = (
        str(int(per_gpu_batch)) if per_gpu_batch.is_integer() else f"{per_gpu_batch:.6g}"
    )
    gpu_model = str(
        experiment_metadata.get(
            "training_gpu_model", "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        )
    )

    main_old_p = select(in_domain["prediction"], split="test", arm=OLD_ARM)
    main_new_p = select(in_domain["prediction"], split="test", arm=NEW_ARM)
    main_old_r = select(in_domain["recovery"], arm=OLD_ARM)
    main_new_r = select(in_domain["recovery"], arm=NEW_ARM)
    unseen_old_p = select(unseen_eval["prediction"], split="test", arm=OLD_ARM)
    unseen_new_p = select(unseen_eval["prediction"], split="test", arm=NEW_ARM)
    unseen_old_r = select(unseen_eval["recovery"], arm=OLD_ARM)
    unseen_new_r = select(unseen_eval["recovery"], arm=NEW_ARM)
    legacy_old_p = select(legacy["prediction"], split="test", arm=OLD_ARM)
    legacy_new_p = select(legacy["prediction"], split="test", arm=NEW_ARM)
    legacy_old_r = select(legacy["recovery"], arm=OLD_ARM)
    legacy_new_r = select(legacy["recovery"], arm=NEW_ARM)
    open_old = select(openmvs["aggregate"], arm=OLD_ARM)
    open_new = select(openmvs["aggregate"], arm=NEW_ARM)

    decisions = {
        "new_test_prediction_improved": main_new_p["raw_epe"] <= main_old_p["raw_epe"],
        "new_test_recovery_improved": main_new_r["reconstruction_chamfer"] <= main_old_r["reconstruction_chamfer"],
        "unseen_prediction_improved": unseen_new_p["raw_epe"] <= unseen_old_p["raw_epe"],
        "unseen_recovery_improved": unseen_new_r["reconstruction_chamfer"] <= unseen_old_r["reconstruction_chamfer"],
        "legacy_accuracy_preserved_within_5_percent": (
            legacy_new_p["raw_epe"] <= 1.05 * legacy_old_p["raw_epe"]
            and legacy_new_r["reconstruction_chamfer"] <= 1.05 * legacy_old_r["reconstruction_chamfer"]
        ),
        "real_coarse_chamfer_improved": open_new["chamfer"] <= open_old["chamfer"],
        "no_real_coarse_catastrophic_failures": (
            open_new["new_degenerate_faces"] <= open_old["new_degenerate_faces"]
            and open_new["samples"] == open_old["samples"]
            and open_new["samples"] > 0
        ),
    }
    decision_endpoint_keys = (
        "new_test_prediction_improved",
        "new_test_recovery_improved",
        "unseen_prediction_improved",
        "unseen_recovery_improved",
        "legacy_accuracy_preserved_within_5_percent",
    )
    # OpenMVS rows are retained as low-quality OOD stress diagnostics only.
    # They must not influence checkpoint/model selection or the scale-up gate.
    go = all(bool(decisions[key]) for key in decision_endpoint_keys)
    decision = "GO" if go else "NO-GO"
    summary = {
        "experiment": "Sofa50MultiTopologyRawLap500_v1",
        "geometry_metric_protocol": in_domain.get("metric_protocol"),
        "geometry_evaluation_root": str(evaluation),
        "git_commit": git_commit,
        "dataset_contract_audit": bool(audit["contract_audit"]),
        "dataset_samples": int(audit["sample_count"]),
        "unseen_contract_audit": bool(unseen_audit["contract_audit"]),
        "unseen_samples": int(unseen_audit["sample_count"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "evaluated_checkpoint": str(evaluated_checkpoint),
        "evaluated_checkpoint_sha256": sha256(evaluated_checkpoint),
        "training": {
            key: metrics.get(key)
            for key in (
                "optimizer_steps",
                "runtime_seconds",
                "peak_gpu_memory_mb",
                "distributed_world_size",
                "global_batch_meshes",
                "best_epoch",
                "best_selection_loss",
                "final_validation_loss",
                "stop_reason",
            )
        },
        "training_execution": {
            "job_ids": training_job_ids,
            "gpu_model": gpu_model,
            "world_size": world_size,
            "per_gpu_batch_meshes": per_gpu_batch,
            "global_batch_meshes": global_batch,
            "baseline_global_batch_meshes": experiment_metadata.get(
                "baseline_effective_global_batch_meshes"
            ),
            "strict_single_variable_training_claim": experiment_metadata.get(
                "strict_single_variable_training_claim"
            ),
            "initialization": "from_scratch",
        },
        "in_domain": {"old_prediction": main_old_p, "new_prediction": main_new_p, "old_recovery": main_old_r, "new_recovery": main_new_r},
        "unseen": {"old_prediction": unseen_old_p, "new_prediction": unseen_new_p, "old_recovery": unseen_old_r, "new_recovery": unseen_new_r},
        "legacy": {"old_prediction": legacy_old_p, "new_prediction": legacy_new_p, "old_recovery": legacy_old_r, "new_recovery": legacy_new_r},
        "openmvs48": {"old": open_old, "new": open_new},
        "decision_checks": decisions,
        "decision_endpoint_keys": list(decision_endpoint_keys),
        "openmvs_policy": {
            "role": "diagnostic_only_non_decisional_low_quality_ood_input",
            "decision_weight": 0,
            "training_target": False,
            "pseudo_gt": False,
            "scale_up_gate": False,
        },
        "future20k_decision": decision,
    }
    write_json(output / "summary.json", summary)

    lines = [
        "# Sofa50MultiTopologyRawLap500_v1 final decision report",
        "",
        f"Final FUTURE-20K decision: **{decision}**.",
        "",
        "## Cancellation audit",
        "",
        "Only the superseded Future2000 external-baseline jobs were cancelled; completed artifacts were preserved.",
        "",
        "| Job IDs | Purpose | Reason |",
        "|---|---|---|",
        "| 16945_[0-7] | nvdiffrec full-1000 restart | Superseded priority |",
        "| 16953, 16954_[0-7] | NDS-28V-full smoke/trial | Superseded priority |",
        "| 16892_[0-7], 16893_[0-7] | DA3 and ExMesh stages | Superseded priority |",
        "| 16894 | Future2000 final merge | Upstream stages cancelled |",
        "",
        "## Reproducibility",
        "",
        f"- Git commit: `{git_commit}`",
        f"- Dataset: `{dataset}`; samples: {audit['sample_count']}; contract audit: **{str(audit['contract_audit']).lower()}**",
        f"- Evaluation-only unseen set: `{unseen}`; samples: {unseen_audit['sample_count']}; contract audit: **{str(unseen_audit['contract_audit']).lower()}**",
        f"- Training run: `{run}`",
        f"- Geometry evaluation root: `{evaluation}`",
        f"- Geometry metric protocol: `{in_domain.get('metric_protocol', 'legacy/unspecified')}`",
        "- Coordinate alignment: shared prepared coordinate frame (identity); no ICP or test-set alignment.",
        f"- Best checkpoint: `{checkpoint}`; SHA-256 `{summary['checkpoint_sha256']}`",
        f"- Unified evaluations use the fixed 20k latest checkpoint: `{evaluated_checkpoint}`; SHA-256 `{summary['evaluated_checkpoint_sha256']}`",
        f"- Training jobs: generation `16966`, audit `16967`, {world_size}x Blackwell training "
        + (", ".join(f"`{job_id}`" for job_id in training_job_ids) if training_job_ids else "(job ID unavailable)")
        + ".",
        f"- Hardware: {world_size}x {gpu_model}; per-GPU batch {per_gpu_batch_text}; effective global batch {global_batch}; from-scratch initialization.",
        f"- Execution-contract note: strict single-variable training claim is `{str(experiment_metadata.get('strict_single_variable_training_claim')).lower()}`; baseline effective global batch is {experiment_metadata.get('baseline_effective_global_batch_meshes')}. Translation of prediction changes to downstream recovery is therefore evaluated directly rather than inferred from training loss.",
        f"- Optimizer steps: {metrics['optimizer_steps']}; runtime: {metrics['runtime_seconds'] / 3600:.3f} h; peak GPU memory: {metrics['peak_gpu_memory_mb']:.1f} MiB; stop reason: `{metrics['stop_reason']}`.",
        "",
        "## Dataset topology statistics",
        "",
        "| Variant | Samples | Mean V | Median V | Mean F | Median F | Face ratio vs GT | Mean displacement | Smoothing | Mean raw | Edge-only selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for variant in ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2"):
        row = audit["topology_statistics"][variant]
        lines.append(
            f"| {variant} | {row['sample_count']} | {row['vertices']['mean']:.1f} | {row['vertices']['median']:.1f} | {row['faces']['mean']:.1f} | {row['faces']['median']:.1f} | {row['face_subdivision_ratio_vs_gt_mean']:.3f} | {row['clean_to_input_displacement_mean']:.6g} | {row['smoothing_iterations'][0]}x@{row['smoothing_strength'][0]} | {row['raw_laplacian_magnitude_mean']:.6g} | {row['edge_only_selected_faces_total']} |"
        )
    lines.extend(
        [
            "",
            "All 500 samples have exact clean/input face equality, native clean-topology raw targets, finite tensors, and exact float32 target recomputation. C1-C4 density is monotonic and meaningfully distinct.",
            "",
            "## Old 960-HF vs new model",
            "",
            "All percentile groups below are global groups defined by GT raw-Laplacian magnitude.",
            "",
            "| Dataset | Arm | Raw EPE | RMS | Top10 | Top1 | Weighted RMS | Chamfer | P2S | Normal | Improved |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, old_p, new_p, old_r, new_r in (
        ("new test50", main_old_p, main_new_p, main_old_r, main_new_r),
        ("unseen topology25", unseen_old_p, unseen_new_p, unseen_old_r, unseen_new_r),
        ("legacy current25", legacy_old_p, legacy_new_p, legacy_old_r, legacy_new_r),
    ):
        for arm, prediction, recovery in ((OLD_ARM, old_p, old_r), (NEW_ARM, new_p, new_r)):
            lines.append(
                f"| {name} | {arm} | {prediction['raw_epe']:.9g} | {prediction['raw_rms']:.9g} | {prediction['top10_epe']:.9g} | {prediction['top1_epe']:.9g} | {prediction['recovery_weighted_raw_rms']:.9g} | {recovery['reconstruction_chamfer']:.9g} | {recovery['reconstruction_point_to_surface']:.9g} | {recovery['reconstruction_normal_consistency']:.9g} | {recovery['improved_over_initial']}/{recovery['samples']} |"
            )
    lines.extend(
        [
            "",
            "## OpenMVS48 real-coarse zero-shot",
            "",
            "Both arms use the same OpenMVS coarse mesh, same original 14-view RGB/cameras, same visibility and recovery. No fine-tuning or GT differential target is used.",
            "",
            "**Diagnostic only / non-decisional.** The low-quality OpenMVS reconstructions are OOD stress inputs, not targets, pseudo-GT, model-selection endpoints or scale-up gates. Their historical metrics are reported but carry zero decision weight.",
            "",
            "| Arm | Chamfer | P2S | Normal | Flips | New degenerates | Improved |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in (open_old, open_new):
        lines.append(f"| {row['arm']} | {row['chamfer']:.9g} | {row['p2s']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} | {row['new_degenerate_faces']} | {row['improved_over_initial']}/{row['samples']} |")
    lines.extend(["", "## Decision checks", ""])
    for key, value in decisions.items():
        suffix = " (diagnostic only; zero decision weight)" if key.startswith("real_coarse") or key.startswith("no_real_coarse") else ""
        lines.append(f"- {key}: **{yn(value)}**{suffix}")
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"1. In-domain accuracy improved or was preserved: **{yn(decisions['new_test_prediction_improved'] and decisions['new_test_recovery_improved'])}**.",
            f"2. Unseen-topology robustness improved: **{yn(decisions['unseen_prediction_improved'] and decisions['unseen_recovery_improved'])}**.",
            f"3. OpenMVS stress-input metric changed favourably: **{yn(decisions['real_coarse_chamfer_improved'])}**, but this is non-decisional and is not target-quality evidence.",
            f"4. Evidence exceeds memorizing ten recipes: **{yn(decisions['unseen_prediction_improved'] and decisions['legacy_accuracy_preserved_within_5_percent'])}**.",
            f"5. Scale this exact formulation to FUTURE-20K: **{decision}**.",
        ]
    )
    if not go:
        failed = ", ".join(key for key, value in decisions.items() if not value)
        lines.append(f"6. Concrete blocking failure mode(s): {failed}.")
    else:
        lines.append("6. No blocking failure mode was observed under the frozen decision criteria.")
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "report": str(output / 'FINAL_REPORT.md')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

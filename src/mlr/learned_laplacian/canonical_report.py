from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def finalize_canonical_report(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    config = _read_json(run / "config.json")
    training = _read_json(run / "metrics.json")
    history = _read_json_list(run / "training_history.json")
    evaluation = _read_json(run / "evaluation_summary.json")
    _write_csv(run / "train_history.csv", history)
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in run.glob("checkpoint_*.pt"):
        shutil.copyfile(checkpoint, checkpoint_dir / checkpoint.name)
    expanded_source = Path(evaluation["expanded_manifest"])
    (run / "expanded_inference_manifest.json").write_text(
        expanded_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest_dir = run / "manifests_used"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run / "dataset_manifest.json", manifest_dir / "gt_query_manifest.json")
    shutil.copyfile(
        run / "expanded_inference_manifest.json",
        manifest_dir / "expanded_inference_manifest.json",
    )
    checkpoints = evaluation["checkpoint_metrics"]
    expanded = evaluation["expanded_validation"]
    confidence = evaluation["confidence_calibration"]
    original_ablation = [
        row for row in evaluation["image_ablation"] if row["condition"] == "original_rgb"
    ]
    zero_ablation = [
        row for row in evaluation["image_ablation"] if row["condition"] == "zero_rgb"
    ]
    final_correct = original_ablation[-1]
    final_zero = zero_ablation[-1]
    main = _variant(expanded, "main_confidence")
    hard = _variant(expanded, "hard_visibility_only")
    zero = _variant(expanded, "zero_rgb")
    confidence_correlation = float(
        confidence[0]["global_confidence_negative_error_correlation"]
    )
    confidence_monotonic = all(
        float(confidence[index]["normalized_laplacian_error"])
        >= float(confidence[index + 1]["normalized_laplacian_error"])
        for index in range(len(confidence) - 1)
    )
    rgb_better = (
        float(final_correct["normalized_laplacian_mse"])
        < float(final_zero["normalized_laplacian_mse"])
    )
    confidence_better = (
        main.get("refined_chamfer") is not None
        and hard.get("refined_chamfer") is not None
        and float(main["refined_chamfer"]) < float(hard["refined_chamfer"])
    )
    high_curvature_improved = (
        float(checkpoints[-1]["validation_high_10_percent_cosine"])
        > float(checkpoints[0]["validation_high_10_percent_cosine"])
    )
    image_dependence_increased = (
        float(checkpoints[-1]["correct_minus_zero_rgb_cosine_gap"])
        > float(checkpoints[0]["correct_minus_zero_rgb_cosine_gap"])
    )
    improved_meshes = int(main.get("better_than_initial", 0))
    summary = {
        "method_contract": {
            "main_target": "absolute GT h2-normalized uniform Laplacian",
            "target_equation": "delta_gt_hat=(L_gt@V_gt)/(h_gt^2+1e-12)",
            "h_definition": "arithmetic mean of unique undirected one-ring edge lengths",
            "inference_equation": "delta_pred_raw=delta_hat_prediction*(h_current^2+1e-12)",
            "recovery_weight": "renderer_visible_any*confidence_prediction",
            "normalized_to_raw_conversions": 1,
        },
        "dataset": {
            "name": "Sofa50",
            "split_counts": {"train": 40, "validation": 5, "test": 5},
            "gt_query_manifest": evaluation["gt_manifest"],
            "expanded_manifest": evaluation["expanded_manifest"],
            "thingi10k_used": False,
        },
        "training": training,
        "checkpoint_metrics": checkpoints,
        "image_ablation": evaluation["image_ablation"],
        "confidence_calibration": confidence,
        "expanded_validation": expanded,
        "conclusions": {
            "correct_rgb_outperforms_zero_rgb": rgb_better,
            "image_dependence_increased": image_dependence_increased,
            "high_10_percent_cosine_improved": high_curvature_improved,
            "confidence_negative_error_correlation": confidence_correlation,
            "confidence_error_monotonic_by_quantile": confidence_monotonic,
            "learned_confidence_improves_over_hard_visibility": confidence_better,
            "validation_meshes_improved": improved_meshes,
            "validation_mesh_count": int(main.get("mesh_count", 5)),
            "main_introduced_flips": main.get("introduced_flips"),
            "main_new_degeneracies": main.get("new_degeneracies"),
            "direct_and_residual_comparison_status": (
                "legacy controlled diagnostics only; a comparable real-expanded target "
                "does not exist and was not fabricated"
            ),
        },
        "legacy_baselines": evaluation["legacy_baselines"],
        "oracle_confidence": evaluation["oracle_confidence"],
        "visualization_failures": evaluation["visualization_failures"],
        "config": config,
    }
    (run / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run / "REPORT.md").write_text(
        _report(
            summary,
            checkpoints,
            expanded,
            confidence,
            final_correct,
            final_zero,
            main,
            hard,
            zero,
        ),
        encoding="utf-8",
    )
    return summary


def _report(
    summary: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    expanded: Sequence[Mapping[str, Any]],
    confidence: Sequence[Mapping[str, Any]],
    final_correct: Mapping[str, Any],
    final_zero: Mapping[str, Any],
    main: Mapping[str, Any],
    hard: Mapping[str, Any],
    zero: Mapping[str, Any],
) -> str:
    conclusions = summary["conclusions"]
    final_checkpoint = checkpoints[-1]
    lines = [
        "# Sofa50 50-mesh / 2000-epoch absolute h² learned-Laplacian report",
        "",
        "## Method and data contract",
        "",
        "The canonical model predicts the **absolute GT h²-normalized uniform "
        "Laplacian**, never a displacement or Laplacian residual. Training uses "
        "`delta_gt_hat=(L_gt@V_gt)/(h_gt²+1e-12)`, where `h_gt` is the arithmetic "
        "mean of unique undirected one-ring GT edge lengths. Inference recomputes "
        "`h_current` on each current expanded mesh and converts exactly once with "
        "`delta_pred_raw=delta_hat_prediction*(h_current²+1e-12)`. Recovery uses "
        "`renderer_visible_any*confidence_prediction` and an anchor to `X0`.",
        "",
        f"Sofa50 split: 40 train / 5 validation / 5 test. GT-query manifest: "
        f"`{summary['dataset']['gt_query_manifest']}`. Expanded manifest: "
        f"`{summary['dataset']['expanded_manifest']}`. Thingi10K was not used.",
        "",
        "## Training and checkpoint prediction",
        "",
        "| Epoch | Train loss | Val loss | Val cosine | High-10% cosine | Pred/GT norm | Correct-zero RGB loss gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoints:
        lines.append(
            f"| {row['epoch']} | {_fmt(row.get('train_loss'))} | {_fmt(row.get('validation_loss'))} | "
            f"{_fmt(row.get('validation_global_cosine'))} | "
            f"{_fmt(row.get('validation_high_10_percent_cosine'))} | "
            f"{_fmt(row.get('prediction_to_gt_norm_ratio'))} | "
            f"{_fmt(row.get('correct_minus_zero_rgb_loss_gap'))} |"
        )
    lines.extend(
        [
            "",
            "At epoch 2000, correct RGB vs zero RGB normalized MSE is "
            f"`{_fmt(final_correct['normalized_laplacian_mse'])}` vs "
            f"`{_fmt(final_zero['normalized_laplacian_mse'])}`; global cosine is "
            f"`{_fmt(final_correct['global_cosine'])}` vs "
            f"`{_fmt(final_zero['global_cosine'])}`.",
            "",
            "Final correct-RGB endpoint errors (mean vector L2) are "
            f"`{_fmt(final_checkpoint['validation_normalized_laplacian_vector_endpoint_error'])}` "
            "in normalized space and "
            f"`{_fmt(final_checkpoint['validation_raw_laplacian_vector_endpoint_error'])}` "
            "after exact h² denormalization. The normalized errors are "
            f"`{_fmt(final_checkpoint['validation_high_10_percent_normalized_laplacian_error'])}` "
            "on the top 10% GT-magnitude vertices, "
            f"`{_fmt(final_checkpoint['validation_high_1_percent_normalized_laplacian_error'])}` "
            "on the top 1%, and "
            f"`{_fmt(final_checkpoint['validation_smooth_bottom_90_percent_normalized_laplacian_error'])}` "
            "on the remaining smooth 90%. Per-view-count errors are retained in "
            "`checkpoint_metrics.csv`.",
            "",
            "## Expanded validation",
            "",
            "| Variant | Initial Chamfer | Refined Chamfer | P2S | Normal | Flips | Better than initial |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "main_confidence": "Main + confidence",
        "hard_visibility_only": "Hard visibility only",
        "zero_rgb": "Zero RGB",
        "direct_displacement_baseline": "Direct displacement baseline",
        "normalized_laplacian_residual_baseline": "Normalized Lap residual baseline",
    }
    for row in expanded:
        if row.get("refined_chamfer") is None:
            lines.append(
                f"| {labels[row['variant']]} | N/A | N/A | N/A | N/A | N/A | N/A — {row['note']} |"
            )
        else:
            lines.append(
                f"| {labels[row['variant']]} | {_fmt(row['initial_chamfer'])} | "
                f"{_fmt(row['refined_chamfer'])} | {_fmt(row['refined_point_to_surface'])} | "
                f"{_fmt(row['refined_normal_consistency'])} | {row['introduced_flips']} | "
                f"{row['better_than_initial']}/{row['mesh_count']} |"
            )
    lines.extend(
        [
            "",
            "## Confidence calibration",
            "",
            "| Confidence bin | Mean confidence | Normalized Lap error | Raw Lap error | Vertex count |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in confidence:
        lines.append(
            f"| {row['confidence_bin']} | {_fmt(row['mean_confidence'])} | "
            f"{_fmt(row['normalized_laplacian_error'])} | "
            f"{_fmt(row['raw_laplacian_error'])} | {row['vertex_count']} |"
        )
    lines.extend(
        [
            "",
            f"Correlation between confidence and negative normalized error: "
            f"`{_fmt(conclusions['confidence_negative_error_correlation'])}`.",
            "",
            "Oracle confidence on real expanded meshes is intentionally unavailable: "
            f"{summary['oracle_confidence']['reason']}",
            "",
            "## Required questions",
            "",
            "1. **Final formulation:** yes—absolute h²-normalized GT Laplacian prediction.",
            "2. **GT target:** `delta_gt_raw=L_gt@V_gt`; `delta_gt_hat=delta_gt_raw/(h_gt²+1e-12)`.",
            "3. **Scale:** arithmetic mean of unique undirected one-ring edge lengths; epsilon `1e-12`.",
            "4. **Inference scale:** yes, `h_current` is recomputed on every current expanded graph.",
            "5. **Conversion count:** exactly once, enforced by the canonical helper and regression tests.",
            f"6. **Correct RGB vs ablations:** correct RGB beats zero RGB: **{_yes(conclusions['correct_rgb_outperforms_zero_rgb'])}**. Shuffled and cross-object values are in `image_ablation.csv`.",
            f"7. **Image dependence over training:** increased from the first saved checkpoint: **{_yes(conclusions['image_dependence_increased'])}**.",
            f"8. **High-curvature learning:** top-10% cosine improved: **{_yes(conclusions['high_10_percent_cosine_improved'])}**.",
            f"9. **Confidence meaning:** inverse-error correlation is `{_fmt(conclusions['confidence_negative_error_correlation'])}`; quantile error is monotonic: **{_yes(conclusions['confidence_error_monotonic_by_quantile'])}**.",
            f"10. **Confidence recovery benefit:** main Chamfer `{_fmt(main.get('refined_chamfer'))}` vs hard visibility `{_fmt(hard.get('refined_chamfer'))}`; improves: **{_yes(conclusions['learned_confidence_improves_over_hard_visibility'])}**.",
            f"11. **Meshes improved:** `{conclusions['validation_meshes_improved']}/{conclusions['validation_mesh_count']}`.",
            f"12. **Spikes/flips:** main introduced `{main.get('introduced_flips')}` flips and `{main.get('new_degeneracies')}` new degeneracies; legacy controlled results are not silently mixed with this real-expanded table.",
            "13. **Absolute vs residual:** no valid like-for-like real-expanded conclusion. The existing residual branch is a 500-step controlled same-topology diagnostic; expanding it would violate the no-fabricated-target constraint.",
            "14. **Absolute vs displacement:** likewise not comparable on real expanded queries without a vertexwise target. Legacy values remain linked in `summary.json`, while differential metrics are reported for the main method.",
            f"15. **Second-order RGB hypothesis:** evidence is **{'supportive' if conclusions['correct_rgb_outperforms_zero_rgb'] else 'not supportive'}** at 2000 epochs; this statement is limited by the expanded recovery result.",
            f"16. **Dominant limitation:** **{_dominant_limitation(conclusions)}**.",
            "17. **Continue beyond 2000:** **no**. The requested controlled budget is complete; further training should require a new preregistered experiment.",
            f"18. **Smallest next experiment:** {_next_experiment(conclusions)}",
            "",
            "Fixed-camera panels, recovered OBJ files, confidence heatmaps, predicted/GT "
            "normalized-Laplacian magnitude heatmaps, and per-vertex NPZ diagnostics are "
            "preserved in this run directory. Failed renders, if any, are listed in "
            "`summary.json` rather than discarded.",
        ]
    )
    return "\n".join(lines) + "\n"


def _dominant_limitation(conclusions: Mapping[str, Any]) -> str:
    if not conclusions["correct_rgb_outperforms_zero_rgb"]:
        return "image correspondence / prediction accuracy"
    if not conclusions["confidence_error_monotonic_by_quantile"]:
        return "confidence calibration"
    if conclusions["validation_meshes_improved"] == 0:
        return "cross-graph transfer / recovery"
    return "prediction accuracy"


def _next_experiment(conclusions: Mapping[str, Any]) -> str:
    if not conclusions["confidence_error_monotonic_by_quantile"]:
        return "freeze this checkpoint and calibrate only the confidence head on the five validation meshes."
    if conclusions["validation_meshes_improved"] == 0:
        return "on one validation mesh, replace learned predictions with a correspondence-free analytic current-graph identity target to isolate solver amplification."
    return "repeat only the fixed checkpoint expanded recovery on the five held-out test meshes."


def _variant(rows: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return next(row for row in rows if row["variant"] == name)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6g}"


def _yes(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected list: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

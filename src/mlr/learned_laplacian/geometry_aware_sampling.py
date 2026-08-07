from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .multi_dataset import PreparedMeshDataset


ARM_LAYOUT = {
    "G0_uniform": "canonical_0001",
    "G1_mild_high_lap": "importance_0001",
    "G2_strong_high_lap": "strong_importance_0001",
    "G3_smooth_biased": "smooth_importance_0001",
}
GROUP_ORDER = (
    "all",
    "lowest_10",
    "smooth_bottom_90",
    "high_top_10",
    "high_top_1_to_10",
    "high_top_1",
)


def analyze_geometry_aware_sampling(
    baseline_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    *,
    visualization_mesh_count: int = 2,
) -> dict[str, Any]:
    baseline_root = Path(baseline_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    arm_dirs = {
        "G0_uniform": baseline_root / "arms" / ARM_LAYOUT["G0_uniform"],
        "G1_mild_high_lap": baseline_root / "arms" / ARM_LAYOUT["G1_mild_high_lap"],
        "G2_strong_high_lap": output_root / "arms" / ARM_LAYOUT["G2_strong_high_lap"],
        "G3_smooth_biased": output_root / "arms" / ARM_LAYOUT["G3_smooth_biased"],
    }
    _require_arm_artifacts(arm_dirs)

    metrics, contribution_rows = _exact_metrics_and_contributions(arm_dirs)
    exposure_rows = _exposure_rows(arm_dirs)
    comparison_rows = _comparison_rows(metrics)
    _write_csv(analysis_dir / "main_comparison.csv", comparison_rows)
    _write_csv(analysis_dir / "training_exposure.csv", exposure_rows)
    _write_csv(analysis_dir / "error_contributions.csv", contribution_rows)

    dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    visualization_ids = _write_heatmaps(
        dataset,
        arm_dirs,
        analysis_dir / "heatmaps",
        mesh_count=visualization_mesh_count,
    )
    deltas = _improvement_summary(metrics)
    verdict = _verdict(deltas)
    contract = _contract_audit(arm_dirs)
    recovery = {
        "performed": False,
        "reason": (
            "Prediction-side criterion was not met; canonical recovery was not run."
            if verdict != "Supported"
            else "Prediction-side criterion was met; recovery remains an optional separate sanity check."
        ),
    }
    summary = {
        "experiment": "Sofa50 geometry-aware vertex sampling, 1000 optimizer steps",
        "hypothesis": (
            "Uniform vertex sampling under-allocates optimization exposure to "
            "geometrically informative high-Laplacian regions."
        ),
        "verdict": verdict,
        "contract_audit": contract,
        "sampling_design": {
            "G0_uniform": "full-mesh uniform exposure",
            "G1_mild_high_lap": "50% uniform + 25% top10 + 25% top1-10",
            "G2_strong_high_lap": "25% uniform + 50% top10 + 25% top1",
            "G3_smooth_biased": "50% uniform + 50% bottom90",
        },
        "arms": {name: str(path) for name, path in arm_dirs.items()},
        "exact_query_metrics": metrics,
        "relative_changes_vs_G0": deltas,
        "training_exposure": exposure_rows,
        "error_contributions": contribution_rows,
        "visualization_mesh_ids": visualization_ids,
        "recovery_sanity_check": recovery,
        "artifacts": {
            "main_comparison_csv": str(analysis_dir / "main_comparison.csv"),
            "training_exposure_csv": str(analysis_dir / "training_exposure.csv"),
            "error_contributions_csv": str(analysis_dir / "error_contributions.csv"),
            "heatmap_directory": str(analysis_dir / "heatmaps"),
        },
    }
    _write_json(analysis_dir / "summary.json", summary)
    (analysis_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _require_arm_artifacts(arm_dirs: Mapping[str, Path]) -> None:
    required = (
        "config.json",
        "training_history.json",
        "screening_summary.json",
        "training_vertex_exposure.json",
    )
    for name, arm_dir in arm_dirs.items():
        for filename in required:
            if not (arm_dir / filename).is_file():
                raise FileNotFoundError(f"Missing {name} artifact: {arm_dir / filename}")
        if len(list((arm_dir / "fixed_query_predictions").glob("*__exact.npz"))) != 5:
            raise ValueError(f"{name} must contain five exact-query prediction files.")


def _exact_metrics_and_contributions(
    arm_dirs: Mapping[str, Path],
) -> tuple[dict[str, dict[str, dict[str, float | int]]], list[dict[str, Any]]]:
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    contribution_rows: list[dict[str, Any]] = []
    for arm, arm_dir in arm_dirs.items():
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        group_masks: dict[str, list[np.ndarray]] = {name: [] for name in GROUP_ORDER}
        for path in sorted((arm_dir / "fixed_query_predictions").glob("*__exact.npz")):
            payload = np.load(path)
            prediction = payload["prediction"].astype(np.float64)
            target = payload["target"].astype(np.float64)
            predictions.append(prediction)
            targets.append(target)
            masks = _magnitude_masks(np.linalg.norm(target, axis=1))
            for name in GROUP_ORDER:
                group_masks[name].append(masks[name])
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        masks = {name: np.concatenate(parts) for name, parts in group_masks.items()}
        residual_norm = np.linalg.norm(prediction - target, axis=1)
        squared_error = np.square(residual_norm)
        metrics[arm] = {}
        for group in GROUP_ORDER:
            mask = masks[group]
            pred = prediction[mask]
            gt = target[mask]
            endpoint = residual_norm[mask]
            pred_mag = np.linalg.norm(pred, axis=1)
            gt_mag = np.linalg.norm(gt, axis=1)
            per_vertex_cos = F.cosine_similarity(
                torch.from_numpy(pred), torch.from_numpy(gt), dim=-1, eps=1e-8
            ).numpy()
            metrics[arm][group] = {
                "vertex_count": int(mask.sum()),
                "mean_normalized_endpoint_error": float(endpoint.mean()),
                "median_normalized_endpoint_error": float(np.median(endpoint)),
                "cosine_similarity": float(per_vertex_cos.mean()),
                "global_cosine": float(
                    np.dot(pred.reshape(-1), gt.reshape(-1))
                    / max(np.linalg.norm(pred) * np.linalg.norm(gt), 1e-12)
                ),
                "prediction_to_gt_magnitude_ratio": float(
                    pred_mag.mean() / max(gt_mag.mean(), 1e-12)
                ),
                "prediction_to_gt_global_norm_ratio": float(
                    np.linalg.norm(pred) / max(np.linalg.norm(gt), 1e-12)
                ),
                "mean_gt_magnitude": float(gt_mag.mean()),
                "mean_prediction_magnitude": float(pred_mag.mean()),
                "mean_residual_magnitude": float(endpoint.mean()),
                "group_relative_error": float(endpoint.mean() / max(gt_mag.mean(), 1e-12)),
            }
            contribution_rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "vertex_fraction": float(mask.mean()),
                    "squared_error_total_fraction": float(
                        squared_error[mask].sum() / max(squared_error.sum(), 1e-12)
                    ),
                    "absolute_endpoint_error_total_fraction": float(
                        residual_norm[mask].sum() / max(residual_norm.sum(), 1e-12)
                    ),
                }
            )
    return metrics, contribution_rows


def _magnitude_masks(magnitude: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(magnitude, kind="stable")
    count = len(order)
    low10 = max(1, round(0.10 * count))
    top10 = max(1, round(0.10 * count))
    top1 = max(1, round(0.01 * count))

    def make(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(count, dtype=bool)
        mask[indices] = True
        return mask

    top10_indices = order[count - top10 :]
    top1_indices = order[count - top1 :]
    return {
        "all": np.ones(count, dtype=bool),
        "lowest_10": make(order[:low10]),
        "smooth_bottom_90": make(order[: count - top10]),
        "high_top_10": make(top10_indices),
        "high_top_1_to_10": make(np.setdiff1d(top10_indices, top1_indices)),
        "high_top_1": make(top1_indices),
    }


def _exposure_rows(arm_dirs: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] = {}
    payloads = {
        arm: _read_json(path / "training_vertex_exposure.json")
        for arm, path in arm_dirs.items()
    }
    for group in GROUP_ORDER:
        if group == "high_top_1_to_10" and group not in payloads["G0_uniform"]["group_draw_counts"]:
            count = (
                int(payloads["G0_uniform"]["group_draw_counts"]["high_top_10"])
                - int(payloads["G0_uniform"]["group_draw_counts"]["high_top_1"])
            )
            baseline[group] = count / int(payloads["G0_uniform"]["selected_rows"])
        else:
            baseline[group] = float(payloads["G0_uniform"]["group_draw_fractions"][group])
    for arm, payload in payloads.items():
        total = int(payload["selected_rows"])
        for group in GROUP_ORDER:
            if group == "high_top_1_to_10" and group not in payload["group_draw_counts"]:
                count = (
                    int(payload["group_draw_counts"]["high_top_10"])
                    - int(payload["group_draw_counts"]["high_top_1"])
                )
            else:
                count = int(payload["group_draw_counts"][group])
            fraction = count / max(total, 1)
            rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "sampled_vertex_count": count,
                    "exposure_fraction": fraction,
                    "exposure_multiplier_vs_G0": fraction / max(baseline[group], 1e-12),
                }
            )
    return rows


def _comparison_rows(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]]
) -> list[dict[str, Any]]:
    fields = (
        "mean_normalized_endpoint_error",
        "median_normalized_endpoint_error",
        "cosine_similarity",
        "global_cosine",
        "prediction_to_gt_magnitude_ratio",
        "prediction_to_gt_global_norm_ratio",
        "mean_gt_magnitude",
        "mean_prediction_magnitude",
        "mean_residual_magnitude",
        "group_relative_error",
    )
    rows: list[dict[str, Any]] = []
    for group in ("high_top_10", "high_top_1", "smooth_bottom_90", "lowest_10", "all"):
        for field in fields:
            rows.append(
                {
                    "metric": field,
                    "group": group,
                    **{arm: values[group][field] for arm, values in metrics.items()},
                }
            )
    return rows


def _improvement_summary(
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]]
) -> dict[str, dict[str, float]]:
    baseline = metrics["G0_uniform"]
    result: dict[str, dict[str, float]] = {}
    for arm in ("G1_mild_high_lap", "G2_strong_high_lap", "G3_smooth_biased"):
        values = metrics[arm]
        result[arm] = {
            "top10_improvement": _improvement(
                values["high_top_10"]["mean_normalized_endpoint_error"],
                baseline["high_top_10"]["mean_normalized_endpoint_error"],
            ),
            "top1_improvement": _improvement(
                values["high_top_1"]["mean_normalized_endpoint_error"],
                baseline["high_top_1"]["mean_normalized_endpoint_error"],
            ),
            "smooth_degradation": -_improvement(
                values["smooth_bottom_90"]["mean_normalized_endpoint_error"],
                baseline["smooth_bottom_90"]["mean_normalized_endpoint_error"],
            ),
            "overall_improvement": _improvement(
                values["all"]["mean_normalized_endpoint_error"],
                baseline["all"]["mean_normalized_endpoint_error"],
            ),
        }
    return result


def _improvement(value: float | int, baseline: float | int) -> float:
    return (float(baseline) - float(value)) / max(abs(float(baseline)), 1e-12)


def _verdict(changes: Mapping[str, Mapping[str, float]]) -> str:
    high_arms = (changes["G1_mild_high_lap"], changes["G2_strong_high_lap"])
    supported_arm = any(
        values["top10_improvement"] >= 0.05
        and values["top1_improvement"] >= 0.05
        and values["smooth_degradation"] < min(
            values["top10_improvement"], values["top1_improvement"]
        )
        for values in high_arms
    )
    control = changes["G3_smooth_biased"]
    if supported_arm and not (
        control["top10_improvement"] >= 0.05 and control["top1_improvement"] >= 0.05
    ):
        return "Supported"
    if all(
        values["top10_improvement"] <= 0.0 and values["top1_improvement"] <= 0.0
        for values in high_arms
    ):
        return "Not supported"
    return "Inconclusive"


def _contract_audit(arm_dirs: Mapping[str, Path]) -> dict[str, Any]:
    configs = {arm: _read_json(path / "config.json") for arm, path in arm_dirs.items()}
    reference = configs["G0_uniform"]
    invariant_fields = (
        "seed",
        "input_mode",
        "target_mode",
        "target_semantics",
        "target_scaling",
        "query_training",
        "renderer_visibility",
        "image_encoder",
        "model",
        "confidence",
        "data_loading",
        "recovery",
    )
    summaries = {
        arm: _read_json(path / "screening_summary.json") for arm, path in arm_dirs.items()
    }
    return {
        "invariant_config_fields_equal": all(
            config.get(field) == reference.get(field)
            for config in configs.values()
            for field in invariant_fields
        ),
        "invariant_config_fields": list(invariant_fields),
        "same_seed": len({int(config["seed"]) for config in configs.values()}) == 1,
        "all_fresh_start": all(
            config["screening"]["resume_checkpoint"] is None for config in configs.values()
        ),
        "optimizer_steps": {
            arm: int(summary["optimizer_steps"]) for arm, summary in summaries.items()
        },
        "history_records": {
            arm: len(_read_json_list(path / "training_history.json"))
            for arm, path in arm_dirs.items()
        },
        "exact_prediction_files": {
            arm: len(list((path / "fixed_query_predictions").glob("*__exact.npz")))
            for arm, path in arm_dirs.items()
        },
    }


def _write_heatmaps(
    dataset: PreparedMeshDataset,
    arm_dirs: Mapping[str, Path],
    output_dir: Path,
    *,
    mesh_count: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids: list[str] = []
    for index in range(min(mesh_count, len(dataset))):
        sample = dataset.load_static(index)
        sample_id = str(sample["sample_id"])
        selected_ids.append(sample_id)
        vertices = sample["vertices"].float().cpu().numpy()
        faces = sample["faces"].long().cpu().numpy()
        arm_payloads = {
            arm: np.load(path / "fixed_query_predictions" / f"{sample_id}__exact.npz")
            for arm, path in arm_dirs.items()
        }
        target = arm_payloads["G0_uniform"]["target"]
        target_mag = np.linalg.norm(target, axis=1)
        errors = {
            arm: np.linalg.norm(payload["prediction"] - payload["target"], axis=1)
            for arm, payload in arm_payloads.items()
        }
        common_error_max = float(np.quantile(np.concatenate(list(errors.values())), 0.99))
        plots = {
            "gt_laplacian_magnitude": (target_mag, "viridis", 0.0, float(np.quantile(target_mag, 0.99))),
            **{
                f"{arm}_prediction_error": (values, "magma", 0.0, common_error_max)
                for arm, values in errors.items()
            },
            "G1_minus_G0_error": (
                errors["G1_mild_high_lap"] - errors["G0_uniform"],
                "coolwarm",
                None,
                None,
            ),
            "G2_minus_G0_error": (
                errors["G2_strong_high_lap"] - errors["G0_uniform"],
                "coolwarm",
                None,
                None,
            ),
        }
        for name, (values, cmap, vmin, vmax) in plots.items():
            output_path = output_dir / f"{sample_id}__{name}.png"
            if output_path.is_file():
                continue
            if vmin is None:
                bound = max(float(np.quantile(np.abs(values), 0.99)), 1e-8)
                vmin, vmax = -bound, bound
            figure = plt.figure(figsize=(8, 7), dpi=180)
            axis = figure.add_subplot(111, projection="3d")
            collection = Poly3DCollection(
                vertices[faces], linewidths=0.0, edgecolors="none", rasterized=True
            )
            collection.set_array(values[faces].mean(axis=1))
            collection.set_cmap(cmap)
            collection.set_clim(vmin, vmax)
            axis.add_collection3d(collection)
            minimum = vertices.min(axis=0)
            maximum = vertices.max(axis=0)
            center = 0.5 * (minimum + maximum)
            radius = 0.5 * float((maximum - minimum).max())
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=25, azim=-55)
            axis.set_axis_off()
            axis.set_title(f"{sample_id}\n{name}")
            figure.colorbar(collection, ax=axis, shrink=0.65, pad=0.02)
            figure.tight_layout()
            figure.savefig(output_path, bbox_inches="tight")
            plt.close(figure)
    return selected_ids


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(summary: Mapping[str, Any]) -> str:
    metrics = summary["exact_query_metrics"]
    arms = tuple(ARM_LAYOUT)
    lines = [
        "# Sofa50 geometry-aware vertex sampling",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "`||delta_hat_GT||` is used only as a differential-signal / geometry-information proxy, not as strict curvature.",
        "",
        "## Main exact-query comparison",
        "",
        "| metric | " + " | ".join(arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    rows = (
        ("top10 mean endpoint", "high_top_10", "mean_normalized_endpoint_error"),
        ("top1 mean endpoint", "high_top_1", "mean_normalized_endpoint_error"),
        ("smooth90 mean endpoint", "smooth_bottom_90", "mean_normalized_endpoint_error"),
        ("overall mean endpoint", "all", "mean_normalized_endpoint_error"),
        ("overall global cosine", "all", "global_cosine"),
        ("overall pred/GT global norm", "all", "prediction_to_gt_global_norm_ratio"),
        ("lowest10 predicted magnitude", "lowest_10", "mean_prediction_magnitude"),
    )
    for label, group, field in rows:
        lines.append(
            f"| {label} | "
            + " | ".join(f"{float(metrics[arm][group][field]):.6f}" for arm in arms)
            + " |"
        )
    lines.extend(
        [
            "",
        "## Relative changes versus G0",
        "",
        "```json",
        json.dumps(summary["relative_changes_vs_G0"], indent=2),
        "```",
        "",
        "## Recovery",
        "",
        summary["recovery_sanity_check"]["reason"],
        "",
        ]
    )
    return "\n".join(lines)

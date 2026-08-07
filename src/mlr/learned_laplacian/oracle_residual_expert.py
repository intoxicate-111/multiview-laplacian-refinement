from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .geometry_aware_sampling import _exact_metrics_and_contributions
from .multi_trainer import _build_model
from .trainer import _seed_everything


ARM_LAYOUT = {"E0_baseline": "oracle_expert_e0", "E1_residual_expert": "oracle_expert_e1"}


def analyze_oracle_residual_expert(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    arm_dirs = {name: output_root / "arms" / arm for name, arm in ARM_LAYOUT.items()}
    for name, arm_dir in arm_dirs.items():
        if not (arm_dir / "screening_summary.json").is_file():
            raise FileNotFoundError(f"Missing {name} artifacts: {arm_dir}")
        if len(list((arm_dir / "fixed_query_predictions").glob("*__exact.npz"))) != 5:
            raise ValueError(f"{name} must contain five exact-query prediction files.")

    metrics, _ = _exact_metrics_and_contributions(arm_dirs)
    e0 = metrics["E0_baseline"]
    e1 = metrics["E1_residual_expert"]
    comparison = {
        "overall_endpoint": _pair(e0, e1, "all", "mean_normalized_endpoint_error"),
        "global_cosine": _pair(e0, e1, "all", "global_cosine"),
        "prediction_to_gt_norm": _pair(
            e0, e1, "all", "prediction_to_gt_global_norm_ratio"
        ),
        "top10_endpoint": _pair(
            e0, e1, "high_top_10", "mean_normalized_endpoint_error"
        ),
        "top1_endpoint": _pair(
            e0, e1, "high_top_1", "mean_normalized_endpoint_error"
        ),
        "smooth90_endpoint": _pair(
            e0, e1, "smooth_bottom_90", "mean_normalized_endpoint_error"
        ),
    }
    top10_improvement = _endpoint_improvement(comparison["top10_endpoint"])
    top1_improvement = _endpoint_improvement(comparison["top1_endpoint"])
    overall_degradation = -_endpoint_improvement(comparison["overall_endpoint"])
    smooth_degradation = -_endpoint_improvement(comparison["smooth90_endpoint"])
    verdict = (
        "Supported"
        if top10_improvement >= 0.05
        and overall_degradation <= 0.02
        and smooth_degradation <= 0.02
        else "Not supported"
        if top10_improvement <= 0.0
        else "Inconclusive"
    )
    config_e0 = _read_json(arm_dirs["E0_baseline"] / "config.json")
    config_e1 = _read_json(arm_dirs["E1_residual_expert"] / "config.json")
    initialization = _initialization_audit(config_e0, config_e1)
    per_mesh = _per_mesh_changes(arm_dirs)
    summaries = {
        name: _read_json(path / "screening_summary.json")
        for name, path in arm_dirs.items()
    }
    contract = {
        "same_seed": int(config_e0["seed"]) == int(config_e1["seed"]),
        "same_canonical_contract_except_expert": _same_contract_except_expert(
            config_e0, config_e1
        ),
        "uniform_full_vertex_training": all(
            config["training"]["vertex_sampling"]["mode"] == "full"
            for config in (config_e0, config_e1)
        ),
        "exact_validation": all(
            config["query_training"]["apply_to_validation"] is False
            for config in (config_e0, config_e1)
        ),
        "fresh_start": all(
            config["screening"]["resume_checkpoint"] is None
            for config in (config_e0, config_e1)
        ),
        "optimizer_steps": {
            name: int(summary["optimizer_steps"]) for name, summary in summaries.items()
        },
        **initialization,
    }
    summary = {
        "experiment": "Sofa50 oracle top10 residual expert, 1000 optimizer steps",
        "verdict": verdict,
        "comparison": comparison,
        "relative_changes": {
            "top10_improvement": top10_improvement,
            "top1_improvement": top1_improvement,
            "smooth90_degradation": smooth_degradation,
            "overall_degradation": overall_degradation,
            "global_cosine_change": (
                comparison["global_cosine"]["E1"]
                - comparison["global_cosine"]["E0"]
            ),
        },
        "contract_audit": contract,
        "per_mesh_changes": per_mesh,
        "top10_improved_mesh_count": sum(
            row["top10_improvement"] > 0 for row in per_mesh
        ),
        "expert": {
            "gate": "per-mesh clean GT ||delta_hat|| top 10% oracle mask",
            "branch": "shared graph feature -> Linear(64,32) -> ReLU -> Linear(32,3)",
            "bottom90_residual": "exactly zero by multiplicative boolean mask",
            "learned_gate": False,
        },
        "full_exact_query_metrics": metrics,
        "next_step": (
            "A learned gate is worth a separate follow-up."
            if verdict == "Supported"
            else "Stop this direction; do not build a full MoE or learned gate."
        ),
    }
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _pair(
    e0: Mapping[str, Mapping[str, float | int]],
    e1: Mapping[str, Mapping[str, float | int]],
    group: str,
    field: str,
) -> dict[str, float]:
    return {"E0": float(e0[group][field]), "E1": float(e1[group][field])}


def _endpoint_improvement(pair: Mapping[str, float]) -> float:
    return (pair["E0"] - pair["E1"]) / max(abs(pair["E0"]), 1e-12)


def _initialization_audit(
    config_e0: Mapping[str, Any], config_e1: Mapping[str, Any]
) -> dict[str, Any]:
    seed = int(config_e0["seed"])
    _seed_everything(seed)
    model_e0 = _build_model(config_e0, None, False).cpu()
    _seed_everything(seed)
    model_e1 = _build_model(config_e1, None, False).cpu()
    state_e0 = model_e0.state_dict()
    state_e1 = model_e1.state_dict()
    common = sorted(set(state_e0).intersection(state_e1))
    canonical_equal = all(torch.equal(state_e0[name], state_e1[name]) for name in common)
    residual_keys = sorted(name for name in state_e1 if name.startswith("oracle_residual_expert."))
    final_weight = state_e1["oracle_residual_expert.2.weight"]
    final_bias = state_e1["oracle_residual_expert.2.bias"]
    return {
        "canonical_initial_parameter_tensors_equal": canonical_equal,
        "canonical_initial_tensor_count": len(common),
        "expert_parameter_count": sum(
            parameter.numel()
            for name, parameter in model_e1.named_parameters()
            if name.startswith("oracle_residual_expert.")
        ),
        "expert_state_keys": residual_keys,
        "expert_initial_output_layer_zero": bool(
            torch.count_nonzero(final_weight) == 0 and torch.count_nonzero(final_bias) == 0
        ),
    }


def _same_contract_except_expert(
    config_e0: Mapping[str, Any], config_e1: Mapping[str, Any]
) -> bool:
    left = json.loads(json.dumps(config_e0))
    right = json.loads(json.dumps(config_e1))
    left["model"].pop("oracle_residual_expert", None)
    right["model"].pop("oracle_residual_expert", None)
    left["screening"]["arm"] = "paired"
    right["screening"]["arm"] = "paired"
    return left == right


def _per_mesh_changes(arm_dirs: Mapping[str, Path]) -> list[dict[str, Any]]:
    e0_dir = arm_dirs["E0_baseline"] / "fixed_query_predictions"
    e1_dir = arm_dirs["E1_residual_expert"] / "fixed_query_predictions"
    rows: list[dict[str, Any]] = []
    for e0_path in sorted(e0_dir.glob("*__exact.npz")):
        e0 = np.load(e0_path)
        e1 = np.load(e1_dir / e0_path.name)
        target = e0["target"]
        magnitude = np.linalg.norm(target, axis=1)
        top_count = max(1, round(0.10 * len(target)))
        top = np.zeros(len(target), dtype=bool)
        top[np.argsort(magnitude, kind="stable")[-top_count:]] = True
        error_e0 = np.linalg.norm(e0["prediction"] - target, axis=1)
        error_e1 = np.linalg.norm(e1["prediction"] - target, axis=1)
        rows.append(
            {
                "sample_id": e0_path.name.removesuffix("__exact.npz"),
                "top10_improvement": float(
                    (error_e0[top].mean() - error_e1[top].mean())
                    / max(error_e0[top].mean(), 1e-12)
                ),
                "smooth90_degradation": float(
                    (error_e1[~top].mean() - error_e0[~top].mean())
                    / max(error_e0[~top].mean(), 1e-12)
                ),
                "overall_degradation": float(
                    (error_e1.mean() - error_e0.mean()) / max(error_e0.mean(), 1e-12)
                ),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _report(summary: Mapping[str, Any]) -> str:
    comparison = summary["comparison"]
    lines = [
        "# Sofa50 oracle residual expert diagnostic",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "| metric | E0 baseline | E1 residual expert |",
        "|---|---:|---:|",
    ]
    for name, pair in comparison.items():
        lines.append(f"| {name} | {pair['E0']:.6f} | {pair['E1']:.6f} |")
    lines.extend(
        (
            "",
            "```json",
            json.dumps(summary["relative_changes"], indent=2),
            "```",
            "",
            summary["next_step"],
            "",
        )
    )
    return "\n".join(lines)

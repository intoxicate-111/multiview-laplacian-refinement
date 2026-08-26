#!/usr/bin/env python3
from __future__ import annotations

"""Create the standalone frozen report for the completed Uniform hybrid run."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from report_sofa50_uniform_vs_cotangent_hybrid import (
    GROUPS,
    METRIC_PROTOCOL,
    _aggregate,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    return f"{value:.9g}" if isinstance(value, float) else str(value)


def _training_stability(run: Path) -> dict[str, Any]:
    rows = json.loads((run / "training_step_history.json").read_text(encoding="utf-8"))
    diagnostic = [row for row in rows if row.get("pcg_iterations_mean") is not None]
    if not diagnostic:
        return {"logged_intervals": 0}

    def mean_present(field: str) -> float | None:
        values = [float(row[field]) for row in diagnostic if row.get(field) is not None]
        return float(np.mean(values)) if values else None

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
        "delta_pred_gradient_norm_mean": mean_present("delta_pred_gradient_norm"),
        "v_direct_gradient_norm_mean": mean_present("v_direct_gradient_norm"),
        "prediction_head_gradient_norm_mean": mean_present("prediction_head_gradient_norm"),
        "direct_head_gradient_norm_mean": mean_present("direct_head_gradient_norm"),
        "peak_gpu_memory_mb": float(
            max(float(row["peak_gpu_memory_mb"]) for row in diagnostic)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    domains = ("matched_v2", "legacy_v1", "unseen_recipes_v1")
    payloads = {
        domain: _read(args.evaluation_dir / f"{domain}_uniform.json")
        for domain in domains
    }
    all_rows = [row for domain in domains for row in payloads[domain]["rows"]]
    aggregates: list[dict[str, Any]] = []
    for domain in domains:
        splits = sorted({str(row["split"]) for row in payloads[domain]["rows"]})
        for split in splits:
            rows = [row for row in payloads[domain]["rows"] if row["split"] == split]
            aggregates.append(_aggregate(rows, {"domain": domain, "split": split, "arm": "uniform"}))

    recipe_rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_rows = [
            row
            for row in payloads["matched_v2"]["rows"]
            if row["split"] == split
        ]
        for group, recipes in GROUPS.items():
            selected = [row for row in split_rows if row["recipe"] in recipes]
            recipe_rows.append(
                _aggregate(
                    selected,
                    {"domain": "matched_v2", "split": split, "group": group, "arm": "uniform"},
                )
            )

    run = args.run.resolve()
    metrics = _read(run / "metrics.json")
    run_config = _read(run / "run_config.json")
    config = run_config.get("experiment_config", run_config)
    hybrid = config["training"]["hybrid_single_geometry_loss"]
    recovery_aux = config["training"]["recovery_aware_geometry_loss"]
    matched_counts = {
        split: sum(row["split"] == split for row in payloads["matched_v2"]["rows"])
        for split in ("validation", "test")
    }
    checkpoint_hashes = {payload["checkpoint_sha256"] for payload in payloads.values()}
    parameter_counts = {int(payload["parameter_count"]) for payload in payloads.values()}
    checks = {
        "all_evaluations_audited": all(bool(payload["contract_audit"]) for payload in payloads.values()),
        "uniform_operator": all(payload["operator"] == "uniform_random_walk" for payload in payloads.values()),
        "same_frozen_checkpoint": len(checkpoint_hashes) == 1,
        "same_parameter_count": len(parameter_counts) == 1,
        "matched_validation_test_50": matched_counts == {"validation": 50, "test": 50},
        "legacy_test_50": len(payloads["legacy_v1"]["rows"]) == 50,
        "unseen_recipes_test_25": len(payloads["unseen_recipes_v1"]["rows"]) == 25,
        "optimizer_steps_20000": int(metrics["optimizer_steps"]) == 20000,
        "single_final_geometry_loss": bool(hybrid["enabled"]) and not bool(recovery_aux["enabled"]),
        "lambda_3e_minus_2": float(hybrid["lambda"]) == 3e-2,
    }
    summary = {
        "contract_audit": all(checks.values()),
        "strict_execution_contract": False,
        "strict_execution_contract_reason": (
            "training resumed at epoch boundary step 11150 with world size 4 and "
            "accumulation 2 after starting with world size 8 and accumulation 1; "
            "effective global batch remained 8"
        ),
        "contract_checks": checks,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "parameter_count": next(iter(parameter_counts)),
        "lambda": float(hybrid["lambda"]),
        "optimizer_steps": int(metrics["optimizer_steps"]),
        "best_epoch": int(metrics["best_epoch"]),
        "best_selection_loss": float(metrics["best_selection_loss"]),
        "aggregates": aggregates,
        "recipe_breakdown": recipe_rows,
        "training_stability": _training_stability(run),
        "metric_protocol": METRIC_PROTOCOL,
    }
    if not summary["contract_audit"]:
        raise RuntimeError(f"Standalone report contract failed: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "aggregate.csv", aggregates)
    _write_csv(args.output_dir / "recipe_breakdown.csv", recipe_rows)
    _write_csv(args.output_dir / "per_sample.csv", all_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Sofa50 v2 end-to-end Uniform direct–Laplacian hybrid",
        "",
        f"Contract audit: **{str(summary['contract_audit']).lower()}**. Strict execution contract: **false**.",
        "",
        "The run completed 20,000 optimizer steps. It started on 8 Blackwell GPUs and resumed at the epoch boundary at step 11,150 on 4 Blackwell GPUs with accumulation 2; effective global batch remained 8.",
        "",
        "## Frozen formulation",
        "",
        "The model predicts a latent Uniform-Laplacian field `delta_hat` and a direct displacement `DeltaV_hat`, with `V_direct = V_input + DeltaV_hat`.",
        "",
        "Recovery solves `V_H = argmin_V ||L_U V-delta_hat||_2^2 + lambda ||V-V_direct||_2^2`, where `L_U=I-D^-1 A` and `lambda=3e-2`.",
        "",
        "With `A=L_U^T L_U+lambda I` and final geometry gradient `g=dL/dV_H`, implicit differentiation gives `z=A^-1 g`, `dL/d(delta_hat)=L_U z`, and `dL/d(V_direct)=lambda z`.",
        "",
        "Training uses only the final same-index geometry loss; no auxiliary raw-Laplacian or direct-displacement loss is active.",
        "",
        "## Unified-v2 geometry",
        "",
        "| Domain | Split | Samples | Initial CD | Refined CD | Gain | Eta | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened | Vertex RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {domain} | {split} | {samples} | {initial} | {refined} | {gain} | {eta} | {p95} | {fscore} | {normal} | {flips} / {rate} | {deg} | {improved}/{worsened} | {vrms} |".format(
                domain=row["domain"], split=row["split"], samples=row["samples"],
                initial=_fmt(row["initial_chamfer"]), refined=_fmt(row["refined_chamfer"]),
                gain=f"{100 * row['relative_chamfer_gain']:+.2f}%", eta=_fmt(row["eta"]),
                p95=_fmt(row["p2s_p95"]), fscore=_fmt(row["fscore"]), normal=_fmt(row["normal_consistency"]),
                flips=row["introduced_flipped_faces"], rate=f"{100 * row['normalized_flip_rate']:.3f}%",
                deg=row["new_degenerate_faces"], improved=row["improved"], worsened=row["worsened"],
                vrms=_fmt(row["same_index_recovered_vertex_rms"]),
            )
        )
    stability = summary["training_stability"]
    lines.extend(
        [
            "",
            "## Training and numerical stability",
            "",
            f"Best epoch `{summary['best_epoch']}`; validation selection loss `{_fmt(summary['best_selection_loss'])}`; parameter count `{summary['parameter_count']}`.",
            "",
            f"Logged intervals `{stability.get('logged_intervals', 0)}`; PCG iterations mean/max `{_fmt(stability.get('pcg_iterations_mean'))}` / `{_fmt(stability.get('pcg_iterations_max'))}`; failed solves `{stability.get('pcg_failed_solves')}`; NaN/Inf `{stability.get('nan_inf_count')}`; peak GPU memory `{_fmt(stability.get('peak_gpu_memory_mb'))} MiB`.",
            "",
            f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`.",
            "",
            f"Metric protocol: `{METRIC_PROTOCOL}`.",
        ]
    )
    (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"contract_audit": True, "output": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXTERNAL = (
    ("OpenMVS RefineMesh", "openmvs_refinemesh"),
    ("NDS", "nds"),
    ("NeRF2Mesh", "nerf2mesh"),
    ("ExMesh", "exmesh"),
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    learned = _read(args.learned_analysis / "aggregate.json")
    lap_train = _read(args.laplacian_run / "metrics.json")
    disp_train = _read(args.displacement_run / "metrics.json")
    lap_history = _read(args.laplacian_run / "training_history.json")
    disp_history = _read(args.displacement_run / "training_history.json")
    external = {
        key: _read(args.external_output / key / "aggregate.json")
        for _, key in EXTERNAL
    }
    qualitative = _read(args.qualitative_dir / "manifest.json")
    status = {
        "Initial/current mesh": {"status": "completed", "samples": 1000},
        "Learned current-graph raw Laplacian": {"status": "completed", "samples": 1000},
        "Direct displacement": {"status": "completed", "samples": 1000},
    }
    for label, key in EXTERNAL:
        result = external[key]
        status[label] = {
            "status": result["status"],
            "samples": result["total_samples"],
            "completed_samples": result["completed_samples"],
            "failed_samples": result["failed_samples"],
            "failure_reasons": result["failure_reasons"],
        }
    overall_status = (
        "completed"
        if all(result["status"] == "completed" for result in external.values())
        else "completed_with_external_failures"
    )
    payload = {
        "status": overall_status,
        "experiment": "Future2000 GT-ADAPTIVE 2000 objects x 5 variants",
        "test_samples": learned["test_samples"],
        "input_contract": "same current mesh + same 28 RGB views + same cameras; GT only after inference for metrics",
        "training": {
            "laplacian": _training_summary(lap_train, lap_history),
            "direct_displacement": _training_summary(disp_train, disp_history),
        },
        "learned_evaluation": learned,
        "external_evaluation": external,
        "baseline_status": status,
        "qualitative": qualitative,
        "job_ids": args.job_ids,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "FINAL_REPORT.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "baseline_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "FINAL_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    (args.output_dir / "commands.sh").write_text(
        _commands(args), encoding="utf-8"
    )
    return payload


def _training_summary(metrics: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    first = history[0]
    last = history[-1]
    return {
        "optimizer_steps": metrics["optimizer_steps"],
        "completed_epochs": metrics["completed_epochs"],
        "first_epoch_train_loss": first["train_loss"],
        "first_epoch_validation_loss": first["validation_loss"],
        "final_train_loss": metrics["final_train_loss"],
        "final_validation_loss": metrics["final_validation_loss"],
        "train_loss_drop_fraction": _drop(first["train_loss"], last["train_loss"]),
        "validation_loss_drop_fraction": _drop(first["validation_loss"], last["validation_loss"]),
        "best_epoch": metrics["best_epoch"],
        "best_selection_loss": metrics["best_selection_loss"],
        "runtime_seconds": metrics["runtime_seconds"],
        "mean_optimizer_step_seconds": metrics["mean_optimizer_step_seconds"],
        "peak_gpu_memory_mb": metrics["peak_gpu_memory_mb"],
        "distributed_world_size": metrics.get("distributed_world_size", 1),
        "stop_reason": metrics["stop_reason"],
    }


def _drop(first: float, last: float) -> float:
    return (float(first) - float(last)) / float(first) if float(first) else 0.0


def _metric_rows(payload: dict[str, Any]) -> list[list[str]]:
    learned = payload["learned_evaluation"]
    rows = []
    for label, key in (
        ("Current", "initial"),
        ("Learned Laplacian", "laplacian"),
        ("Direct displacement", "displacement"),
    ):
        result = learned["methods"][key]
        rows.append(
            [
                label,
                "1000",
                _fmt(result["chamfer"]["mean"]),
                _fmt(result["p2s_mean"]["mean"]),
                _fmt(result["p2s_p95"]["mean"]),
                _fmt(result["fscore"]["mean"]),
                _fmt(result["normal_consistency"]["mean"]),
                str(result.get("improved_meshes", "—")),
            ]
        )
    for label, key in EXTERNAL:
        result = payload["external_evaluation"][key]
        metrics = result["metrics"]
        completed = result["completed_samples"]
        rows.append(
            [
                label,
                f"{completed}/1000",
                _stat(metrics["refined_chamfer"]),
                _stat(metrics["refined_p2s_mean"]),
                _stat(metrics["refined_p2s_p95"]),
                _stat(metrics["refined_fscore"]),
                _stat(metrics["refined_normal_consistency"]),
                str(result["improved_meshes"]),
            ]
        )
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Future2000 unified refinement report",
        "",
        "## Contract and protocol",
        "",
        f"- Test set: exactly {payload['test_samples']} held-out variants (200 objects × 5).",
        f"- Input boundary: {payload['input_contract']}.",
        "- Learned arms use the same backbone/config; only the supervised target and recovery differ.",
        "- Surface protocol: 3,000 deterministic samples, seed 7, F-score threshold 0.01.",
        "",
        "## Training",
        "",
        "| Arm | Steps | Epochs | Epoch-1 train | Epoch-1 val | Final train | Final val | Train drop | Val drop | Runtime (h) | Peak GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Learned Laplacian", "laplacian"), ("Direct displacement", "direct_displacement")):
        item = payload["training"][key]
        lines.append(
            f"| {label} | {item['optimizer_steps']} | {item['completed_epochs']} | "
            f"{_fmt(item['first_epoch_train_loss'])} | {_fmt(item['first_epoch_validation_loss'])} | "
            f"{_fmt(item['final_train_loss'])} | {_fmt(item['final_validation_loss'])} | "
            f"{item['train_loss_drop_fraction']:.2%} | {item['validation_loss_drop_fraction']:.2%} | "
            f"{item['runtime_seconds']/3600:.2f} | {_fmt(item['peak_gpu_memory_mb'])} |"
        )
    lines.extend(
        [
            "",
            "## Held-out geometry results",
            "",
            "| Method | Completed | Chamfer ↓ | P2S mean ↓ | P2S p95 ↓ | F-score ↑ | Normal consistency ↑ | Improved meshes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _metric_rows(payload):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## External baseline status", ""])
    for label, info in payload["baseline_status"].items():
        failures = info.get("failed_samples", 0)
        lines.append(f"- {label}: **{info['status']}**, failed={failures}.")
        for reason, count in info.get("failure_reasons", {}).items():
            lines.append(f"  - {count}× {reason}")
    lines.extend(["", "## Qualitative comparisons", ""])
    for record in payload["qualitative"]["records"]:
        lines.append(f"- `{record['sample_id']}`: `{record['image']}`")
    lines.extend(
        [
            "",
            "The JSON companion contains means, medians, standard deviations, bootstrap 95% confidence intervals, runtime, memory, diagnostics, pinned commits, and per-shard metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def _commands(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "#!/bin/bash",
            "# Reproducibility record. Job IDs are also stored in FINAL_REPORT.json.",
            "# Training/evaluation scripts and pinned baseline config live in this repository.",
            f"# learned_analysis={args.learned_analysis}",
            f"# external_output={args.external_output}",
            f"# job_ids={','.join(args.job_ids)}",
            "",
        ]
    )


def _stat(value: dict[str, Any] | None) -> str:
    return "N/A" if value is None else _fmt(value["mean"])


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laplacian-run", type=Path, required=True)
    parser.add_argument("--displacement-run", type=Path, required=True)
    parser.add_argument("--learned-analysis", type=Path, required=True)
    parser.add_argument("--external-output", type=Path, required=True)
    parser.add_argument("--qualitative-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-ids", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Unified Sofa50 comparison for C0/C1/C2, F0/F1/F2, and C2F2 3-seed runs.

This script intentionally reuses the repository's existing capacity evaluator
(`scripts/analyze_sofa50_capacity_2000.py::evaluate_arm`) so that all runs use
exactly the same metric contract:

- exact GT-query validation
- original RGB / zero RGB / shuffled RGB / cross-object RGB / shuffled-view-order
- zero-predictor baseline
- all / smooth-bottom-90 / high-top-10 / high-top-1 groups
- endpoint error, cosine, prediction/GT norm ratio, and RGB dependence

Outputs:
  <output-dir>/main_comparison.csv
  <output-dir>/c2f2_3seed_stats.csv
  <output-dir>/pairwise_changes.csv
  <output-dir>/summary.json
  <output-dir>/REPORT.md
  <output-dir>/predictions/<run-label>/...

The script also records the configured optimizer-step budget and marks pairwise
comparisons as same-budget or mixed-budget. Mixed-budget comparisons are kept
for reference but should not be used as causal capacity/resolution conclusions.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
SRC = REPO_ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


DEFAULT_MANIFEST = Path.home() / "sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs/learned_laplacian"
DEFAULT_C2F2_ROOT = DEFAULT_RUNS_ROOT / "sofa50_c2_f2_50000step_3seed"
DEFAULT_OUTPUT = DEFAULT_RUNS_ROOT / "sofa50_cf_c2f2_comparison"
SEEDS = (7, 17, 27)

RESOLUTION_ARMS = {
    "C0F0": "image_resolution_f0",
    "C0F1": "image_resolution_f1",
    "C0F2_res": "image_resolution_f2",
}
CAPACITY_ARMS = {
    "C0F2_cap": "C0_16_64",
    "C1F2": "C1_32_128",
    "C2F2_cap": "C2_64_256",
}

REPORT_GROUPS = ("all", "smooth_bottom_90", "high_top_10", "high_top_1")

# Metrics reported in the compact CSV/statistics tables.
METRIC_FIELDS = (
    "all_endpoint",
    "smooth90_endpoint",
    "top10_endpoint",
    "top1_endpoint",
    "all_global_cosine",
    "top10_global_cosine",
    "all_norm_ratio",
    "all_rgb_gap",
    "top10_rgb_gap",
    "top1_rgb_gap",
    "all_rgb_relative_improvement",
    "all_improvement_vs_zero_predictor",
    "original_training_loss",
)

LOWER_IS_BETTER = {
    "all_endpoint",
    "smooth90_endpoint",
    "top10_endpoint",
    "top1_endpoint",
    "original_training_loss",
}
HIGHER_IS_BETTER = set(METRIC_FIELDS) - LOWER_IS_BETTER


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_capacity_analyzer():
    path = REPO_ROOT / "scripts/analyze_sofa50_capacity_2000.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing existing analyzer: {path}\n"
            "This comparison script deliberately reuses evaluate_arm() from it."
        )
    spec = importlib.util.spec_from_file_location("sofa50_capacity_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "evaluate_arm"):
        raise AttributeError(f"{path} has no evaluate_arm()")
    return module


def score_root(path: Path, tokens: Iterable[str]) -> tuple[int, float]:
    name = path.name.lower()
    score = sum(1 for token in tokens if token in name)
    if "50000" in name or "50k" in name:
        score += 10
    return score, path.stat().st_mtime


def discover_resolution_root(runs_root: Path) -> Path | None:
    candidates: list[Path] = []
    if not runs_root.is_dir():
        return None
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if all((p / "arms" / arm).is_dir() for arm in RESOLUTION_ARMS.values()):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: score_root(p, ("resolution", "image")))


def discover_capacity_root(runs_root: Path) -> Path | None:
    candidates: list[Path] = []
    if not runs_root.is_dir():
        return None
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if all((p / arm).is_dir() for arm in CAPACITY_ARMS.values()):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: score_root(p, ("capacity", "ablation")))


def config_path(run_dir: Path) -> Path:
    for name in ("config.json", "launch_config.json", "run_config.json"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No config file found in {run_dir}")


def configured_steps(config: Mapping[str, Any]) -> int | None:
    multi = config.get("multi_object_training", {})
    value = multi.get("max_optimizer_steps")
    return int(value) if isinstance(value, (int, float)) else None


def completed_steps(run_dir: Path, config: Mapping[str, Any]) -> int | None:
    screening = run_dir / "screening_summary.json"
    if screening.is_file():
        value = read_json(screening).get("optimizer_steps")
        if isinstance(value, (int, float)):
            return int(value)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        for key in (
            "optimizer_steps",
            "completed_optimizer_steps",
            "global_optimizer_steps",
            "max_optimizer_steps",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        if metrics.get("stop_reason") == "max_optimizer_steps":
            return configured_steps(config)

    return configured_steps(config)


def run_metadata(run_dir: Path, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    config = evaluation["config"]
    image = config.get("image_encoder", {})
    model = config.get("model", {})
    first_stride = int(image.get("first_stride", 1))
    second_stride = int(image.get("second_stride", 1))
    stride_product = first_stride * second_stride
    nominal_resolution = 960 // stride_product if stride_product > 0 else None
    return {
        "run_dir": str(run_dir),
        "seed": int(config.get("seed", -1)),
        "configured_steps": configured_steps(config),
        "completed_steps": completed_steps(run_dir, config),
        "feature_dim": image.get("feature_dim"),
        "hidden_dim": model.get("hidden_dim"),
        "graph_layers": model.get("num_graph_layers"),
        "first_stride": first_stride,
        "second_stride": second_stride,
        "nominal_feature_resolution": nominal_resolution,
        "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
        "parameter_count": evaluation.get("parameter_count", {}).get("total"),
    }


def extract_row(label: str, family: str, run_dir: Path, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = evaluation["metrics"]
    conditions = aggregate["conditions"]
    original = conditions["original_rgb"]
    image_dep = aggregate["image_dependence"]
    vs_zero = aggregate["improvement_vs_zero_predictor"]

    row = {
        "label": label,
        "family": family,
        **run_metadata(run_dir, evaluation),
        "all_endpoint": original["all"]["mean_endpoint_error"],
        "smooth90_endpoint": original["smooth_bottom_90"]["mean_endpoint_error"],
        "top10_endpoint": original["high_top_10"]["mean_endpoint_error"],
        "top1_endpoint": original["high_top_1"]["mean_endpoint_error"],
        "all_global_cosine": original["all"].get("global_cosine"),
        "top10_global_cosine": original["high_top_10"].get("global_cosine"),
        "all_norm_ratio": original["all"].get("prediction_to_gt_global_norm_ratio"),
        "all_rgb_gap": image_dep["all"]["endpoint_zero_minus_original"],
        "top10_rgb_gap": image_dep["high_top_10"]["endpoint_zero_minus_original"],
        "top1_rgb_gap": image_dep["high_top_1"]["endpoint_zero_minus_original"],
        "all_rgb_relative_improvement": image_dep["all"]["endpoint_relative_improvement_with_rgb"],
        "all_improvement_vs_zero_predictor": vs_zero["all"]["endpoint_relative_improvement_vs_zero_predictor"],
        "original_training_loss": original.get("training_loss"),
    }
    return row


def evaluate_one(
    analyzer: Any,
    dataset: PreparedMeshDataset,
    device: torch.device,
    label: str,
    family: str,
    run_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"Missing best.pt: {run_dir / 'best.pt'}")
    cfg = read_json(config_path(run_dir))
    seed = int(cfg.get("seed", 7))
    print(f"[evaluate] {label}: seed={seed} dir={run_dir}", flush=True)
    evaluation = analyzer.evaluate_arm(
        label,
        run_dir,
        dataset,
        device,
        seed,
        output_dir / "predictions" / label,
    )
    return extract_row(label, family, run_dir, evaluation), dict(evaluation)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean_std_rows(rows: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metric in METRIC_FIELDS:
        values = np.asarray([float(r[metric]) for r in rows if finite(r.get(metric))], dtype=np.float64)
        if values.size == 0:
            continue
        result.append(
            {
                "group": group_name,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return result


def row_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in rows}


def pairwise_row(name: str, baseline: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    base_steps = baseline.get("completed_steps")
    target_steps = target.get("completed_steps")
    out: dict[str, Any] = {
        "comparison": name,
        "baseline": baseline["label"],
        "target": target["label"],
        "baseline_steps": base_steps,
        "target_steps": target_steps,
        "same_budget": base_steps is not None and target_steps is not None and int(base_steps) == int(target_steps),
    }
    for metric in METRIC_FIELDS:
        a = baseline.get(metric)
        b = target.get(metric)
        if not (finite(a) and finite(b)):
            continue
        a = float(a)
        b = float(b)
        out[f"{metric}_baseline"] = a
        out[f"{metric}_target"] = b
        out[f"{metric}_delta"] = b - a
        if metric in LOWER_IS_BETTER:
            out[f"{metric}_relative_improvement"] = (a - b) / max(abs(a), 1e-12)
        elif metric in HIGHER_IS_BETTER:
            out[f"{metric}_improvement"] = b - a
    return out


def build_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = row_by_label(rows)
    specs = [
        ("resolution_F1_vs_F0", "C0F0", "C0F1"),
        ("resolution_F2_vs_F0", "C0F0", "C0F2_res"),
        ("resolution_F2_vs_F1", "C0F1", "C0F2_res"),
        ("capacity_C1_vs_C0", "C0F2_cap", "C1F2"),
        ("capacity_C2_vs_C0", "C0F2_cap", "C2F2_cap"),
        ("capacity_C2_vs_C1", "C1F2", "C2F2_cap"),
        ("selected_C2F2_seed7_vs_C0F2", "C0F2_res", "C2F2_seed7"),
    ]
    result: list[dict[str, Any]] = []
    for name, a, b in specs:
        if a in by and b in by:
            result.append(pairwise_row(name, by[a], by[b]))
    return result


def mean_metric(stats_rows: list[dict[str, Any]], metric: str) -> tuple[float | None, float | None]:
    for row in stats_rows:
        if row["metric"] == metric:
            return float(row["mean"]), float(row["std"])
    return None, None


def fmt(value: Any, digits: int = 6) -> str:
    if not finite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run",
        "Steps",
        "Seed",
        "Feat",
        "Hidden",
        "Res",
        "All EPE↓",
        "Top10↓",
        "Top1↓",
        "Cos↑",
        "Norm",
        "RGB gap↑",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["label"]),
                    str(r.get("completed_steps", "?")),
                    str(r.get("seed", "?")),
                    str(r.get("feature_dim", "?")),
                    str(r.get("hidden_dim", "?")),
                    str(r.get("nominal_feature_resolution", "?")),
                    fmt(r.get("all_endpoint")),
                    fmt(r.get("top10_endpoint")),
                    fmt(r.get("top1_endpoint")),
                    fmt(r.get("all_global_cosine")),
                    fmt(r.get("all_norm_ratio")),
                    fmt(r.get("all_rgb_gap")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(
    rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    roots: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Sofa50 C/F/C2F2 comparison")
    lines.append("")
    lines.append("All checkpoints are freshly evaluated with the same exact-query metric contract from `analyze_sofa50_capacity_2000.py::evaluate_arm`.")
    lines.append("")
    lines.append("## Main comparison")
    lines.append("")
    lines.append(markdown_table(rows))
    lines.append("")

    lines.append("## C2F2 three-seed robustness")
    lines.append("")
    for metric in ("all_endpoint", "top10_endpoint", "top1_endpoint", "all_global_cosine", "all_rgb_gap"):
        mean, std = mean_metric(stats_rows, metric)
        lines.append(f"- `{metric}`: {fmt(mean)} ± {fmt(std)}")
    lines.append("")

    lines.append("## Controlled comparisons")
    lines.append("")
    if not pairwise:
        lines.append("No pairwise comparisons were available.")
    else:
        for p in pairwise:
            budget = "same budget" if p["same_budget"] else "MIXED BUDGET"
            all_imp = p.get("all_endpoint_relative_improvement")
            top10_imp = p.get("top10_endpoint_relative_improvement")
            cos_delta = p.get("all_global_cosine_improvement")
            rgb_delta = p.get("all_rgb_gap_improvement")
            lines.append(
                f"- **{p['comparison']}** ({budget}): "
                f"all-EPE improvement={fmt(None if all_imp is None else 100.0 * all_imp)}%, "
                f"top10 improvement={fmt(None if top10_imp is None else 100.0 * top10_imp)}%, "
                f"cosine Δ={fmt(cos_delta)}, RGB-gap Δ={fmt(rgb_delta)}."
            )
    lines.append("")

    mixed = [p for p in pairwise if not p["same_budget"]]
    if mixed:
        lines.append("## Budget warning")
        lines.append("")
        lines.append("The following comparisons use different optimizer-step budgets and are descriptive only:")
        for p in mixed:
            lines.append(f"- {p['comparison']}: {p['baseline_steps']} vs {p['target_steps']} steps")
        lines.append("")

    if "C0F2_res" in row_by_label(rows) and "C0F2_cap" in row_by_label(rows):
        a = row_by_label(rows)["C0F2_res"]
        b = row_by_label(rows)["C0F2_cap"]
        lines.append("## C0F2 anchor audit")
        lines.append("")
        lines.append(
            "`C0F2_res` and `C0F2_cap` are separate training runs of the common L-shaped anchor. "
            "They are kept separate rather than silently merged."
        )
        lines.append(
            f"- all-EPE difference (cap - res): {fmt(float(b['all_endpoint']) - float(a['all_endpoint']))}"
        )
        lines.append("")

    lines.append("## Run roots")
    lines.append("")
    for key, value in roots.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("Interpretation rule: use C0F0→C0F1→C0F2 for resolution, C0F2→C1F2→C2F2 for capacity only when budgets match, and the three new C2F2 seeds for robustness of the selected configuration.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--resolution-root", type=Path, default=None)
    parser.add_argument("--capacity-root", type=Path, default=None)
    parser.add_argument("--c2f2-root", type=Path, default=DEFAULT_C2F2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-capacity",
        action="store_true",
        help="Skip old C0/C1/C2 runs if only the resolution and new C2F2 runs are needed.",
    )
    parser.add_argument(
        "--skip-resolution",
        action="store_true",
        help="Skip old F0/F1/F2 runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    c2f2_root = args.c2f2_root.expanduser().resolve()

    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    resolution_root = None
    if not args.skip_resolution:
        resolution_root = (
            args.resolution_root.expanduser().resolve()
            if args.resolution_root is not None
            else discover_resolution_root(runs_root)
        )
        if resolution_root is None:
            raise FileNotFoundError(
                "Could not auto-discover an F0/F1/F2 resolution root. "
                "Pass --resolution-root explicitly."
            )

    capacity_root = None
    if not args.skip_capacity:
        capacity_root = (
            args.capacity_root.expanduser().resolve()
            if args.capacity_root is not None
            else discover_capacity_root(runs_root)
        )
        if capacity_root is None:
            raise FileNotFoundError(
                "Could not auto-discover a C0/C1/C2 capacity root. "
                "Pass --capacity-root explicitly or use --skip-capacity."
            )

    if not c2f2_root.is_dir():
        raise FileNotFoundError(f"C2F2 3-seed root not found: {c2f2_root}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    analyzer = load_capacity_analyzer()
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}

    if resolution_root is not None:
        for label, dirname in RESOLUTION_ARMS.items():
            run_dir = resolution_root / "arms" / dirname
            row, raw = evaluate_one(analyzer, dataset, device, label, "resolution", run_dir, output_dir)
            rows.append(row)
            raw_results[label] = raw

    if capacity_root is not None:
        for label, dirname in CAPACITY_ARMS.items():
            run_dir = capacity_root / dirname
            row, raw = evaluate_one(analyzer, dataset, device, label, "capacity", run_dir, output_dir)
            rows.append(row)
            raw_results[label] = raw

    new_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        label = f"C2F2_seed{seed}"
        run_dir = c2f2_root / f"seed_{seed}"
        row, raw = evaluate_one(analyzer, dataset, device, label, "selected_3seed", run_dir, output_dir)
        if int(row["seed"]) != seed:
            raise ValueError(f"{label} config seed is {row['seed']}, expected {seed}")
        rows.append(row)
        new_seed_rows.append(row)
        raw_results[label] = raw

    stats_rows = mean_std_rows(new_seed_rows, "C2F2_3seed")
    pairwise = build_pairwise(rows)

    # Compact aggregate row for convenience in plotting/tables.
    mean_row: dict[str, Any] = {
        "label": "C2F2_3seed_mean",
        "family": "selected_3seed_summary",
        "seed": "7,17,27",
        "configured_steps": new_seed_rows[0].get("configured_steps"),
        "completed_steps": new_seed_rows[0].get("completed_steps"),
        "feature_dim": new_seed_rows[0].get("feature_dim"),
        "hidden_dim": new_seed_rows[0].get("hidden_dim"),
        "graph_layers": new_seed_rows[0].get("graph_layers"),
        "first_stride": new_seed_rows[0].get("first_stride"),
        "second_stride": new_seed_rows[0].get("second_stride"),
        "nominal_feature_resolution": new_seed_rows[0].get("nominal_feature_resolution"),
    }
    for stat in stats_rows:
        mean_row[stat["metric"]] = stat["mean"]
        mean_row[f"{stat['metric']}_std"] = stat["std"]

    main_rows = rows + [mean_row]

    roots = {
        "manifest": str(manifest),
        "resolution_root": str(resolution_root) if resolution_root is not None else None,
        "capacity_root": str(capacity_root) if capacity_root is not None else None,
        "c2f2_root": str(c2f2_root),
        "output_dir": str(output_dir),
    }

    summary = {
        "experiment": "Sofa50 unified C0/C1/C2 + F0/F1/F2 + C2F2 3-seed comparison",
        "metric_contract": "scripts/analyze_sofa50_capacity_2000.py::evaluate_arm",
        "roots": roots,
        "rows": main_rows,
        "c2f2_3seed_stats": stats_rows,
        "pairwise_changes": pairwise,
        "notes": [
            "C0F0/C0F1/C0F2_res form the controlled resolution ablation.",
            "C0F2_cap/C1F2/C2F2_cap form the historical capacity ablation.",
            "C0F2_res and C0F2_cap are separate training runs of the common L-shaped anchor and are not silently merged.",
            "C2F2_seed7/17/27 are the new 50k selected-configuration robustness runs.",
            "Pairwise rows explicitly report whether optimizer-step budgets match.",
            "Do not infer C×F interaction from this L-shaped design because C1F0/C1F1/C2F0/C2F1 are absent.",
        ],
    }

    write_csv(output_dir / "main_comparison.csv", main_rows)
    write_csv(output_dir / "c2f2_3seed_stats.csv", stats_rows)
    write_csv(output_dir / "pairwise_changes.csv", pairwise)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        build_report(rows, stats_rows, pairwise, roots), encoding="utf-8"
    )

    print("\n=== Done ===", flush=True)
    for name in ("REPORT.md", "main_comparison.csv", "c2f2_3seed_stats.csv", "pairwise_changes.csv", "summary.json"):
        print(output_dir / name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

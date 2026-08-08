#!/usr/bin/env python3
from __future__ import annotations

VERSION = "2026-08-08-expanded-query-fix-v2"

"""Full Sofa50 comparison: prediction + real-expanded recovery + renderings.

Compares the existing 50k C0F0/C0F1/C0F2 resolution arms with the new
C2F2 50k seeds (7, 17, 27).

Prediction metrics reuse scripts/analyze_sofa50_capacity_2000.py::evaluate_arm.
Expanded recovery reuses mlr.learned_laplacian.canonical_experiment._evaluate_expanded,
which in turn uses the canonical current-graph recovery and reconstruction code.

Outputs include:
  prediction_comparison.csv
  prediction_pairwise.csv
  c2f2_prediction_3seed_stats.csv
  expanded_comparison.csv                 # primary main_confidence rows
  expanded_all_variants.csv               # main/hard-visibility/zero-RGB
  expanded_per_mesh.csv
  c2f2_expanded_3seed_stats.csv
  summary.json
  REPORT.md
  prediction_outputs/<label>/...
  expanded_runs/<label>/recovered_meshes/...
  expanded_runs/<label>/fixed-camera_visualizations/*.png
  renderings/cross_model_main_confidence/*.png
"""

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "scripts"
    else Path.cwd()
)
SRC = REPO_ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.canonical_experiment import _evaluate_expanded, _first_camera
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.visualization import render_mesh_comparison_grid


DEFAULT_GT_MANIFEST = (
    Path.home() / "sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
)
DEFAULT_EXPANDED_MANIFEST = (
    Path.home()
    / "sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json"
)
DEFAULT_RUNS = REPO_ROOT / "runs/learned_laplacian"
DEFAULT_RESOLUTION_ROOT = DEFAULT_RUNS / "sofa50_image_resolution_ablation_50000step"
DEFAULT_C2F2_ROOT = DEFAULT_RUNS / "sofa50_c2_f2_50000step_3seed"
DEFAULT_OUTPUT = DEFAULT_RUNS / "sofa50_cf_c2f2_comparison_full"

RESOLUTION_ARMS = {
    "C0F0": "image_resolution_f0",
    "C0F1": "image_resolution_f1",
    "C0F2": "image_resolution_f2",
}
SEEDS = (7, 17, 27)

PRED_METRICS = (
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
)

EXPANDED_STATS = (
    "refined_chamfer",
    "refined_point_to_surface",
    "refined_normal_consistency",
    "introduced_flips",
    "new_degeneracies",
    "better_than_initial",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}" if finite(value) else "n/a"


def load_capacity_analyzer() -> Any:
    path = REPO_ROOT / "scripts/analyze_sofa50_capacity_2000.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing analyzer: {path}")
    spec = importlib.util.spec_from_file_location("sofa50_capacity_analyzer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "evaluate_arm"):
        raise AttributeError(f"{path} has no evaluate_arm()")
    return module


def config_path(run_dir: Path) -> Path:
    for name in ("config.json", "launch_config.json", "run_config.json"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No config file found in {run_dir}")


def best_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No best checkpoint found in {run_dir}")


def configured_steps(config: Mapping[str, Any]) -> int | None:
    value = config.get("multi_object_training", {}).get("max_optimizer_steps")
    return int(value) if isinstance(value, (int, float)) else None


def completed_steps(run_dir: Path, config: Mapping[str, Any]) -> int | None:
    screening = run_dir / "screening_summary.json"
    if screening.is_file():
        value = read_json(screening).get("optimizer_steps")
        if isinstance(value, (int, float)):
            return int(value)
    metrics = run_dir / "metrics.json"
    if metrics.is_file():
        payload = read_json(metrics)
        for key in (
            "optimizer_steps",
            "completed_optimizer_steps",
            "global_optimizer_steps",
            "max_optimizer_steps",
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        if payload.get("stop_reason") == "max_optimizer_steps":
            return configured_steps(config)
    return configured_steps(config)


def source_runs(resolution_root: Path, c2f2_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for label, dirname in RESOLUTION_ARMS.items():
        runs.append(
            {
                "label": label,
                "family": "resolution",
                "source_dir": resolution_root / "arms" / dirname,
                "expected_seed": 7,
            }
        )
    for seed in SEEDS:
        runs.append(
            {
                "label": f"C2F2_seed{seed}",
                "family": "selected_3seed",
                "source_dir": c2f2_root / f"seed_{seed}",
                "expected_seed": seed,
            }
        )
    return runs


def validate_run(run: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["source_dir"]).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")
    cfg_path = config_path(run_dir)
    checkpoint = best_checkpoint(run_dir)
    config = read_json(cfg_path)
    seed = int(config.get("seed", -1))
    expected_seed = int(run["expected_seed"])
    if seed != expected_seed:
        raise ValueError(
            f"{run['label']} config seed={seed}, expected {expected_seed}: {cfg_path}"
        )
    return {
        **dict(run),
        "source_dir": run_dir,
        "config_path": cfg_path,
        "checkpoint": checkpoint,
        "config": config,
    }


def prediction_row(
    label: str,
    family: str,
    run_dir: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    config = evaluation["config"]
    image = config.get("image_encoder", {})
    model = config.get("model", {})
    first_stride = int(image.get("first_stride", 1))
    second_stride = int(image.get("second_stride", 1))
    nominal_res = 960 // max(first_stride * second_stride, 1)
    aggregate = evaluation["metrics"]
    original = aggregate["conditions"]["original_rgb"]
    image_dep = aggregate["image_dependence"]
    return {
        "label": label,
        "family": family,
        "steps": completed_steps(run_dir, config),
        "seed": int(config.get("seed", -1)),
        "feature_dim": image.get("feature_dim"),
        "hidden_dim": model.get("hidden_dim"),
        "feature_resolution": nominal_res,
        "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
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
    }


def evaluate_prediction(
    analyzer: Any,
    dataset: PreparedMeshDataset,
    device: torch.device,
    run: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = str(run["label"])
    run_dir = Path(run["source_dir"])
    seed = int(run["config"].get("seed", 7))
    print(f"[prediction] {label}: {run_dir}", flush=True)
    evaluation = analyzer.evaluate_arm(
        label,
        run_dir,
        dataset,
        device,
        seed,
        output_dir / "prediction_outputs" / label,
    )
    return prediction_row(label, str(run["family"]), run_dir, evaluation), dict(evaluation)


def prepare_expanded_workspace(run: Mapping[str, Any], output_dir: Path) -> Path:
    workspace = output_dir / "expanded_runs" / str(run["label"])
    workspace.mkdir(parents=True, exist_ok=True)
    # Canonical _evaluate_expanded requires these exact filenames.
    shutil.copy2(Path(run["config_path"]), workspace / "config.json")
    target = workspace / "checkpoint_best.pt"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.symlink(Path(run["checkpoint"]).resolve(), target)
    except OSError:
        shutil.copy2(Path(run["checkpoint"]), target)
    return workspace


def main_variant(aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in aggregate:
        if row.get("variant") == "main_confidence":
            return dict(row)
    raise KeyError("Expanded evaluation returned no main_confidence row")


def evaluate_expanded_run(
    run: Mapping[str, Any],
    expanded_manifest: Path,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]], Path]:
    label = str(run["label"])
    workspace = prepare_expanded_workspace(run, output_dir)
    config = read_json(workspace / "config.json")

    # Real expanded-query samples are inference inputs, not GT-query augmentation
    # samples. Mirror run_canonical_experiment_evaluation(): disable the training
    # query-perturbation contract before _prepare_object_static() sees them.
    expanded_config = copy.deepcopy(config)
    query_cfg = expanded_config.setdefault("query_training", {})
    query_cfg["enabled"] = False
    query_cfg["apply_to_validation"] = False
    query_cfg["zero_initial_laplacian"] = True

    print(f"[expanded] {label}: {expanded_manifest}", flush=True)
    per_mesh, aggregate, failures = _evaluate_expanded(
        workspace,
        expanded_manifest,
        expanded_config,
        device,
    )
    main = main_variant(aggregate)
    initial = main.get("initial_chamfer")
    refined = main.get("refined_chamfer")
    rel = None
    if finite(initial) and finite(refined):
        rel = (float(initial) - float(refined)) / max(abs(float(initial)), 1e-12)
    row = {
        "label": label,
        "family": run["family"],
        "seed": int(config.get("seed", -1)),
        "steps": completed_steps(Path(run["source_dir"]), config),
        "initial_chamfer": initial,
        "refined_chamfer": refined,
        "chamfer_relative_improvement_vs_initial": rel,
        "refined_point_to_surface": main.get("refined_point_to_surface"),
        "refined_normal_consistency": main.get("refined_normal_consistency"),
        "introduced_flips": main.get("introduced_flips"),
        "new_degeneracies": main.get("new_degeneracies"),
        "better_than_initial": main.get("better_than_initial"),
        "mesh_count": main.get("mesh_count"),
        "workspace": str(workspace),
    }
    all_variant_rows: list[dict[str, Any]] = []
    for agg in aggregate:
        if agg.get("refined_chamfer") is None:
            continue
        all_variant_rows.append(
            {
                "label": label,
                "seed": int(config.get("seed", -1)),
                **dict(agg),
            }
        )
    per_mesh_rows = [
        {"label": label, "seed": int(config.get("seed", -1)), **dict(x)}
        for x in per_mesh
    ]
    return row, all_variant_rows, per_mesh_rows, failures, workspace


def stats(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], group: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for metric in metrics:
        values = np.asarray(
            [float(row[metric]) for row in rows if finite(row.get(metric))],
            dtype=np.float64,
        )
        if not len(values):
            continue
        out.append(
            {
                "group": group,
                "metric": metric,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return out


def pairwise_prediction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by = {str(row["label"]): row for row in rows}
    specs = (
        ("F1_vs_F0", "C0F0", "C0F1"),
        ("F2_vs_F0", "C0F0", "C0F2"),
        ("F2_vs_F1", "C0F1", "C0F2"),
        ("C2F2_seed7_vs_C0F2", "C0F2", "C2F2_seed7"),
    )
    out: list[dict[str, Any]] = []
    for name, a_name, b_name in specs:
        if a_name not in by or b_name not in by:
            continue
        a, b = by[a_name], by[b_name]
        row: dict[str, Any] = {"comparison": name, "baseline": a_name, "target": b_name}
        for metric in PRED_METRICS:
            av, bv = a.get(metric), b.get(metric)
            if not (finite(av) and finite(bv)):
                continue
            avf, bvf = float(av), float(bv)
            row[f"{metric}_baseline"] = avf
            row[f"{metric}_target"] = bvf
            row[f"{metric}_delta"] = bvf - avf
            if "endpoint" in metric:
                row[f"{metric}_relative_improvement"] = (avf - bvf) / max(abs(avf), 1e-12)
        out.append(row)
    return out


def recovery_contract_audit(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contracts = {str(run["label"]): run["config"].get("recovery", {}) for run in runs}
    serialized = {json.dumps(value, sort_keys=True) for value in contracts.values()}
    return {
        "same_recovery_config": len(serialized) == 1,
        "configs": contracts,
    }


def render_cross_model_grids(
    expanded_manifest: Path,
    workspaces: Mapping[str, Path],
    output_dir: Path,
) -> list[dict[str, str]]:
    dataset = PreparedMeshDataset.from_manifest(expanded_manifest, "validation")
    failures: list[dict[str, str]] = []
    render_dir = output_dir / "renderings" / "cross_model_main_confidence"
    render_dir.mkdir(parents=True, exist_ok=True)
    ordered_labels = ["C0F0", "C0F1", "C0F2", "C2F2_seed7", "C2F2_seed17", "C2F2_seed27"]
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        try:
            vertices = static["vertices"].detach().cpu().numpy()
            faces = static["faces"].detach().cpu().numpy()
            panels: list[tuple[str, Mesh]] = [
                ("Initial", Mesh(vertices, faces).ensure_normals())
            ]
            if static.get("gt_vertices") is not None and static.get("gt_faces") is not None:
                panels.append(
                    (
                        "GT",
                        Mesh(
                            static["gt_vertices"].detach().cpu().numpy(),
                            static["gt_faces"].detach().cpu().numpy(),
                        ).ensure_normals(),
                    )
                )
            for label in ordered_labels:
                workspace = workspaces[label]
                mesh_path = (
                    workspace
                    / "recovered_meshes"
                    / sample_id
                    / "main_confidence"
                    / "predicted_refined.obj"
                )
                if not mesh_path.is_file():
                    raise FileNotFoundError(mesh_path)
                panels.append((label, load_mesh(mesh_path)))
            camera = _first_camera(static, image_size=320)
            render_mesh_comparison_grid(
                panels,
                camera,
                render_dir / f"{sample_id}.png",
                image_size=320,
                columns=3,
            )
        except Exception as error:
            failures.append(
                {
                    "sample_id": sample_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return failures


def prediction_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = ["Run", "Seed", "Res", "All EPE↓", "Top10↓", "Top1↓", "Cos↑", "Norm", "RGB gap↑"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row.get("seed")),
                    str(row.get("feature_resolution")),
                    fmt(row.get("all_endpoint")),
                    fmt(row.get("top10_endpoint")),
                    fmt(row.get("top1_endpoint")),
                    fmt(row.get("all_global_cosine")),
                    fmt(row.get("all_norm_ratio")),
                    fmt(row.get("all_rgb_gap")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def expanded_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = ["Run", "Init Chamfer", "Refined↓", "Δ vs init", "P2S↓", "Normal↑", "Flips↓", "Better"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        imp = row.get("chamfer_relative_improvement_vs_initial")
        imp_text = f"{100.0 * float(imp):.3f}%" if finite(imp) else "n/a"
        better = (
            f"{row.get('better_than_initial')}/{row.get('mesh_count')}"
            if row.get("mesh_count") is not None
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(row.get("initial_chamfer")),
                    fmt(row.get("refined_chamfer")),
                    imp_text,
                    fmt(row.get("refined_point_to_surface")),
                    fmt(row.get("refined_normal_consistency")),
                    str(row.get("introduced_flips", "n/a")),
                    better,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def stat_value(stat_rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[Any, Any]:
    for row in stat_rows:
        if row.get("metric") == metric:
            return row.get("mean"), row.get("std")
    return None, None


def build_report(
    prediction_rows: Sequence[Mapping[str, Any]],
    prediction_stats: Sequence[Mapping[str, Any]],
    expanded_rows: Sequence[Mapping[str, Any]],
    expanded_stats: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    roots: Mapping[str, Any],
    render_failures: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Sofa50 resolution + C2F2 full comparison",
        "",
        "Prediction metrics use the existing exact-query evaluator. Expanded metrics use the canonical real-expanded recovery path with the same expanded validation manifest.",
        "",
        "## Prediction comparison",
        "",
        prediction_table(prediction_rows),
        "",
        "## Real expanded recovery (main confidence)",
        "",
        expanded_table(expanded_rows),
        "",
        "## C2F2 three-seed robustness",
        "",
    ]
    for metric in ("all_endpoint", "top10_endpoint", "top1_endpoint", "all_global_cosine", "all_rgb_gap"):
        mean, std = stat_value(prediction_stats, metric)
        lines.append(f"- prediction `{metric}`: {fmt(mean)} ± {fmt(std)}")
    for metric in ("refined_chamfer", "refined_point_to_surface", "refined_normal_consistency", "better_than_initial"):
        mean, std = stat_value(expanded_stats, metric)
        lines.append(f"- expanded `{metric}`: {fmt(mean)} ± {fmt(std)}")
    lines.extend(
        [
            "",
            "## Recovery contract audit",
            "",
            f"- same recovery config across all compared runs: **{audit['same_recovery_config']}**",
            "- expanded manifest is shared across all runs.",
            "- each run uses its own best checkpoint but the same canonical current-graph recovery implementation.",
            "",
            "## Renderings",
            "",
            "Each expanded run retains the previous fixed-camera four-panel rendering (`initial`, `main_confidence`, `hard_visibility_only`, `zero_rgb`).",
            "Cross-model images are written under `renderings/cross_model_main_confidence/` and compare Initial, GT (when present), C0F0, C0F1, C0F2, and all three C2F2 seeds using the same camera for each validation object.",
            f"Cross-model rendering failures: `{len(render_failures)}`.",
            "",
            "## Run roots",
            "",
        ]
    )
    for key, value in roots.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path, default=DEFAULT_GT_MANIFEST)
    parser.add_argument("--expanded-manifest", type=Path, default=DEFAULT_EXPANDED_MANIFEST)
    parser.add_argument("--resolution-root", type=Path, default=DEFAULT_RESOLUTION_ROOT)
    parser.add_argument("--c2f2-root", type=Path, default=DEFAULT_C2F2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip exact-query prediction reevaluation and run expanded recovery/rendering only.",
    )
    parser.add_argument(
        "--skip-expanded",
        action="store_true",
        help="Skip expanded recovery/rendering and run prediction comparison only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_manifest = args.gt_manifest.expanduser().resolve()
    expanded_manifest = args.expanded_manifest.expanduser().resolve()
    resolution_root = args.resolution_root.expanduser().resolve()
    c2f2_root = args.c2f2_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (gt_manifest,):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.skip_expanded and not expanded_manifest.is_file():
        raise FileNotFoundError(expanded_manifest)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    raw_runs = source_runs(resolution_root, c2f2_root)
    runs = [validate_run(run) for run in raw_runs]
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    prediction_raw: dict[str, Any] = {}
    if not args.skip_prediction:
        analyzer = load_capacity_analyzer()
        gt_dataset = PreparedMeshDataset.from_manifest(gt_manifest, "validation")
        for run in runs:
            row, raw = evaluate_prediction(analyzer, gt_dataset, device, run, output_dir)
            prediction_rows.append(row)
            prediction_raw[str(run["label"])] = raw

    expanded_rows: list[dict[str, Any]] = []
    expanded_all_variants: list[dict[str, Any]] = []
    expanded_per_mesh: list[dict[str, Any]] = []
    expanded_failures: list[dict[str, str]] = []
    workspaces: dict[str, Path] = {}
    if not args.skip_expanded:
        for run in runs:
            row, variants, per_mesh, failures, workspace = evaluate_expanded_run(
                run, expanded_manifest, device, output_dir
            )
            expanded_rows.append(row)
            expanded_all_variants.extend(variants)
            expanded_per_mesh.extend(per_mesh)
            workspaces[str(run["label"])] = workspace
            expanded_failures.extend(
                [{"label": str(run["label"]), **failure} for failure in failures]
            )

    c2_pred = [row for row in prediction_rows if str(row["label"]).startswith("C2F2_seed")]
    pred_stats = stats(c2_pred, PRED_METRICS, "C2F2_3seed") if c2_pred else []
    c2_expanded = [row for row in expanded_rows if str(row["label"]).startswith("C2F2_seed")]
    exp_stats = stats(c2_expanded, EXPANDED_STATS, "C2F2_3seed") if c2_expanded else []
    pred_pairwise = pairwise_prediction(prediction_rows) if prediction_rows else []
    audit = recovery_contract_audit(runs)

    cross_render_failures: list[dict[str, str]] = []
    if not args.skip_expanded:
        cross_render_failures = render_cross_model_grids(
            expanded_manifest, workspaces, output_dir
        )

    roots = {
        "gt_manifest": str(gt_manifest),
        "expanded_manifest": str(expanded_manifest),
        "resolution_root": str(resolution_root),
        "c2f2_root": str(c2f2_root),
        "output_dir": str(output_dir),
    }

    summary = {
        "experiment": "Sofa50 C0F0/C0F1/C0F2 + C2F2 three-seed prediction/recovery/rendering comparison",
        "roots": roots,
        "prediction_rows": prediction_rows,
        "prediction_pairwise": pred_pairwise,
        "c2f2_prediction_stats": pred_stats,
        "expanded_main_confidence": expanded_rows,
        "expanded_all_variants": expanded_all_variants,
        "c2f2_expanded_stats": exp_stats,
        "recovery_contract_audit": audit,
        "expanded_render_failures": expanded_failures,
        "cross_model_render_failures": cross_render_failures,
        "notes": [
            "C0F0/C0F1/C0F2 isolate feature resolution at C0 capacity.",
            "C0F2 vs C2F2_seed7 is the matched-seed 50k capacity comparison.",
            "C2F2 seeds 7/17/27 quantify selected-model robustness.",
            "Primary expanded endpoint is whether refined Chamfer is lower than the same initial expanded mesh.",
            "All expanded runs use the real expanded manifest; no expanded GT differential target is fabricated.",
        ],
    }

    write_csv(output_dir / "prediction_comparison.csv", prediction_rows)
    write_csv(output_dir / "prediction_pairwise.csv", pred_pairwise)
    write_csv(output_dir / "c2f2_prediction_3seed_stats.csv", pred_stats)
    write_csv(output_dir / "expanded_comparison.csv", expanded_rows)
    write_csv(output_dir / "expanded_all_variants.csv", expanded_all_variants)
    write_csv(output_dir / "expanded_per_mesh.csv", expanded_per_mesh)
    write_csv(output_dir / "c2f2_expanded_3seed_stats.csv", exp_stats)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        build_report(
            prediction_rows,
            pred_stats,
            expanded_rows,
            exp_stats,
            audit,
            roots,
            cross_render_failures,
        ),
        encoding="utf-8",
    )

    print("\n=== Full comparison complete ===", flush=True)
    for name in (
        "REPORT.md",
        "prediction_comparison.csv",
        "expanded_comparison.csv",
        "expanded_all_variants.csv",
        "expanded_per_mesh.csv",
        "summary.json",
    ):
        print(output_dir / name, flush=True)
    print(output_dir / "renderings/cross_model_main_confidence", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

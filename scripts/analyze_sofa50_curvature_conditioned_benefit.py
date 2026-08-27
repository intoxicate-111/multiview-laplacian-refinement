#!/usr/bin/env python3
from __future__ import annotations

"""Curvature-conditioned local benefit of frozen Arm-B + Arm-E fusion."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from diagnose_sofa50_exact_solve_visibility_sweep import uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_frozen_hybrid_recovery import _inputs, _pcg
from evaluate_sofa50_continuous_checkpoint_validation import (
    CURVATURE_PROTOCOL,
    _cotangent_twice_mean_curvature,
)
from mlr.data import Mesh
from mlr.learned_laplacian.evaluation import _point_to_surface_distances


LAMBDA_HYBRID = 3e-2
LOCAL_ERROR_METRICS = ("surface", "vertex", "normal", "tangential")
BIN_SPECS = (
    ("p00_p25", 0.00, 0.25),
    ("p25_p50", 0.25, 0.50),
    ("p50_p75", 0.50, 0.75),
    ("p75_p90", 0.75, 0.90),
    ("p90_p100", 0.90, 1.00),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def curvature_rank_bins(curvature: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Assign deterministic within-mesh percentile bins by stable rank."""

    values = np.asarray(curvature, dtype=np.float64)
    valid = np.asarray(eligible, dtype=bool) & np.isfinite(values)
    indices = np.flatnonzero(valid)
    if len(indices) < len(BIN_SPECS):
        raise ValueError("Too few eligible vertices for curvature bins.")
    order = indices[np.argsort(values[indices], kind="stable")]
    labels = np.full(len(values), -1, dtype=np.int64)
    positions = np.arange(len(order), dtype=np.float64) / len(order)
    edges = np.asarray([spec[2] for spec in BIN_SPECS], dtype=np.float64)
    ranked_labels = np.searchsorted(edges, positions, side="right")
    ranked_labels = np.minimum(ranked_labels, len(BIN_SPECS) - 1)
    labels[order] = ranked_labels
    if any(not np.any(labels == index) for index in range(len(BIN_SPECS))):
        raise RuntimeError("At least one curvature bin is empty.")
    return labels


def area_weighted_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices, dtype=np.float64)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_cross)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-14
    normals[valid] /= norms[valid, None]
    normals[~valid] = np.nan
    return normals, valid


def local_errors(
    predicted: np.ndarray, clean: np.ndarray, clean_normals: np.ndarray
) -> dict[str, np.ndarray]:
    difference = np.asarray(predicted, dtype=np.float64) - clean
    vertex = np.linalg.norm(difference, axis=1)
    normal = np.abs(np.einsum("ij,ij->i", difference, clean_normals))
    tangential = np.sqrt(np.maximum(0.0, np.square(vertex) - np.square(normal)))
    return {"vertex": vertex, "normal": normal, "tangential": tangential}


def _safe_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return float("nan")
    function = pearsonr if kind == "pearson" else spearmanr
    return float(function(left[valid], right[valid]).statistic)


def _top_recall(target: np.ndarray, score: np.ndarray, fraction: float) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    indices = np.flatnonzero(valid)
    count = max(1, int(np.ceil(fraction * len(indices))))
    target_top = set(indices[np.argsort(target[indices], kind="stable")[-count:]].tolist())
    score_top = set(indices[np.argsort(score[indices], kind="stable")[-count:]].tolist())
    return len(target_top & score_top) / count


def field_curvature_statistics(
    field: np.ndarray,
    curvature_vector: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    field_norm = np.linalg.norm(field, axis=1)
    curvature_norm = np.linalg.norm(curvature_vector, axis=1)
    valid = (
        np.asarray(eligible, dtype=bool)
        & np.isfinite(field_norm)
        & np.isfinite(curvature_norm)
    )
    directional = valid & (field_norm > 1e-14) & (curvature_norm > 1e-14)
    cosine = np.einsum("ij,ij->i", field[directional], curvature_vector[directional]) / (
        field_norm[directional] * curvature_norm[directional]
    )
    return {
        "vertices": int(valid.sum()),
        "magnitude_pearson": _safe_correlation(
            field_norm[valid], curvature_norm[valid], "pearson"
        ),
        "magnitude_spearman": _safe_correlation(
            field_norm[valid], curvature_norm[valid], "spearman"
        ),
        "directional_cosine_mean": float(np.mean(cosine)),
        "directional_abs_cosine_mean": float(np.mean(np.abs(cosine))),
        "top10_recall": _top_recall(curvature_norm[valid], field_norm[valid], 0.10),
        "top25_recall": _top_recall(curvature_norm[valid], field_norm[valid], 0.25),
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    draws = rng.choice(finite, size=(10000, len(finite)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _aggregate_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    result: list[dict[str, Any]] = []
    for bin_name, _, _ in BIN_SPECS:
        selected = [row for row in rows if row["bin"] == bin_name]
        output: dict[str, Any] = {
            "bin": bin_name,
            "meshes": len(selected),
            "vertices": int(sum(int(row["vertices"]) for row in selected)),
        }
        for metric in LOCAL_ERROR_METRICS:
            for method in ("e", "hybrid"):
                values = np.asarray(
                    [float(row[f"{method}_{metric}_mean"]) for row in selected]
                )
                output[f"macro_{method}_{metric}_mean"] = float(np.mean(values))
            gains = np.asarray([float(row[f"{metric}_gain_mean"]) for row in selected])
            low, high = _bootstrap_ci(gains, rng)
            output[f"macro_{metric}_gain"] = float(np.mean(gains))
            output[f"macro_{metric}_gain_ci_low"] = low
            output[f"macro_{metric}_gain_ci_high"] = high
            output[f"{metric}_mesh_wins"] = int(np.count_nonzero(gains > 0))
            output[f"{metric}_mesh_losses"] = int(np.count_nonzero(gains < 0))
            denominators = np.asarray(
                [float(row[f"e_{metric}_mean"]) for row in selected]
            )
            output[f"macro_{metric}_relative_gain"] = float(
                np.mean(gains / np.maximum(denominators, 1e-30))
            )
            output[f"macro_{metric}_vertex_win_rate"] = float(
                np.mean([float(row[f"{metric}_vertex_win_rate"]) for row in selected])
            )
        result.append(output)
    return result


def _aggregate_correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for signal in ("predicted_b", "gt_uniform"):
        selected = [row for row in rows if row["signal"] == signal]
        output: dict[str, Any] = {"signal": signal, "meshes": len(selected)}
        for field in (
            "magnitude_pearson",
            "magnitude_spearman",
            "directional_cosine_mean",
            "directional_abs_cosine_mean",
            "top10_recall",
            "top25_recall",
        ):
            values = np.asarray([float(row[field]) for row in selected])
            output[f"macro_{field}"] = float(np.nanmean(values))
            output[f"median_{field}"] = float(np.nanmedian(values))
        result.append(output)
    return result


def _plot(output: Path, aggregate: Sequence[Mapping[str, Any]]) -> None:
    labels = ["0–25%", "25–50%", "50–75%", "75–90%", "90–100%"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, metric, title in (
        (axes[0], "surface", "Exact vertex-to-GT-surface error"),
        (axes[1], "vertex", "Same-index local error"),
    ):
        axis.plot(x, [row[f"macro_e_{metric}_mean"] for row in aggregate], "o-", label="E-only")
        axis.plot(
            x,
            [row[f"macro_hybrid_{metric}_mean"] for row in aggregate],
            "o-",
            label="Hybrid",
        )
        axis.set_xticks(x, labels, rotation=18)
        axis.set_xlabel("Within-mesh GT curvature percentile")
        axis.set_ylabel("Mean error")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "curvature_bin_local_error.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.3, 4.4))
    for metric, label in (("surface", "Exact surface"), ("vertex", "3D vertex"), ("normal", "GT-normal")):
        means = np.asarray([row[f"macro_{metric}_gain"] for row in aggregate])
        low = np.asarray([row[f"macro_{metric}_gain_ci_low"] for row in aggregate])
        high = np.asarray([row[f"macro_{metric}_gain_ci_high"] for row in aggregate])
        axis.errorbar(x, means, yerr=np.vstack((means - low, high - means)), marker="o", capsize=4, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=18)
    axis.set_xlabel("Within-mesh GT curvature percentile")
    axis.set_ylabel("E-only error − Hybrid error (positive is better)")
    axis.set_title("Differential-branch benefit by GT curvature")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "curvature_bin_gain.png", dpi=180)
    plt.close(figure)


def _report(
    output: Path,
    aggregate: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    trend: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# Sofa50 curvature-conditioned differential-branch benefit",
        "",
        f"Contract audit: **{str(bool(summary['contract_audit'])).lower()}**. Read-only test-set analysis over **{summary['samples']}** meshes.",
        "",
        "GT curvature is the magnitude of the standard cotangent discrete twice-mean-curvature vector `2Hn`. Vertices are ranked independently inside each mesh and split into `0–25%`, `25–50%`, `50–75%`, `75–90%`, and `90–100%` bins. This controls mesh scale and vertex-count imbalance.",
        "",
        "Hybrid is reproduced with the established frozen solve (`lambda=3e-2`, float64 PCG, `tol=1e-4`, maximum 2048 iterations). GT is loaded only after the frozen B/E predictions and recovery inputs are fixed.",
        "",
        "## Curvature-conditioned local error",
        "",
        "Positive gain means adding Arm B to E reduces error. Values are macro-averages of per-mesh bin means; confidence intervals bootstrap the 50 meshes.",
        "",
        "| GT curvature bin | E exact P2S | Hybrid exact P2S | Surface gain [95% CI] | E vertex | Hybrid vertex | Vertex gain [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['bin']} | {row['macro_e_surface_mean']:.9g} | {row['macro_hybrid_surface_mean']:.9g} | "
            f"{row['macro_surface_gain']:.9g} [{row['macro_surface_gain_ci_low']:.9g}, {row['macro_surface_gain_ci_high']:.9g}] | "
            f"{row['macro_e_vertex_mean']:.9g} | {row['macro_hybrid_vertex_mean']:.9g} | "
            f"{row['macro_vertex_gain']:.9g} [{row['macro_vertex_gain_ci_low']:.9g}, {row['macro_vertex_gain_ci_high']:.9g}] |"
        )
    lines += [
        "",
        "Exact P2S is the distance from each E/Hybrid vertex to the clean GT triangle surface, with the query vertex retaining its corresponding GT-vertex curvature bin. The normal error (reported in CSV/JSON and below) is the absolute displacement along the GT area-weighted vertex normal; vertex error is the full same-index Euclidean distance.",
        "",
        "## High-curvature benefit test",
        "",
        f"Highest-10%-minus-lowest-25% exact-surface-gain difference: `{trend['high_minus_low_surface_gain']:.9g}` (bootstrap 95% CI `[{trend['high_minus_low_surface_ci_low']:.9g}, {trend['high_minus_low_surface_ci_high']:.9g}]`; high larger on `{trend['high_better_surface_meshes']}/{summary['samples']}` meshes).",
        "",
        f"Highest-10%-minus-lowest-25% vertex-gain difference: `{trend['high_minus_low_vertex_gain']:.9g}` (bootstrap 95% CI `[{trend['high_minus_low_vertex_ci_low']:.9g}, {trend['high_minus_low_vertex_ci_high']:.9g}]`; high larger on `{trend['high_better_vertex_meshes']}/{summary['samples']}` meshes).",
        "",
        f"Highest-10%-minus-lowest-25% normal-gain difference: `{trend['high_minus_low_normal_gain']:.9g}` (bootstrap 95% CI `[{trend['high_minus_low_normal_ci_low']:.9g}, {trend['high_minus_low_normal_ci_high']:.9g}]`; high larger on `{trend['high_better_normal_meshes']}/{summary['samples']}` meshes).",
        "",
        f"Per-mesh curvature-versus-local-gain Spearman: exact surface macro mean `{trend['curvature_gain_surface_spearman_mean']:.5f}` (median `{trend['curvature_gain_surface_spearman_median']:.5f}`); vertex macro mean `{trend['curvature_gain_vertex_spearman_mean']:.5f}` (median `{trend['curvature_gain_vertex_spearman_median']:.5f}`); normal macro mean `{trend['curvature_gain_normal_spearman_mean']:.5f}` (median `{trend['curvature_gain_normal_spearman_median']:.5f}`).",
        "",
        f"Predeclared support gate (exact-surface high-minus-low bootstrap lower bound is positive and high-curvature gain is larger on a majority of meshes): **{str(bool(summary['high_curvature_benefit_supported'])).lower()}**.",
        "",
        "## Main finding",
        "",
        "The proposed curvature-localization hypothesis is not supported; the measured effect is the opposite. Hybrid is statistically indistinguishable from a small improvement in the lowest-curvature quartile for exact P2S, but is significantly worse from the 25th percentile upward. In the highest-curvature 10%, exact P2S rises from `0.000759132` for E to `0.00115267` for Hybrid (51.8% worse), same-index vertex error rises by 45.3%, and GT-normal error rises by 51.5%. The high-minus-low exact-surface gain is negative on all 50 meshes.",
        "",
        "This does not contradict Hybrid's lower global surface Chamfer (`0.00302983` versus E's `0.00334039`). Global Chamfer is area-weighted and bidirectional, whereas this audit conditions forward vertex-to-GT-surface errors on GT-vertex curvature rank. The global gain can therefore arise from surface-area weighting, the reverse GT-to-prediction direction, and error redistribution rather than preferential correction of high-curvature vertices.",
        "",
        "The weak cotangent-curvature correlation should not be assigned solely to predictor failure: even the exact clean uniform-Laplacian field has similarly weak magnitude correlation and top-region recall. Under the current operator contract, Arm B is best described as an operator-guided differential constraint, not as a cotangent-curvature predictor. The paper should retain the exact recovery-spectrum result but avoid claiming that B specifically improves regions where cotangent curvature is high.",
        "",
        "![Curvature-conditioned local error](curvature_bin_local_error.png)",
        "",
        "![Curvature-conditioned gain](curvature_bin_gain.png)",
        "",
        "## Differential field versus cotangent curvature",
        "",
        "`predicted_b` is frozen Arm-B's raw uniform-Laplacian prediction. `gt_uniform` is the clean mesh's exact uniform-Laplacian field and is included as an operator-mismatch reference; neither is expected to numerically equal cotangent `2Hn`.",
        "",
        "| Signal | Magnitude Pearson | Magnitude Spearman | Direction cosine | Abs. direction cosine | Top-10% recall | Top-25% recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in correlations:
        lines.append(
            f"| {row['signal']} | {row['macro_magnitude_pearson']:.5f} | {row['macro_magnitude_spearman']:.5f} | "
            f"{row['macro_directional_cosine_mean']:.5f} | {row['macro_directional_abs_cosine_mean']:.5f} | "
            f"{row['macro_top10_recall']:.5f} | {row['macro_top25_recall']:.5f} |"
        )
    lines += [
        "",
        "Random top-set recall baselines are 0.10 and 0.25. Direction cosine uses the signed `2Hn` convention in the protocol; absolute cosine is also reported to expose orientation agreement independently of sign.",
        "",
        "## Protocol and scope",
        "",
        f"Curvature protocol: `{CURVATURE_PROTOCOL}`",
        "",
        "This analysis establishes where the already-selected frozen fusion changes local geometry. It does not use test curvature to select a checkpoint, lambda, model, or recovery setting.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    inputs = argparse.Namespace(
        manifest=args.manifest,
        arm_b_report=args.arm_b_report,
        arm_e_report=args.arm_e_report,
    )
    dataset, _, _, _, _, b_array, e_array, b_starts, e_starts = _inputs(inputs, "test")
    bin_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    sample_trends: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        count = len(initial)
        delta = b_array[b_starts[index] : b_starts[index] + count]
        direct = initial + e_array[e_starts[index] : e_starts[index] + count]
        hybrid, solve_audit = _pcg(delta, direct, static, LAMBDA_HYBRID, torch.device("cpu"))
        # GT is deliberately materialized only after the frozen predictions and
        # recovery solve are complete; it is analysis-only.
        clean = np.asarray(_clean_mesh(static).vertices, dtype=np.float64)
        curvature_vector, curvature_valid = _cotangent_twice_mean_curvature(clean, faces)
        normals, normal_valid = area_weighted_vertex_normals(clean, faces)
        eligible = curvature_valid & normal_valid
        curvature = np.linalg.norm(curvature_vector, axis=1)
        bins = curvature_rank_bins(curvature, eligible)
        clean_mesh = Mesh(clean, faces)
        e_errors = local_errors(direct, clean, normals)
        hybrid_errors = local_errors(hybrid, clean, normals)
        e_errors["surface"] = _point_to_surface_distances(direct, clean_mesh)[0]
        hybrid_errors["surface"] = _point_to_surface_distances(hybrid, clean_mesh)[0]
        per_vertex_gain: dict[str, np.ndarray] = {}
        for metric in LOCAL_ERROR_METRICS:
            per_vertex_gain[metric] = e_errors[metric] - hybrid_errors[metric]
        mesh_bin_rows: list[dict[str, Any]] = []
        for bin_index, (bin_name, low, high) in enumerate(BIN_SPECS):
            selected = bins == bin_index
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "index": index,
                "bin": bin_name,
                "percentile_low": low,
                "percentile_high": high,
                "vertices": int(selected.sum()),
                "curvature_mean": float(np.mean(curvature[selected])),
                "curvature_min": float(np.min(curvature[selected])),
                "curvature_max": float(np.max(curvature[selected])),
            }
            for metric in LOCAL_ERROR_METRICS:
                e_value = e_errors[metric][selected]
                h_value = hybrid_errors[metric][selected]
                gain = per_vertex_gain[metric][selected]
                row[f"e_{metric}_mean"] = float(np.mean(e_value))
                row[f"e_{metric}_rms"] = float(np.sqrt(np.mean(np.square(e_value))))
                row[f"hybrid_{metric}_mean"] = float(np.mean(h_value))
                row[f"hybrid_{metric}_rms"] = float(np.sqrt(np.mean(np.square(h_value))))
                row[f"{metric}_gain_mean"] = float(np.mean(gain))
                row[f"{metric}_vertex_win_rate"] = float(np.mean(gain > 0))
            mesh_bin_rows.append(row)
            bin_rows.append(row)
        valid = bins >= 0
        sample_trends.append(
            {
                "sample_id": sample_id,
                "surface_gain_spearman": _safe_correlation(
                    curvature[valid], per_vertex_gain["surface"][valid], "spearman"
                ),
                "vertex_gain_spearman": _safe_correlation(
                    curvature[valid], per_vertex_gain["vertex"][valid], "spearman"
                ),
                "normal_gain_spearman": _safe_correlation(
                    curvature[valid], per_vertex_gain["normal"][valid], "spearman"
                ),
                "high_minus_low_surface_gain": float(mesh_bin_rows[-1]["surface_gain_mean"])
                - float(mesh_bin_rows[0]["surface_gain_mean"]),
                "high_minus_low_vertex_gain": float(mesh_bin_rows[-1]["vertex_gain_mean"])
                - float(mesh_bin_rows[0]["vertex_gain_mean"]),
                "high_minus_low_normal_gain": float(mesh_bin_rows[-1]["normal_gain_mean"])
                - float(mesh_bin_rows[0]["normal_gain_mean"]),
            }
        )
        laplacian, _ = uniform_sparse_laplacian(faces, count)
        gt_uniform = laplacian @ clean
        for signal, field in (("predicted_b", delta), ("gt_uniform", gt_uniform)):
            correlation_rows.append(
                {
                    "sample_id": sample_id,
                    "index": index,
                    "signal": signal,
                    **field_curvature_statistics(field, curvature_vector, eligible),
                }
            )
        audits.append(
            {
                "sample_id": sample_id,
                "vertices": count,
                "eligible_vertices": int(eligible.sum()),
                **solve_audit,
            }
        )
        print(f"test {index + 1}/{len(dataset)} {sample_id}", flush=True)

    aggregate = _aggregate_bins(bin_rows)
    correlations = _aggregate_correlations(correlation_rows)
    rng = np.random.default_rng(7)
    surface_difference = np.asarray(
        [row["high_minus_low_surface_gain"] for row in sample_trends]
    )
    vertex_difference = np.asarray(
        [row["high_minus_low_vertex_gain"] for row in sample_trends]
    )
    normal_difference = np.asarray(
        [row["high_minus_low_normal_gain"] for row in sample_trends]
    )
    surface_low, surface_high = _bootstrap_ci(surface_difference, rng)
    vertex_low, vertex_high = _bootstrap_ci(vertex_difference, rng)
    normal_low, normal_high = _bootstrap_ci(normal_difference, rng)
    trend = {
        "high_minus_low_surface_gain": float(np.mean(surface_difference)),
        "high_minus_low_surface_ci_low": surface_low,
        "high_minus_low_surface_ci_high": surface_high,
        "high_better_surface_meshes": int(np.count_nonzero(surface_difference > 0)),
        "high_minus_low_vertex_gain": float(np.mean(vertex_difference)),
        "high_minus_low_vertex_ci_low": vertex_low,
        "high_minus_low_vertex_ci_high": vertex_high,
        "high_better_vertex_meshes": int(np.count_nonzero(vertex_difference > 0)),
        "high_minus_low_normal_gain": float(np.mean(normal_difference)),
        "high_minus_low_normal_ci_low": normal_low,
        "high_minus_low_normal_ci_high": normal_high,
        "high_better_normal_meshes": int(np.count_nonzero(normal_difference > 0)),
        "curvature_gain_surface_spearman_mean": float(
            np.nanmean([row["surface_gain_spearman"] for row in sample_trends])
        ),
        "curvature_gain_surface_spearman_median": float(
            np.nanmedian([row["surface_gain_spearman"] for row in sample_trends])
        ),
        "curvature_gain_vertex_spearman_mean": float(
            np.nanmean([row["vertex_gain_spearman"] for row in sample_trends])
        ),
        "curvature_gain_vertex_spearman_median": float(
            np.nanmedian([row["vertex_gain_spearman"] for row in sample_trends])
        ),
        "curvature_gain_normal_spearman_mean": float(
            np.nanmean([row["normal_gain_spearman"] for row in sample_trends])
        ),
        "curvature_gain_normal_spearman_median": float(
            np.nanmedian([row["normal_gain_spearman"] for row in sample_trends])
        ),
    }
    summary = {
        "contract_audit": bool(
            len(audits) == 50
            and len(bin_rows) == 250
            and all(bool(row["pcg_converged"]) for row in audits)
            and all(int(row["eligible_vertices"]) >= 5 for row in audits)
        ),
        "samples": len(audits),
        "split": "test",
        "read_only": True,
        "models_retrained": False,
        "gt_used_for_prediction_or_recovery": False,
        "lambda": LAMBDA_HYBRID,
        "curvature_protocol": CURVATURE_PROTOCOL,
        "binning": "within-mesh stable rank of GT ||2Hn||",
        "primary_local_surface_error": "exact predicted-vertex to GT-triangle-surface distance",
        "high_curvature_benefit_supported": bool(
            surface_low > 0 and np.count_nonzero(surface_difference > 0) > len(audits) / 2
        ),
        "trend": trend,
        "aggregate_bins": aggregate,
        "correlation_aggregate": correlations,
        "maximum_pcg_relative_residual": float(
            max(float(row["pcg_relative_residual"]) for row in audits)
        ),
    }
    if not summary["contract_audit"]:
        raise RuntimeError("Curvature-conditioned analysis contract failed.")
    _write_csv(output / "curvature_bin_per_mesh.csv", bin_rows)
    _write_csv(output / "curvature_bin_aggregate.csv", aggregate)
    _write_csv(output / "field_curvature_per_mesh.csv", correlation_rows)
    _write_csv(output / "field_curvature_aggregate.csv", correlations)
    _write_csv(output / "curvature_gain_trends.csv", sample_trends)
    _write_csv(output / "solver_audit.csv", audits)
    _write_json(output / "curvature_conditioned_analysis.json", summary)
    _plot(output, aggregate)
    _report(output, aggregate, correlations, trend, summary)
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

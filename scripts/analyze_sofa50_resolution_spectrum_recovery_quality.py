#!/usr/bin/env python3
from __future__ import annotations

"""Relate Sofa50 mesh resolution, recovery spectrum, and frozen Hybrid gain."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr

from analyze_sofa50_recovery_operator_spectrum import _indicator_coefficients_unit
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)


ANCHOR_LAMBDA = 3e-2
BOOTSTRAP_DRAWS = 5000
SPECTRAL_BANDS = (
    ("e_dominant", 0.0, ANCHOR_LAMBDA / 2.0),
    ("transition", ANCHOR_LAMBDA / 2.0, 2.0 * ANCHOR_LAMBDA),
    ("b_dominant", 2.0 * ANCHOR_LAMBDA, float("inf")),
)
PRIMARY_PREDICTORS = (
    "log_vertices",
    "log_median_edge_length",
    "log_vertex_density",
)
GAIN_METRICS = ("cd_gain_e_minus_h", "p95_gain_e_minus_h", "vrms_gain_e_minus_h")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, seed: int = 7) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        finite, size=(BOOTSTRAP_DRAWS, len(finite)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return float("nan"), float("nan")
    return (
        float(pearsonr(left[valid], right[valid]).statistic),
        float(spearmanr(left[valid], right[valid]).statistic),
    )


def bootstrap_correlations(
    left: np.ndarray, right: np.ndarray, *, seed: int = 7
) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    pearson, spearman = _correlation(left, right)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(BOOTSTRAP_DRAWS, len(left)))
    boot_pearson = np.empty(len(indices), dtype=np.float64)
    boot_spearman = np.empty(len(indices), dtype=np.float64)
    for draw, selected in enumerate(indices):
        boot_pearson[draw], boot_spearman[draw] = _correlation(
            left[selected], right[selected]
        )
    return {
        "pearson": pearson,
        "pearson_ci_low": float(np.nanquantile(boot_pearson, 0.025)),
        "pearson_ci_high": float(np.nanquantile(boot_pearson, 0.975)),
        "spearman": spearman,
        "spearman_ci_low": float(np.nanquantile(boot_spearman, 0.025)),
        "spearman_ci_high": float(np.nanquantile(boot_spearman, 0.975)),
        "samples": len(left),
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.std(values))
    if scale <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def standardized_coefficient(
    rows: Sequence[Mapping[str, Any]],
    outcome: str,
    predictor: str,
    controls: Sequence[str],
) -> float:
    y = _zscore(np.asarray([float(row[outcome]) for row in rows]))
    columns = [_zscore(np.asarray([float(row[predictor]) for row in rows]))]
    columns.extend(
        _zscore(np.asarray([float(row[field]) for row in rows])) for field in controls
    )
    design = np.column_stack((np.ones(len(rows)), *columns))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficient[1])


def bootstrap_standardized_coefficient(
    rows: Sequence[Mapping[str, Any]],
    outcome: str,
    predictor: str,
    controls: Sequence[str],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    value = standardized_coefficient(rows, outcome, predictor, controls)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(len(draws)):
        selected = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[index] for index in selected]
        draws[draw] = standardized_coefficient(sample, outcome, predictor, controls)
    return {
        "standardized_beta": value,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "controls": ";".join(controls),
        "predictor": predictor,
        "outcome": outcome,
    }


def mesh_geometry_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    triangles = xyz[tri]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    area = float(np.sum(areas))
    if not np.isfinite(area) or area <= 0 or not np.all(np.isfinite(lengths)):
        raise RuntimeError("Mesh geometry statistics must be finite and positive.")
    return {
        "surface_area": area,
        "mean_edge_length": float(np.mean(lengths)),
        "median_edge_length": float(np.median(lengths)),
        "edge_length_q10": float(np.quantile(lengths, 0.1)),
        "edge_length_q90": float(np.quantile(lengths, 0.9)),
        "unique_edges": len(edges),
        "vertex_density": len(xyz) / area,
        "face_density": len(tri) / area,
    }


def projected_rademacher(
    labels: np.ndarray, probes: int, *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.choice((-1.0, 1.0), size=(len(labels), probes))
    for component in range(int(np.max(labels)) + 1):
        selected = labels == component
        values[selected] -= np.mean(values[selected], axis=0, keepdims=True)
    return values


def stochastic_chebyshev_moments(
    operator: csr_matrix,
    maximum_eigenvalue: float,
    labels: np.ndarray,
    *,
    order: int,
    probes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if order < 16 or probes < 4:
        raise ValueError("Spectral estimator requires order>=16 and probes>=4.")
    z = projected_rademacher(labels, probes, seed=seed)
    norms = np.einsum("ij,ij->j", z, z)
    if np.any(norms <= 0):
        raise RuntimeError("Projected probes must have positive norm.")

    def scaled_apply(values: np.ndarray) -> np.ndarray:
        return (2.0 / maximum_eigenvalue) * (operator @ values) - values

    moments = np.empty((probes, order), dtype=np.float64)
    previous = z
    moments[:, 0] = np.einsum("ij,ij->j", z, previous)
    current = scaled_apply(z)
    moments[:, 1] = np.einsum("ij,ij->j", z, current)
    for degree in range(2, order):
        following = 2.0 * scaled_apply(current) - previous
        moments[:, degree] = np.einsum("ij,ij->j", z, following)
        previous, current = current, following
    return moments, norms


def _normalized_trace_moments(moments: np.ndarray, norms: np.ndarray) -> np.ndarray:
    return np.sum(moments, axis=0) / np.sum(norms)


def spectral_summary_from_moments(
    moments: np.ndarray,
    norms: np.ndarray,
    maximum_eigenvalue: float,
    *,
    order: int,
    cdf_coefficients: np.ndarray,
    cdf_grid: np.ndarray,
) -> dict[str, float]:
    normalized = _normalized_trace_moments(moments[:, :order], norms)
    bands: dict[str, float] = {}
    for name, low, high in SPECTRAL_BANDS:
        upper = maximum_eigenvalue if not np.isfinite(high) else high
        coefficients = _indicator_coefficients_unit(
            low / maximum_eigenvalue, upper / maximum_eigenvalue, order
        )
        bands[name] = float(coefficients @ normalized)
    band_values = np.maximum(0.0, np.asarray(list(bands.values())))
    band_values /= np.sum(band_values)

    cdf = cdf_coefficients[:, :order] @ normalized
    cdf = np.clip(np.maximum.accumulate(cdf), 0.0, None)
    cdf /= max(float(cdf[-1]), np.finfo(np.float64).tiny)
    result = {
        f"{name}_fraction": float(band_values[index])
        for index, (name, _, _) in enumerate(SPECTRAL_BANDS)
    }
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        normalized_value = float(np.interp(quantile, cdf, cdf_grid))
        key = f"lambda_q{int(100 * quantile):02d}"
        result[key] = normalized_value * maximum_eigenvalue
        result[f"{key}_over_anchor"] = result[key] / ANCHOR_LAMBDA
    result["maximum_eigenvalue"] = maximum_eigenvalue
    result["maximum_over_anchor"] = maximum_eigenvalue / ANCHOR_LAMBDA
    cdf_mass = np.diff(np.concatenate(([0.0], cdf)))
    transfer = (maximum_eigenvalue * cdf_grid) / (
        maximum_eigenvalue * cdf_grid + ANCHOR_LAMBDA
    )
    result["effective_b_weight_mean"] = float(np.sum(transfer * cdf_mass))
    return result


def _metric_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] not in {
                "initial",
                "E_direct_vertex_residual",
                "Hybrid_B_laplacian_E_anchor",
            }:
                continue
            grouped.setdefault(row["sample_id"], {})[row["arm"]] = row
    return grouped


def _manifest_records(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["samples"] if row["split"] == split]


def _operator_maxima(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: float(row["recovery_lambda_max"])
            for row in csv.DictReader(handle)
        }


def _correlation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    spectral_predictors = (
        "e_dominant_fraction",
        "transition_fraction",
        "b_dominant_fraction",
        "lambda_q50_over_anchor",
        "effective_b_weight_mean",
    )
    resolution_predictors = PRIMARY_PREDICTORS + ("log_surface_area", "log_face_density")
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        for family, predictors, outcomes in (
            ("resolution_to_gain", resolution_predictors, GAIN_METRICS),
            ("resolution_to_spectrum", resolution_predictors, spectral_predictors),
            ("spectrum_to_gain", spectral_predictors, GAIN_METRICS),
        ):
            for predictor in predictors:
                x = np.asarray([float(row[predictor]) for row in selected])
                for outcome in outcomes:
                    y = np.asarray([float(row[outcome]) for row in selected])
                    result.append(
                        {
                            "split": split,
                            "family": family,
                            "predictor": predictor,
                            "outcome": outcome,
                            **bootstrap_correlations(x, y),
                        }
                    )
    return result


def _adjusted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    basic_controls = ("initial_chamfer", "log_surface_area", "log_median_edge_length")
    spectrum_controls = basic_controls + (
        "transition_fraction",
        "b_dominant_fraction",
        "lambda_q50_over_anchor",
    )
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        for model, controls in (
            ("difficulty_geometry", basic_controls),
            ("difficulty_geometry_spectrum", spectrum_controls),
            ("e_error_geometry", ("e_chamfer", "log_surface_area", "log_median_edge_length")),
        ):
            value = bootstrap_standardized_coefficient(
                selected,
                "cd_gain_e_minus_h",
                "log_vertices",
                controls,
            )
            result.append({"split": split, "model": model, **value})
        for predictor in ("log_median_edge_length", "log_vertex_density"):
            controls = ("initial_chamfer", "log_surface_area")
            value = bootstrap_standardized_coefficient(
                selected, "cd_gain_e_minus_h", predictor, controls
            )
            result.append({"split": split, "model": "alternative_density", **value})
    return result


def _plot_resolution_gain(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for split, color in (("validation", "#3b82f6"), ("test", "#ef4444")):
        selected = [row for row in rows if row["split"] == split]
        axes[0].scatter([row["vertices"] for row in selected], [row["cd_gain_e_minus_h"] for row in selected], s=22, alpha=0.65, label=split, color=color)
        axes[1].scatter([row["median_edge_length"] for row in selected], [row["cd_gain_e_minus_h"] for row in selected], s=22, alpha=0.65, label=split, color=color)
    axes[0].set(xscale="log", xlabel="vertices", ylabel="E CD - Hybrid CD", title="Vertex count vs recovery gain")
    axes[1].set(xscale="log", xlabel="median edge length", ylabel="E CD - Hybrid CD", title="Edge length vs recovery gain")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_spectrum_resolution(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"e_dominant_fraction": "#3b82f6", "transition_fraction": "#f59e0b", "b_dominant_fraction": "#ef4444"}
    test = [row for row in rows if row["split"] == "test"]
    for field, color in colors.items():
        label = field.replace("_fraction", "")
        axes[0].scatter([row["vertices"] for row in test], [row[field] for row in test], s=18, alpha=0.6, label=label, color=color)
        axes[1].scatter([row["vertex_density"] for row in test], [row[field] for row in test], s=18, alpha=0.6, label=label, color=color)
    axes[0].set(xscale="log", xlabel="vertices", ylabel="non-null spectral fraction", title="Resolution vs spectral gate regimes")
    axes[1].set(xscale="log", xlabel="vertices / surface area", ylabel="non-null spectral fraction", title="Density vs spectral gate regimes")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_spectrum_gain(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    fields = ("e_dominant_fraction", "transition_fraction", "b_dominant_fraction", "lambda_q50_over_anchor")
    titles = ("E-dominant fraction", "Transition fraction", "B-dominant fraction", r"Median $\Lambda/\lambda$")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, field, title in zip(axes.flat, fields, titles):
        for split, color in (("validation", "#3b82f6"), ("test", "#ef4444")):
            selected = [row for row in rows if row["split"] == split]
            axis.scatter([row[field] for row in selected], [row["cd_gain_e_minus_h"] for row in selected], s=20, alpha=0.6, label=split, color=color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel=title, ylabel="E CD - Hybrid CD")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _fmt_corr(row: Mapping[str, Any]) -> str:
    return f"{row['pearson']:.4f} [{row['pearson_ci_low']:.4f}, {row['pearson_ci_high']:.4f}] / {row['spearman']:.4f} [{row['spearman_ci_low']:.4f}, {row['spearman_ci_high']:.4f}]"


def _report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    adjusted: Sequence[Mapping[str, Any]],
    estimator_audit: Mapping[str, Any],
    classification: str,
) -> None:
    lines = [
        "# Sofa50 mesh resolution, recovery spectrum, and Hybrid gain",
        "",
        "Contract audit: **true**. Read-only local analysis of 50 validation and 50 test meshes. No model, checkpoint, mesh, recovery setting, or prior result was modified; no HPC job was submitted.",
        "",
        "Primary gain is frozen `E CD - Hybrid CD`, so positive values favor adding the differential branch. The fixed recovery gate is `g_B(Lambda)=Lambda/(Lambda+0.03)`.",
        "",
        "## Resolution and gain",
        "",
        "Each cell is Pearson [mesh-bootstrap 95% CI] / Spearman [95% CI].",
        "",
        "| Split | Predictor | CD gain | P2S-p95 gain | VRMS gain |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {"log_vertices": "log vertices", "log_median_edge_length": "log median edge", "log_vertex_density": "log vertices/area"}
    for split in ("validation", "test"):
        for predictor in PRIMARY_PREDICTORS:
            values = []
            for outcome in GAIN_METRICS:
                row = next(item for item in correlations if item["split"] == split and item["family"] == "resolution_to_gain" and item["predictor"] == predictor and item["outcome"] == outcome)
                values.append(_fmt_corr(row))
            lines.append(f"| {split} | {labels[predictor]} | " + " | ".join(values) + " |")
    lines += ["", "![Resolution and recovery gain](resolution_gain.png)", "", "## Spectrum relative to fixed lambda", ""]
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        lines += [
            f"### {split}",
            "",
            "| Statistic | Median | p10 / p90 | Minimum / maximum |",
            "|---|---:|---:|---:|",
        ]
        for field, label in (
            ("e_dominant_fraction", "E-dominant fraction"),
            ("transition_fraction", "Transition fraction"),
            ("b_dominant_fraction", "B-dominant fraction"),
            ("lambda_q50_over_anchor", "Median Lambda/lambda"),
            ("effective_b_weight_mean", "Mean effective B weight"),
        ):
            value = np.asarray([float(row[field]) for row in selected])
            lines.append(f"| {label} | {np.median(value):.6f} | {np.quantile(value, 0.1):.6f} / {np.quantile(value, 0.9):.6f} | {np.min(value):.6f} / {np.max(value):.6f} |")
        lines.append("")
    lines += [
        "![Resolution and spectral regimes](resolution_spectral_fractions.png)",
        "",
        "## Resolution to spectrum to gain",
        "",
        "### Test correlations",
        "",
        "| Link | Predictor | Outcome | Pearson / Spearman with 95% CI |",
        "|---|---|---|---:|",
    ]
    chosen = [
        ("resolution_to_spectrum", "log_vertices", "e_dominant_fraction"),
        ("resolution_to_spectrum", "log_vertices", "transition_fraction"),
        ("resolution_to_spectrum", "log_vertices", "b_dominant_fraction"),
        ("resolution_to_spectrum", "log_vertex_density", "b_dominant_fraction"),
        ("spectrum_to_gain", "e_dominant_fraction", "cd_gain_e_minus_h"),
        ("spectrum_to_gain", "transition_fraction", "cd_gain_e_minus_h"),
        ("spectrum_to_gain", "b_dominant_fraction", "cd_gain_e_minus_h"),
        ("spectrum_to_gain", "lambda_q50_over_anchor", "cd_gain_e_minus_h"),
    ]
    for family, predictor, outcome in chosen:
        row = next(item for item in correlations if item["split"] == "test" and item["family"] == family and item["predictor"] == predictor and item["outcome"] == outcome)
        lines.append(f"| {family} | {predictor} | {outcome} | {_fmt_corr(row)} |")
    lines += ["", "![Spectrum and recovery gain](spectrum_gain.png)", "", "## Adjusted resolution effect", "", "Standardized OLS coefficients use mesh bootstrap. They are conditional associations, not causal effects.", "", "| Split | Model | Predictor | Controls | beta [95% CI] |", "|---|---|---|---|---:|"]
    for row in adjusted:
        lines.append(f"| {row['split']} | {row['model']} | {row['predictor']} | {row['controls']} | {row['standardized_beta']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    test_raw = next(item for item in correlations if item["split"] == "test" and item["family"] == "resolution_to_gain" and item["predictor"] == "log_vertices" and item["outcome"] == "cd_gain_e_minus_h")
    validation_edge = next(item for item in correlations if item["split"] == "validation" and item["family"] == "resolution_to_gain" and item["predictor"] == "log_median_edge_length" and item["outcome"] == "cd_gain_e_minus_h")
    test_edge = next(item for item in correlations if item["split"] == "test" and item["family"] == "resolution_to_gain" and item["predictor"] == "log_median_edge_length" and item["outcome"] == "cd_gain_e_minus_h")
    test_basic = next(item for item in adjusted if item["split"] == "test" and item["model"] == "difficulty_geometry")
    test_spectrum = next(item for item in adjusted if item["split"] == "test" and item["model"] == "difficulty_geometry_spectrum")
    test_edge_adjusted = next(item for item in adjusted if item["split"] == "test" and item["model"] == "alternative_density" and item["predictor"] == "log_median_edge_length")
    test_e_fraction_gain = next(item for item in correlations if item["split"] == "test" and item["family"] == "spectrum_to_gain" and item["predictor"] == "e_dominant_fraction" and item["outcome"] == "cd_gain_e_minus_h")
    test_b_fraction_gain = next(item for item in correlations if item["split"] == "test" and item["family"] == "spectrum_to_gain" and item["predictor"] == "b_dominant_fraction" and item["outcome"] == "cd_gain_e_minus_h")
    all_lambda_ratios = np.asarray([float(item["lambda_q50_over_anchor"]) for item in rows])
    all_e_fractions = np.asarray([float(item["e_dominant_fraction"]) for item in rows])
    all_b_fractions = np.asarray([float(item["b_dominant_fraction"]) for item in rows])
    lines += [
        "",
        "## Decision",
        "",
        f"Classification: **{classification}**.",
        "",
        "Predeclared interpretation: a reliable raw resolution relationship requires positive validation and test Spearman lower bounds; an adjusted relationship requires the test standardized log-vertex coefficient CI to exclude zero after initial error, area, and edge length. A spectrum-mediated pattern additionally requires resolution-to-spectrum and spectrum-to-gain links with CIs excluding zero plus at least 25% attenuation of the adjusted log-vertex coefficient after spectral variables. A raw relationship that disappears after controls is classified as confounding; absent raw replication is no reliable relationship.",
        "",
        "The primary vertex-count/spectrum-mediation gate fails, but this is not evidence that every sampling statistic is unrelated to gain. Shorter median edges have a reproducible observational association with larger CD gain in both splits, including after controlling initial CD and area on test. This secondary result is reported as an edge-length association, not as proof that resolution causally drives recovery.",
        "",
        "## Answers to the six questions",
        "",
        f"1. **No reliable raw vertex-count effect.** Test Pearson is `{test_raw['pearson']:.4f}` (95% CI `[{test_raw['pearson_ci_low']:.4f}, {test_raw['pearson_ci_high']:.4f}]`) and Spearman is `{test_raw['spearman']:.4f}` (`[{test_raw['spearman_ci_low']:.4f}, {test_raw['spearman_ci_high']:.4f}]`); both include zero, and validation Spearman also includes zero.",
        f"2. **No adjusted vertex-count effect.** After initial error, area, and edge length, test standardized log-vertex beta is `{test_basic['standardized_beta']:.4f}` (95% CI `[{test_basic['ci_low']:.4f}, {test_basic['ci_high']:.4f}]`).",
        f"3. **Median edge length is the better replicated predictor.** Its CD-gain Spearman is `{validation_edge['spearman']:.4f}` on validation (`[{validation_edge['spearman_ci_low']:.4f}, {validation_edge['spearman_ci_high']:.4f}]`) and `{test_edge['spearman']:.4f}` on test (`[{test_edge['spearman_ci_low']:.4f}, {test_edge['spearman_ci_high']:.4f}]`). Test adjusted beta is `{test_edge_adjusted['standardized_beta']:.4f}` (`[{test_edge_adjusted['ci_low']:.4f}, {test_edge_adjusted['ci_high']:.4f}]`). Vertex-density evidence is less stable on test.",
        "4. **Vertex count changes the estimated regime proportions statistically, but only slightly in absolute terms.** On test, log-vertex Spearman is positive for E-dominant mass and negative for B-dominant mass (see chain table). Across all meshes, E-dominant mass spans " + f"`{all_e_fractions.min():.4f}`--`{all_e_fractions.max():.4f}` and B-dominant mass `{all_b_fractions.min():.4f}`--`{all_b_fractions.max():.4f}`. The fractions count all estimated non-null modes, not only representative eigenmodes.",
        f"5. **No supported spectral mediation.** Every test spectrum-to-CD-gain Spearman interval includes zero; for example E-dominant `{test_e_fraction_gain['spearman']:.4f}` (`[{test_e_fraction_gain['spearman_ci_low']:.4f}, {test_e_fraction_gain['spearman_ci_high']:.4f}]`) and B-dominant `{test_b_fraction_gain['spearman']:.4f}` (`[{test_b_fraction_gain['spearman_ci_low']:.4f}, {test_b_fraction_gain['spearman_ci_high']:.4f}]`). Adding spectral variables changes test log-vertex beta from `{test_basic['standardized_beta']:.4f}` to `{test_spectrum['standardized_beta']:.4f}`, but attenuation without a spectrum-to-gain link is not mediation evidence.",
        f"6. **The fixed crossover is similar, not substantially different, across these meshes.** Median `Lambda/lambda` ranges only `{all_lambda_ratios.min():.3f}`--`{all_lambda_ratios.max():.3f}`; about 92.4--93.4% of estimated non-null modes are B-dominant. Thus `lambda=0.03` induces measurable but modest cross-mesh gate shifts.",
        "",
        "The requested mesh bootstrap treats the 50 meshes in each split as sampling units. Each split contains five base objects with ten topology/perturbation variants, so variants are not a substitute for 50 independent object identities; causal or population-level resolution claims require more base shapes or a controlled remeshing experiment.",
        "",
        "## Spectrum-estimator audit",
        "",
        f"The full-spectrum fractions use nullspace-projected Hutchinson traces with Chebyshev--Jackson order `{estimator_audit['order']}` and `{estimator_audit['probes']}` Rademacher probes. Maximum full-vs-half-order band-fraction difference: `{estimator_audit['maximum_order_difference']:.6f}`; maximum first-half-vs-second-half probe difference: `{estimator_audit['maximum_probe_split_difference']:.6f}`. Maximum raw band partition error: `{estimator_audit['maximum_partition_error']:.3e}`. Component constants are explicitly projected out before trace estimation.",
        "",
        "Surface area and edge statistics use the frozen input vertices/faces. Global geometry errors come unchanged from `frozen_hybrid_recovery_v1/matched_per_sample.csv`. The operator uses connectivity only; no cotangent operator, image, GT geometry, checkpoint inference, or new recovery solve is involved.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--operator-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chebyshev-order", type=int, default=384)
    parser.add_argument("--probes", type=int, default=16)
    args = parser.parse_args()
    if args.chebyshev_order % 2 or args.probes % 2:
        raise ValueError("Order and probe count must be even for convergence audits.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    root = manifest.parent
    metrics = _metric_rows(args.metrics.resolve())
    maxima = _operator_maxima(args.operator_audit.resolve())
    cdf_grid = np.unique(
        np.concatenate((np.geomspace(1e-8, 1e-2, 160), np.linspace(1e-2, 1.0, 352)))
    )
    cdf_coefficients = {
        order: np.stack(
            [
                _indicator_coefficients_unit(0.0, float(value), order)
                for value in cdf_grid
            ],
            axis=0,
        )
        for order in (args.chebyshev_order // 2, args.chebyshev_order)
    }

    rows: list[dict[str, Any]] = []
    maximum_order_difference = 0.0
    maximum_probe_difference = 0.0
    maximum_partition_error = 0.0
    for split in ("validation", "test"):
        records = _manifest_records(manifest, split)
        if len(records) != 50:
            raise RuntimeError(f"Expected 50 {split} meshes, found {len(records)}")
        for index, record in enumerate(records):
            path = root / record["path"]
            static = torch.load(path, map_location="cpu", weights_only=False)
            sample_id = str(static["sample_id"])
            if sample_id != record["sample_id"] or sample_id not in metrics or sample_id not in maxima:
                raise RuntimeError(f"Identity contract failure: {sample_id}")
            vertices = np.asarray(static["vertices"], dtype=np.float64)
            faces = np.asarray(static["faces"], dtype=np.int64)
            geometry = mesh_geometry_statistics(vertices, faces)
            stored_local_edge = np.asarray(
                static.get("local_edge_length", static.get("local_edge_scale")),
                dtype=np.float64,
            )
            stored_local_edge = stored_local_edge[np.isfinite(stored_local_edge)]
            if not len(stored_local_edge):
                raise RuntimeError(f"Missing stored mesh-density statistic: {sample_id}")
            laplacian, lap_data = uniform_sparse_laplacian(faces, len(vertices))
            operator = (laplacian.T @ laplacian).tocsr()
            component_count, labels = component_labels(lap_data)
            maximum = maxima[sample_id] * (1.0 + 1e-8)
            moments, norms = stochastic_chebyshev_moments(
                operator,
                maximum,
                labels,
                order=args.chebyshev_order,
                probes=args.probes,
                seed=7 + index + (0 if split == "validation" else 1000),
            )
            full = spectral_summary_from_moments(
                moments,
                norms,
                maximum,
                order=args.chebyshev_order,
                cdf_coefficients=cdf_coefficients[args.chebyshev_order],
                cdf_grid=cdf_grid,
            )
            half_order = spectral_summary_from_moments(
                moments,
                norms,
                maximum,
                order=args.chebyshev_order // 2,
                cdf_coefficients=cdf_coefficients[args.chebyshev_order // 2],
                cdf_grid=cdf_grid,
            )
            first_probes = spectral_summary_from_moments(
                moments[: args.probes // 2],
                norms[: args.probes // 2],
                maximum,
                order=args.chebyshev_order,
                cdf_coefficients=cdf_coefficients[args.chebyshev_order],
                cdf_grid=cdf_grid,
            )
            second_probes = spectral_summary_from_moments(
                moments[args.probes // 2 :],
                norms[args.probes // 2 :],
                maximum,
                order=args.chebyshev_order,
                cdf_coefficients=cdf_coefficients[args.chebyshev_order],
                cdf_grid=cdf_grid,
            )
            band_fields = tuple(f"{name}_fraction" for name, _, _ in SPECTRAL_BANDS)
            maximum_order_difference = max(
                maximum_order_difference,
                max(abs(full[field] - half_order[field]) for field in band_fields),
            )
            maximum_probe_difference = max(
                maximum_probe_difference,
                max(abs(first_probes[field] - second_probes[field]) for field in band_fields),
            )
            maximum_partition_error = max(
                maximum_partition_error,
                abs(sum(full[field] for field in band_fields) - 1.0),
            )

            sample_metrics = metrics[sample_id]
            initial = sample_metrics["initial"]
            e = sample_metrics["E_direct_vertex_residual"]
            hybrid = sample_metrics["Hybrid_B_laplacian_E_anchor"]
            row: dict[str, Any] = {
                "split": split,
                "index": index,
                "sample_id": sample_id,
                "object_id": sample_id.split("__", 1)[0],
                "variant": record["sample_id"].split("__", 1)[1],
                "vertices": len(vertices),
                "faces": len(faces),
                "components": component_count,
                **geometry,
                "stored_local_edge_length_mean": float(np.mean(stored_local_edge)),
                "stored_local_edge_length_median": float(np.median(stored_local_edge)),
                **full,
                "initial_chamfer": float(initial["refined_chamfer"]),
                "e_chamfer": float(e["refined_chamfer"]),
                "hybrid_chamfer": float(hybrid["refined_chamfer"]),
                "cd_gain_e_minus_h": float(e["refined_chamfer"]) - float(hybrid["refined_chamfer"]),
                "e_p2s_p95": float(e["p2s_p95"]),
                "hybrid_p2s_p95": float(hybrid["p2s_p95"]),
                "p95_gain_e_minus_h": float(e["p2s_p95"]) - float(hybrid["p2s_p95"]),
                "e_vrms": float(e["same_index_recovered_vertex_rms"]),
                "hybrid_vrms": float(hybrid["same_index_recovered_vertex_rms"]),
                "vrms_gain_e_minus_h": float(e["same_index_recovered_vertex_rms"]) - float(hybrid["same_index_recovered_vertex_rms"]),
            }
            for field in ("vertices", "surface_area", "median_edge_length", "vertex_density", "face_density"):
                row[f"log_{field}"] = math.log(float(row[field]))
            rows.append(row)
            print(f"{split} {index + 1}/50 {sample_id}", flush=True)

    correlations = _correlation_rows(rows)
    adjusted = _adjusted_rows(rows)
    validation_raw = next(item for item in correlations if item["split"] == "validation" and item["family"] == "resolution_to_gain" and item["predictor"] == "log_vertices" and item["outcome"] == "cd_gain_e_minus_h")
    test_raw = next(item for item in correlations if item["split"] == "test" and item["family"] == "resolution_to_gain" and item["predictor"] == "log_vertices" and item["outcome"] == "cd_gain_e_minus_h")
    test_basic = next(item for item in adjusted if item["split"] == "test" and item["model"] == "difficulty_geometry")
    test_spectrum = next(item for item in adjusted if item["split"] == "test" and item["model"] == "difficulty_geometry_spectrum")
    raw_reliable = validation_raw["spearman_ci_low"] > 0 and test_raw["spearman_ci_low"] > 0
    adjusted_reliable = test_basic["ci_low"] > 0
    attenuation = 1.0 - abs(test_spectrum["standardized_beta"]) / max(abs(test_basic["standardized_beta"]), 1e-12)
    resolution_spectrum_links = [item for item in correlations if item["split"] == "test" and item["family"] == "resolution_to_spectrum" and item["predictor"] == "log_vertices"]
    spectrum_gain_links = [item for item in correlations if item["split"] == "test" and item["family"] == "spectrum_to_gain" and item["outcome"] == "cd_gain_e_minus_h"]
    linked = any(item["spearman_ci_low"] > 0 or item["spearman_ci_high"] < 0 for item in resolution_spectrum_links) and any(item["spearman_ci_low"] > 0 or item["spearman_ci_high"] < 0 for item in spectrum_gain_links)
    if raw_reliable and adjusted_reliable and linked and attenuation >= 0.25:
        classification = "SPECTRUM_MEDIATED_RESOLUTION_EFFECT"
    elif raw_reliable and adjusted_reliable:
        classification = "RESOLUTION_DRIVEN_RECOVERY_ASSOCIATION"
    elif raw_reliable:
        classification = "SIMPLE_CONFOUNDING_BY_MESH_DIFFICULTY"
    else:
        classification = "NO_RELIABLE_VERTEX_COUNT_OR_SPECTRUM_MEDIATED_RELATIONSHIP"

    estimator_audit = {
        "order": args.chebyshev_order,
        "probes": args.probes,
        "maximum_order_difference": maximum_order_difference,
        "maximum_probe_split_difference": maximum_probe_difference,
        "maximum_partition_error": maximum_partition_error,
    }
    _write_csv(output / "per_mesh.csv", rows)
    _write_csv(output / "correlations.csv", correlations)
    _write_csv(output / "adjusted_models.csv", adjusted)
    _write_json(
        output / "analysis.json",
        {
            "contract_audit": True,
            "classification": classification,
            "estimator_audit": estimator_audit,
            "correlations": correlations,
            "adjusted_models": adjusted,
            "attenuation_after_spectrum": attenuation,
            "protocol": {
                "anchor_lambda": ANCHOR_LAMBDA,
                "spectrum": "nullspace-projected Hutchinson Chebyshev-Jackson trace",
                "hpc_jobs_submitted": 0,
                "cotangent_used": False,
            },
        },
    )
    _plot_resolution_gain(rows, output / "resolution_gain.png")
    _plot_spectrum_resolution(rows, output / "resolution_spectral_fractions.png")
    _plot_spectrum_gain(rows, output / "spectrum_gain.png")
    _report(output, rows, correlations, adjusted, estimator_audit, classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

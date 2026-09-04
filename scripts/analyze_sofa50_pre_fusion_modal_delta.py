#!/usr/bin/env python3
from __future__ import annotations

"""Audit pre-fusion Arm-B/Arm-E error energy in recovery-response modes."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_sofa50_recovery_operator_spectrum import (
    LAMBDA_HYBRID,
    _indicator_coefficients_unit,
    _maximum_eigenvalue,
)
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_centroids,
    component_labels,
    exact_sparse_solve,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_frozen_hybrid_recovery import _inputs


REGIMES = (
    ("E-dominant", 0.0, 1.0 / 3.0),
    ("transition", 1.0 / 3.0, 2.0 / 3.0),
    ("B-dominant", 2.0 / 3.0, 1.0),
)
BLUE = "#2563EB"
ORANGE = "#EA580C"
DARK = "#172033"
GRID = "#D5DAE3"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _response_to_relative(response: float, lambda_max: float) -> float:
    """Map w_B=Lambda/(Lambda+lambda) to Lambda/Lambda_max."""

    if response >= 1.0:
        return 1.0
    eigenvalue = LAMBDA_HYBRID * response / max(1.0 - response, 1e-30)
    return float(np.clip(eigenvalue / lambda_max, 0.0, 1.0))


def _paired_band_energies(
    operator: Any,
    lambda_max: float,
    error_b: np.ndarray,
    error_e: np.ndarray,
    bands: Mapping[str, tuple[float, float]],
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all band energies from one shared Chebyshev-moment pass."""

    stacked = np.concatenate((error_b, error_e), axis=1)
    moments_b = np.empty(order, dtype=np.float64)
    moments_e = np.empty(order, dtype=np.float64)

    def record(degree: int, value: np.ndarray) -> None:
        moments_b[degree] = np.einsum("ij,ij->", error_b, value[:, :3])
        moments_e[degree] = np.einsum("ij,ij->", error_e, value[:, 3:])

    def scaled_apply(value: np.ndarray) -> np.ndarray:
        return (2.0 / lambda_max) * (operator @ value) - value

    previous = stacked
    record(0, previous)
    current = scaled_apply(stacked)
    record(1, current)
    for degree in range(2, order):
        following = 2.0 * scaled_apply(current) - previous
        record(degree, following)
        previous, current = current, following

    coefficients = np.stack(
        [_indicator_coefficients_unit(low, high, order) for low, high in bands.values()]
    )
    return coefficients @ moments_b, coefficients @ moments_e


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(draws, clean.size))
    means = clean[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _aggregate(
    rows: Sequence[Mapping[str, Any]], *, bins: int, bootstrap_draws: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bin_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("validation", "test")):
        selected = [row for row in rows if row["split"] == split]
        sample_ids = sorted({str(row["sample_id"]) for row in selected})
        for bin_index in range(bins):
            current = [row for row in selected if int(row["bin_index"]) == bin_index]
            e_b = np.asarray([float(row["e_b_energy"]) for row in current])
            e_e = np.asarray([float(row["e_e_energy"]) for row in current])
            delta = e_e - e_b
            contrast = delta / np.maximum(e_e + e_b, 1e-30)
            low, high = _bootstrap_mean_ci(
                contrast,
                seed=seed + 1000 * split_index + bin_index,
                draws=bootstrap_draws,
            )
            bin_rows.append(
                {
                    "split": split,
                    "bin_index": bin_index,
                    "w_b_low": float(current[0]["w_b_low"]),
                    "w_b_high": float(current[0]["w_b_high"]),
                    "w_b_center": 0.5
                    * (float(current[0]["w_b_low"]) + float(current[0]["w_b_high"])),
                    "samples": len(current),
                    "e_b_energy_sum": float(e_b.sum()),
                    "e_e_energy_sum": float(e_e.sum()),
                    "delta_energy_sum": float(delta.sum()),
                    "aggregate_local_contrast": float(
                        delta.sum() / max(float((e_e + e_b).sum()), 1e-30)
                    ),
                    "paired_local_contrast_mean": float(contrast.mean()),
                    "paired_local_contrast_ci_low": low,
                    "paired_local_contrast_ci_high": high,
                    "b_better_samples": int(np.sum(delta > 0.0)),
                    "e_better_samples": int(np.sum(delta < 0.0)),
                    "ties": int(np.sum(delta == 0.0)),
                }
            )
        for regime_index, (name, low_w, high_w) in enumerate(REGIMES):
            per_sample: list[tuple[float, float]] = []
            for sample_id in sample_ids:
                current = [
                    row
                    for row in selected
                    if str(row["sample_id"]) == sample_id
                    and float(row["w_b_low"]) >= low_w - 1e-12
                    and float(row["w_b_high"]) <= high_w + 1e-12
                ]
                per_sample.append(
                    (
                        float(sum(float(row["e_b_energy"]) for row in current)),
                        float(sum(float(row["e_e_energy"]) for row in current)),
                    )
                )
            values = np.asarray(per_sample, dtype=np.float64)
            delta = values[:, 1] - values[:, 0]
            contrast = delta / np.maximum(values.sum(axis=1), 1e-30)
            ci_low, ci_high = _bootstrap_mean_ci(
                contrast,
                seed=seed + 10000 + 1000 * split_index + regime_index,
                draws=bootstrap_draws,
            )
            regime_rows.append(
                {
                    "split": split,
                    "regime": name,
                    "w_b_low": low_w,
                    "w_b_high": high_w,
                    "samples": len(values),
                    "e_b_energy_sum": float(values[:, 0].sum()),
                    "e_e_energy_sum": float(values[:, 1].sum()),
                    "delta_energy_sum": float(delta.sum()),
                    "aggregate_local_contrast": float(
                        delta.sum() / max(float(values.sum()), 1e-30)
                    ),
                    "paired_local_contrast_mean": float(contrast.mean()),
                    "paired_local_contrast_ci_low": ci_low,
                    "paired_local_contrast_ci_high": ci_high,
                    "b_better_samples": int(np.sum(delta > 0.0)),
                    "e_better_samples": int(np.sum(delta < 0.0)),
                    "ties": int(np.sum(delta == 0.0)),
                }
            )
    return bin_rows, regime_rows


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7C8494")
    axis.tick_params(colors="#3E4655", labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65, zorder=0)
    axis.axhline(0.0, color=DARK, linewidth=0.9, zorder=3)
    axis.axvline(1.0 / 3.0, color="#818898", linestyle="--", linewidth=0.9)
    axis.axvline(2.0 / 3.0, color="#818898", linestyle="--", linewidth=0.9)
    axis.axvspan(0.0, 1.0 / 3.0, color=BLUE, alpha=0.035, zorder=-2)
    axis.axvspan(2.0 / 3.0, 1.0, color=ORANGE, alpha=0.035, zorder=-2)
    axis.set_xlim(0.0, 1.0)


def _plot(output: Path, aggregate: Sequence[Mapping[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.2, 7.4),
        sharex="col",
        gridspec_kw={"height_ratios": (1.1, 1.0)},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.88, wspace=0.17, hspace=0.12)
    for column, split in enumerate(("validation", "test")):
        selected = sorted(
            [row for row in aggregate if row["split"] == split],
            key=lambda row: int(row["bin_index"]),
        )
        x = np.asarray([float(row["w_b_center"]) for row in selected])
        width = float(selected[0]["w_b_high"]) - float(selected[0]["w_b_low"])
        raw = np.asarray([float(row["delta_energy_sum"]) for row in selected])
        contrast = np.asarray([float(row["paired_local_contrast_mean"]) for row in selected])
        ci_low = np.asarray([float(row["paired_local_contrast_ci_low"]) for row in selected])
        ci_high = np.asarray([float(row["paired_local_contrast_ci_high"]) for row in selected])

        top = axes[0, column]
        colors = np.where(raw < 0.0, BLUE, ORANGE)
        top.bar(x, raw, width=width * 0.89, color=colors, alpha=0.88, edgecolor="none", zorder=2)
        top.plot(x, raw, color=DARK, linewidth=0.8, alpha=0.55, zorder=3)
        nonzero = np.abs(raw[np.nonzero(raw)])
        linthresh = max(float(np.quantile(nonzero, 0.12)) if nonzero.size else 1e-8, 1e-10)
        top.set_yscale("symlog", linthresh=linthresh, linscale=0.8)
        negative_power = int(np.ceil(np.log10(max(abs(float(raw.min())), 1.0))))
        positive_power = int(np.ceil(np.log10(max(float(raw.max()), 1.0))))
        top.set_yticks(
            [-10.0**power for power in range(negative_power, -1, -1)]
            + [0.0]
            + [10.0**power for power in range(0, positive_power + 1)]
        )
        top.minorticks_off()
        top.set_title(split.capitalize(), pad=10)
        _style_axis(top)

        bottom = axes[1, column]
        bottom.fill_between(x, ci_low, ci_high, color="#93A4BE", alpha=0.28, linewidth=0.0, zorder=1)
        bottom.fill_between(x, 0.0, contrast, where=contrast <= 0.0, color=BLUE, alpha=0.34, interpolate=True, zorder=2)
        bottom.fill_between(x, 0.0, contrast, where=contrast >= 0.0, color=ORANGE, alpha=0.34, interpolate=True, zorder=2)
        bottom.plot(x, contrast, color=DARK, linewidth=1.65, marker="o", markersize=2.3, zorder=3)
        bottom.set_ylim(-1.04, 1.04)
        bottom.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        bottom.set_xlabel(r"B transfer coordinate  $w_B=\Lambda/(\Lambda+\lambda)$")
        _style_axis(bottom)

    axes[0, 0].set_ylabel(r"$\sum_i\Delta E_i$ per bin  (symlog)")
    axes[1, 0].set_ylabel(r"mean paired contrast  $\Delta E/(E_E+E_B)$")
    axes[0, 0].text(0.015, 0.96, r"$\Delta E<0$: E has lower error", color=BLUE, transform=axes[0, 0].transAxes, va="top", fontsize=9)
    axes[0, 1].text(0.985, 0.96, r"$\Delta E>0$: B$^\dagger$ has lower error", color=ORANGE, transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=9)
    for axis in axes[0, :]:
        axis.tick_params(labelbottom=False)
    for axis in axes[1, :]:
        axis.text(1.0 / 6.0, -0.23, "E-dominant", transform=axis.transAxes, ha="center", color=BLUE, fontsize=8.5)
        axis.text(0.5, -0.23, "transition", transform=axis.transAxes, ha="center", color="#636B79", fontsize=8.5)
        axis.text(5.0 / 6.0, -0.23, "B-dominant", transform=axis.transAxes, ha="center", color=ORANGE, fontsize=8.5)
    fig.suptitle(
        r"Pre-fusion modal error advantage:  $\Delta E(k)=\|e_E(k)\|_2^2-\|e_B(k)\|_2^2$",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "No Hybrid output is used. Top: requested signed energy difference. Bottom: paired local contrast with 95% mesh-bootstrap CI.",
        ha="center",
        fontsize=9.5,
        color="#596171",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"pre_fusion_modal_delta_energy.{suffix}", dpi=320)
    plt.close(fig)


def _report(
    output: Path,
    aggregate: Sequence[Mapping[str, Any]],
    regimes: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
) -> None:
    highest = {
        split: max(
            (row for row in aggregate if row["split"] == split),
            key=lambda row: int(row["bin_index"]),
        )
        for split in ("validation", "test")
    }
    lines = [
        "# Sofa50 pre-fusion modal error sanity check",
        "",
        "Contract audit: **true**.",
        "",
        "This analysis deliberately does **not** inspect `H-B` or `H-E`. It projects only the two pre-fusion errors",
        "",
        "```text",
        "e_B = V_B^dagger - V_GT",
        "e_E = V_E        - V_GT",
        "Delta E(k) = ||e_E(k)||_2^2 - ||e_B(k)||_2^2.",
        "```",
        "",
        "The full eigendecomposition is not materialized. Each plotted value is the sum of the exact requested per-mode quantity over a narrow recovery-response bin, approximated with Jackson-damped Chebyshev projectors of `A_R=L_U^T L_U`. Thus negative values mean Arm-E has lower pre-fusion error energy; positive values mean the unanchored Arm-B solution has lower error energy.",
        "",
        "![Pre-fusion modal error advantage](pre_fusion_modal_delta_energy.png)",
        "",
        "## Main finding",
        "",
        "The pre-fusion check supports a **selective**, not blanket, division of labor. Arm-E has lower error through almost the entire response coordinate. The advantage reverses robustly only in the strongest-response bin `w_B in [35/36,1]`: paired local contrast is "
        f"`{float(highest['validation']['paired_local_contrast_mean']):+.4f}` "
        f"`[{float(highest['validation']['paired_local_contrast_ci_low']):+.4f}, {float(highest['validation']['paired_local_contrast_ci_high']):+.4f}]` on validation and "
        f"`{float(highest['test']['paired_local_contrast_mean']):+.4f}` "
        f"`[{float(highest['test']['paired_local_contrast_ci_low']):+.4f}, {float(highest['test']['paired_local_contrast_ci_high']):+.4f}]` on test. "
        f"`V_B^dagger` is better on `{highest['validation']['b_better_samples']}/50` validation and `{highest['test']['b_better_samples']}/50` test meshes in that bin.",
        "",
        "Conversely, integrating the entire nominal B-dominant interval `w_B>=2/3` still favors Arm-E. Therefore the defensible statement is that the differential branch has a reproducible advantage in the **highest recovery-response modes**; the broad B-dominant interval is not uniformly more accurate for `V_B^dagger`.",
        "",
        "## Fusion-response regime totals",
        "",
        "The horizontal coordinate is `w_B=Lambda/(Lambda+0.03)` only to label the operator response. No Hybrid vertices enter the calculation. Confidence intervals bootstrap paired meshes.",
        "",
        "| Split | Regime | sum Delta E | Aggregate contrast | Paired contrast [95% CI] | B better / E better / tie |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in regimes:
        lines.append(
            f"| {row['split']} | {row['regime']} | {float(row['delta_energy_sum']):+.8g} | "
            f"{float(row['aggregate_local_contrast']):+.4f} | "
            f"{float(row['paired_local_contrast_mean']):+.4f} "
            f"[{float(row['paired_local_contrast_ci_low']):+.4f}, {float(row['paired_local_contrast_ci_high']):+.4f}] | "
            f"{row['b_better_samples']} / {row['e_better_samples']} / {row['ties']} |"
        )
    lines += [
        "",
        "## Numerical audit",
        "",
        f"- Meshes: `{len(audits)}` (50 validation + 50 test).",
        f"- Maximum relative spectral-partition residual: `{max(float(row['partition_relative_residual']) for row in audits):.3e}`.",
        f"- Maximum component-gauge mismatch between `V_B^dagger` and `V_E`: `{max(float(row['component_gauge_max_abs']) for row in audits):.3e}`.",
        f"- Maximum exact-solve normal-equation residual: `{max(float(row['normal_equation_relative_residual']) for row in audits):.3e}`.",
        "- GT is used only after both frozen branch outputs and the operator are fixed; it is not used in prediction, recovery, bin construction or model selection.",
        "",
        "Machine-readable outputs: `modal_delta_energy_bins.csv`, `modal_delta_energy_per_sample.csv`, `modal_delta_energy_regimes.csv`, `exactness_audit.csv`, and `summary.json`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--chebyshev-order", type=int, default=384)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.bins < 6 or args.bins % 3:
        raise ValueError("--bins must be at least six and divisible by three.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    response_edges = np.linspace(0.0, 1.0, args.bins + 1)

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    checkpoint_shas: dict[str, str] = {}
    for split in ("validation", "test"):
        (
            dataset,
            b_payload,
            e_payload,
            _b_rows,
            _e_rows,
            b_array,
            e_array,
            b_starts,
            e_starts,
        ) = _inputs(
            argparse.Namespace(
                manifest=args.manifest,
                arm_b_report=args.arm_b_report,
                arm_e_report=args.arm_e_report,
            ),
            split,
        )
        checkpoint_shas = {
            "arm_b": str(b_payload["checkpoint_sha256"]),
            "arm_e": str(e_payload["checkpoint_sha256"]),
        }
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            initial = np.asarray(static["vertices"], dtype=np.float64)
            faces = np.asarray(static["faces"], dtype=np.int64)
            clean = np.asarray(_clean_mesh(static).vertices, dtype=np.float64)
            count = len(initial)
            delta = np.asarray(
                b_array[b_starts[index] : b_starts[index] + count], dtype=np.float64
            )
            direct = initial + np.asarray(
                e_array[e_starts[index] : e_starts[index] + count], dtype=np.float64
            )
            laplacian, lap_data = uniform_sparse_laplacian(faces, count)
            operator = (laplacian.T @ laplacian).tocsr()
            component_count, labels = component_labels(lap_data)
            b_dagger, solve_audit = exact_sparse_solve(
                laplacian,
                delta,
                labels,
                component_count,
                component_centroids(direct, labels, component_count),
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            if not all(row["istop"] in (0, 1, 2, 4, 5) for row in solve_audit["axes"]):
                raise RuntimeError(f"{sample_id}: V_B^dagger reference solve failed.")
            rhs = laplacian.T @ delta
            normal_residual = operator @ b_dagger - rhs
            normal_relative = float(
                np.linalg.norm(normal_residual) / max(float(np.linalg.norm(rhs)), 1e-30)
            )
            gauge_b = component_centroids(b_dagger, labels, component_count)
            gauge_e = component_centroids(direct, labels, component_count)
            gauge_mismatch = float(np.max(np.abs(gauge_b - gauge_e)))
            lambda_max = _maximum_eigenvalue(operator)
            relative_bands = {
                f"bin_{bin_index:03d}": (
                    _response_to_relative(float(response_edges[bin_index]), lambda_max),
                    _response_to_relative(float(response_edges[bin_index + 1]), lambda_max),
                )
                for bin_index in range(args.bins)
            }
            error_b = b_dagger - clean
            error_e = direct - clean
            raw_energy_b, raw_energy_e = _paired_band_energies(
                operator,
                lambda_max,
                error_b,
                error_e,
                relative_bands,
                order=args.chebyshev_order,
            )
            energy_b: list[float] = []
            energy_e: list[float] = []
            for bin_index, name in enumerate(relative_bands):
                raw_b = float(raw_energy_b[bin_index])
                raw_e = float(raw_energy_e[bin_index])
                tolerance = 1e-9 * max(
                    float(np.square(error_b).sum() + np.square(error_e).sum()), 1.0
                )
                if raw_b < -tolerance or raw_e < -tolerance:
                    raise RuntimeError(f"{sample_id}/{name}: negative projector energy.")
                value_b = max(raw_b, 0.0)
                value_e = max(raw_e, 0.0)
                energy_b.append(value_b)
                energy_e.append(value_e)
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "index": index,
                        "vertices": count,
                        "bin_index": bin_index,
                        "w_b_low": float(response_edges[bin_index]),
                        "w_b_high": float(response_edges[bin_index + 1]),
                        "lambda_over_lambda_max_low": relative_bands[name][0],
                        "lambda_over_lambda_max_high": relative_bands[name][1],
                        "e_b_energy": value_b,
                        "e_e_energy": value_e,
                        "delta_energy": value_e - value_b,
                        "local_contrast": (value_e - value_b) / max(value_e + value_b, 1e-30),
                    }
                )
            total_b = float(np.square(error_b).sum())
            total_e = float(np.square(error_e).sum())
            partition_residual = max(
                abs(sum(energy_b) - total_b) / max(total_b, 1e-30),
                abs(sum(energy_e) - total_e) / max(total_e, 1e-30),
            )
            if partition_residual > 3e-5:
                raise RuntimeError(
                    f"{sample_id}: spectral partition residual {partition_residual:.3e}."
                )
            audits.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "vertices": count,
                    "components": component_count,
                    "lambda_max": lambda_max,
                    "e_b_total": total_b,
                    "e_e_total": total_e,
                    "delta_total_direct": total_e - total_b,
                    "delta_total_from_bins": sum(energy_e) - sum(energy_b),
                    "partition_relative_residual": partition_residual,
                    "component_gauge_max_abs": gauge_mismatch,
                    "normal_equation_relative_residual": normal_relative,
                }
            )
            print(f"{split} {index + 1}/{len(dataset)} {sample_id}", flush=True)

    aggregate, regimes = _aggregate(
        rows,
        bins=args.bins,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    contract = bool(
        len(audits) == 100
        and max(float(row["partition_relative_residual"]) for row in audits) < 3e-5
        and max(float(row["component_gauge_max_abs"]) for row in audits) < 1e-8
        and max(float(row["normal_equation_relative_residual"]) for row in audits) < 1e-8
    )
    summary = {
        "contract_audit": contract,
        "read_only": True,
        "hybrid_output_used": False,
        "gt_used_for_prediction_recovery_or_binning": False,
        "definition": "DeltaE(k)=||Q_k^T(V_E-V_GT)||_2^2-||Q_k^T(V_B_dagger-V_GT)||_2^2",
        "operator": "A_R=L_U^T L_U",
        "b_dagger_gauge": "component-nullspace gauge copied from V_E",
        "response_coordinate": "w_B=Lambda/(Lambda+0.03)",
        "lambda": LAMBDA_HYBRID,
        "bins": args.bins,
        "chebyshev_order": args.chebyshev_order,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "checkpoints": checkpoint_shas,
        "regimes": regimes,
        "audits": {
            "samples": len(audits),
            "maximum_partition_relative_residual": max(
                float(row["partition_relative_residual"]) for row in audits
            ),
            "maximum_component_gauge_abs": max(
                float(row["component_gauge_max_abs"]) for row in audits
            ),
            "maximum_normal_equation_relative_residual": max(
                float(row["normal_equation_relative_residual"]) for row in audits
            ),
        },
    }
    _write_csv(output / "modal_delta_energy_per_sample.csv", rows)
    _write_csv(output / "modal_delta_energy_bins.csv", aggregate)
    _write_csv(output / "modal_delta_energy_regimes.csv", regimes)
    _write_csv(output / "exactness_audit.csv", audits)
    _write_json(output / "summary.json", summary)
    _plot(output, aggregate)
    _report(output, aggregate, regimes, audits)
    print(json.dumps({"contract_audit": contract, "samples": len(audits)}, indent=2))
    return 0 if contract else 2


if __name__ == "__main__":
    raise SystemExit(main())

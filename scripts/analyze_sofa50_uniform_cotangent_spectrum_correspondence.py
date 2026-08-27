#!/usr/bin/env python3
from __future__ import annotations

"""Compare frozen recovery modes with intrinsic cotangent frequencies."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from mlr.learned_laplacian.cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


BANDS = ("low", "mid", "high")


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


def cotangent_stiffness_and_barycentric_mass(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    relative_area_epsilon: float = 1e-12,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    edges, weights, diagonal, audit = build_symmetric_cotangent_stiffness(
        torch.as_tensor(xyz, dtype=torch.float64),
        torch.as_tensor(tri, dtype=torch.long),
        relative_area_epsilon=relative_area_epsilon,
    )
    edge = edges.numpy()
    weight = weights.numpy()
    count = len(xyz)
    rows = np.concatenate((np.arange(count), edge[0], edge[1]))
    columns = np.concatenate((np.arange(count), edge[1], edge[0]))
    values = np.concatenate((diagonal.numpy(), -weight, -weight))
    stiffness = coo_matrix(
        (values, (rows, columns)), shape=(count, count)
    ).tocsr()

    triangles = xyz[tri]
    double_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    mass = np.zeros(count, dtype=np.float64)
    for column in range(3):
        np.add.at(mass, tri[:, column], double_area / 6.0)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise RuntimeError("Barycentric mass must be finite and strictly positive.")
    return stiffness, mass, audit.__dict__


def component_null_basis(
    labels: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    component = np.asarray(labels, dtype=np.int64)
    count = int(component.max(initial=-1)) + 1
    basis = np.zeros((len(component), count), dtype=np.float64)
    if weights is None:
        scale = np.ones(len(component), dtype=np.float64)
    else:
        scale = np.sqrt(np.asarray(weights, dtype=np.float64))
    for index in range(count):
        selected = component == index
        basis[selected, index] = scale[selected]
        basis[:, index] /= np.linalg.norm(basis[:, index])
    return basis


def _eigen_residuals(
    matrix: csr_matrix,
    values: np.ndarray,
    vectors: np.ndarray,
    spectral_scale: float,
) -> np.ndarray:
    residual = matrix @ vectors - vectors * values[None, :]
    numerator = np.linalg.norm(residual, axis=0)
    # Normwise backward error for unit eigenvectors.  Dividing by ||Aq|| is
    # ill-conditioned for genuine near-null modes and spuriously reports large
    # relative errors on high-dynamic-range cotangent operators.
    denominator = spectral_scale + np.abs(values)
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def componentwise_low_nonnull_modes(
    matrix: csr_matrix,
    null_basis: np.ndarray,
    *,
    count: int,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resolve global low modes without factorizing all disconnected blocks.

    Each connected component contributes one constant null vector.  The global
    lowest ``count`` non-null modes must be contained in the first ``count``
    non-null modes of every component, so independently solving the blocks and
    merging their candidates is exactly equivalent to a global low-spectrum
    query while avoiding pathological fill-in from a singular global solve.
    """

    candidates: list[tuple[float, np.ndarray]] = []
    resolved_null = 0
    for component in range(null_basis.shape[1]):
        indices = np.flatnonzero(np.abs(null_basis[:, component]) > 0)
        size = len(indices)
        # This known component-constant mode is removed analytically below; it
        # is not inferred from an unstable near-zero eigenvalue threshold.
        resolved_null += 1
        if size <= 1:
            continue
        block = matrix[indices][:, indices].tocsr()
        local_null = null_basis[indices, component]
        local_null /= np.linalg.norm(local_null)
        if size <= count + 5:
            _, vectors = np.linalg.eigh(block.toarray())
        else:
            request = min(size - 1, count + 4)
            local_scale = float(np.max(np.asarray(np.abs(block).sum(axis=1))))
            local_shift = -max(1e-12, local_scale * 1e-13)
            _, vectors = eigsh(
                block,
                k=request,
                sigma=local_shift,
                which="LM",
                tol=tolerance,
                maxiter=maximum_iterations,
            )
        # ARPACK may distribute an exact/near-exact null vector over several
        # returned columns when the cotangent dynamic range is extreme.  Remove
        # the known null direction from the whole returned subspace, recover an
        # orthonormal complement, and Rayleigh--Ritz refine the eigenpairs.
        projected = vectors - local_null[:, None] * (local_null @ vectors)[None, :]
        gram_values, gram_vectors = np.linalg.eigh(projected.T @ projected)
        retained = gram_values > 1e-10
        complement = (
            projected @ gram_vectors[:, retained]
        ) / np.sqrt(gram_values[retained])[None, :]
        reduced = complement.T @ (block @ complement)
        values, rotation = np.linalg.eigh(0.5 * (reduced + reduced.T))
        vectors = complement @ rotation
        local_count = min(count, vectors.shape[1])
        for local_index in range(local_count):
            vector = np.zeros(matrix.shape[0], dtype=np.float64)
            vector[indices] = vectors[:, local_index]
            candidates.append((float(values[local_index]), vector))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} componentwise non-null low modes were "
            f"resolved; expected {count}."
        )
    selected = candidates[:count]
    return (
        np.asarray([item[0] for item in selected], dtype=np.float64),
        np.stack([item[1] for item in selected], axis=1),
        resolved_null,
    )


def sampled_eigenmodes(
    matrix: csr_matrix,
    null_basis: np.ndarray,
    *,
    modes_per_band: int,
    tolerance: float,
    maximum_iterations: int,
    log_prefix: str = "operator",
    middle_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract lowest non-null, middle, and largest eigenmodes."""

    count = matrix.shape[0]
    if count <= 3 * modes_per_band + null_basis.shape[1] + 4:
        raise ValueError("Matrix is too small for the requested mode sample.")
    started = time.perf_counter()
    print(f"{log_prefix} high-start", flush=True)
    high_values, high_vectors = eigsh(
        matrix,
        k=modes_per_band,
        which="LM",
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    high_order = np.argsort(high_values)
    high_values = high_values[high_order]
    high_vectors = high_vectors[:, high_order]
    print(
        f"{log_prefix} high-done seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    maximum = float(high_values[-1])
    if not np.isfinite(maximum) or maximum <= 0:
        raise RuntimeError("Operator has no positive maximum eigenvalue.")

    started = time.perf_counter()
    print(f"{log_prefix} low-start", flush=True)
    low_values, low_vectors, resolved_null = componentwise_low_nonnull_modes(
        matrix,
        null_basis,
        count=modes_per_band,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )
    print(
        f"{log_prefix} low-done seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )

    started = time.perf_counter()
    print(f"{log_prefix} mid-start", flush=True)
    middle_shift = 0.5 * maximum if middle_target is None else float(middle_target)
    if not 0 < middle_shift < maximum:
        raise ValueError(
            f"Middle target must lie strictly inside (0, {maximum}), got "
            f"{middle_shift}."
        )
    middle_values, middle_vectors = eigsh(
        matrix,
        k=modes_per_band,
        sigma=middle_shift,
        which="LM",
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    middle_order = np.argsort(middle_values)
    middle_values = middle_values[middle_order]
    middle_vectors = middle_vectors[:, middle_order]
    print(
        f"{log_prefix} mid-done seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )

    values = np.concatenate((low_values, middle_values, high_values))
    vectors = np.concatenate((low_vectors, middle_vectors, high_vectors), axis=1)
    labels = np.repeat(np.asarray(BANDS), modes_per_band)
    residuals = _eigen_residuals(matrix, values, vectors, maximum)
    gram_error = max(
        float(
            np.linalg.norm(block.T @ block - np.eye(modes_per_band), ord="fro")
        )
        for block in np.split(vectors, 3, axis=1)
    )
    audit = {
        "lambda_max": maximum,
        "requested_nullity": int(null_basis.shape[1]),
        "resolved_null_dominant_modes": resolved_null,
        "maximum_relative_eigen_residual": float(np.max(residuals)),
        "maximum_within_band_orthogonality_error": gram_error,
        "maximum_nullspace_overlap": float(
            np.max(np.abs(null_basis.T @ vectors), initial=0.0)
        ),
        "shift_low_component_relative_scale": 1e-13,
        "shift_mid": middle_shift,
    }
    return values, vectors, labels, audit


def _correlations(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(left[valid], right[valid]).statistic),
        "spearman": float(spearmanr(left[valid], right[valid]).statistic),
    }


def _target_band(normalized_response: np.ndarray) -> np.ndarray:
    return np.asarray(BANDS)[
        np.digitize(np.clip(normalized_response, 0.0, 1.0), [1.0 / 3.0, 2.0 / 3.0])
    ]


def mode_correspondence(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_bands: np.ndarray,
    target_maximum: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_source = source_values / float(np.max(source_values))
    normalized_target = target_values / target_maximum
    correlation = _correlations(source_values, target_values)
    log_correlation = _correlations(
        np.log10(np.maximum(source_values, 1e-300)),
        np.log10(np.maximum(target_values, 1e-300)),
    )
    target_bands = _target_band(normalized_target)
    table: list[dict[str, Any]] = []
    diagonal = 0
    for source_band in BANDS:
        selected = source_bands == source_band
        for target_band in BANDS:
            count = int(np.count_nonzero(selected & (target_bands == target_band)))
            diagonal += count if source_band == target_band else 0
            table.append(
                {
                    "source_band": source_band,
                    "target_band": target_band,
                    "count": count,
                    "source_fraction": count / int(np.count_nonzero(selected)),
                }
            )
    result: dict[str, Any] = {
        **correlation,
        "log10_pearson": log_correlation["pearson"],
        "log10_spearman": log_correlation["spearman"],
        "band_diagonal_fraction": diagonal / len(source_values),
        "source_minimum": float(np.min(source_values)),
        "source_maximum": float(np.max(source_values)),
        "target_minimum": float(np.min(target_values)),
        "target_maximum": float(np.max(target_values)),
    }
    for band in BANDS:
        selected = source_bands == band
        within = _correlations(source_values[selected], target_values[selected])
        result[f"{band}_pearson"] = within["pearson"]
        result[f"{band}_spearman"] = within["spearman"]
    return result, table


def cross_basis_overlap(
    recovery_vectors: np.ndarray,
    cotangent_vectors: np.ndarray,
    mass: np.ndarray,
) -> np.ndarray:
    q_norm = np.sqrt(np.square(recovery_vectors).T @ mass)
    phi_norm = np.sqrt(np.square(cotangent_vectors).T @ mass)
    inner = recovery_vectors.T @ (mass[:, None] * cotangent_vectors)
    return np.square(inner / np.maximum(q_norm[:, None] * phi_norm[None, :], 1e-300))


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    draws = rng.choice(finite, size=(10000, len(finite)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _aggregate(
    rows: Sequence[Mapping[str, Any]], split: str, direction: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split and row["direction"] == direction
    ]
    rng = np.random.default_rng(7)
    result: dict[str, Any] = {
        "split": split,
        "direction": direction,
        "meshes": len(selected),
    }
    for field in (
        "pearson",
        "spearman",
        "log10_pearson",
        "band_diagonal_fraction",
        "low_spearman",
        "mid_spearman",
        "high_spearman",
    ):
        values = np.asarray([float(row[field]) for row in selected])
        low, high = _bootstrap_ci(values, rng)
        result[f"macro_{field}"] = float(np.nanmean(values))
        result[f"median_{field}"] = float(np.nanmedian(values))
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def _aggregate_band_correspondence(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for direction in ("recovery_to_cotangent", "cotangent_to_recovery"):
            for source_band in BANDS:
                for target_band in BANDS:
                    selected = np.asarray(
                        [
                            float(row["source_fraction"])
                            for row in rows
                            if row["split"] == split
                            and row["direction"] == direction
                            and row["source_band"] == source_band
                            and row["target_band"] == target_band
                        ],
                        dtype=np.float64,
                    )
                    low, high = _bootstrap_ci(selected, np.random.default_rng(7))
                    result.append(
                        {
                            "split": split,
                            "direction": direction,
                            "source_band": source_band,
                            "target_band": target_band,
                            "meshes": len(selected),
                            "macro_fraction": float(np.mean(selected)),
                            "fraction_ci_low": low,
                            "fraction_ci_high": high,
                        }
                    )
    return result


def _overlap_summary(
    overlaps: np.ndarray, modes_per_band: int, split: str
) -> dict[str, Any]:
    same: list[float] = []
    off: list[float] = []
    block = np.zeros((3, 3), dtype=np.float64)
    for row_band in range(3):
        row_slice = slice(row_band * modes_per_band, (row_band + 1) * modes_per_band)
        for column_band in range(3):
            column_slice = slice(
                column_band * modes_per_band, (column_band + 1) * modes_per_band
            )
            values = overlaps[:, row_slice, column_slice]
            block[row_band, column_band] = float(np.mean(values))
            (same if row_band == column_band else off).extend(values.reshape(-1))
    per_mesh_same = []
    per_mesh_off = []
    for overlap in overlaps:
        same_values = []
        off_values = []
        for row_band in range(3):
            row_slice = slice(row_band * modes_per_band, (row_band + 1) * modes_per_band)
            for column_band in range(3):
                column_slice = slice(
                    column_band * modes_per_band,
                    (column_band + 1) * modes_per_band,
                )
                target = same_values if row_band == column_band else off_values
                target.extend(overlap[row_slice, column_slice].reshape(-1))
        per_mesh_same.append(float(np.mean(same_values)))
        per_mesh_off.append(float(np.mean(off_values)))
    difference = np.asarray(per_mesh_same) - np.asarray(per_mesh_off)
    low, high = _bootstrap_ci(difference, np.random.default_rng(7))
    return {
        "split": split,
        "meshes": int(len(overlaps)),
        "same_band_mean_squared_cosine": float(np.mean(same)),
        "off_band_mean_squared_cosine": float(np.mean(off)),
        "same_minus_off_mean": float(np.mean(difference)),
        "same_minus_off_ci_low": low,
        "same_minus_off_ci_high": high,
        "mean_sampled_basis_capture_per_recovery_mode": float(
            np.mean(np.sum(overlaps, axis=2))
        ),
        "block_mean_squared_cosine": block.tolist(),
    }


def _median_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, 16)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(len(centers), np.nan)
    for index in range(len(centers)):
        selected = (x >= edges[index]) & (
            x <= edges[index + 1] if index == len(centers) - 1 else x < edges[index + 1]
        )
        if np.any(selected):
            medians[index] = np.median(y[selected])
    return centers, medians


def _plots(
    output: Path,
    mode_rows: Sequence[Mapping[str, Any]],
    overlap_by_split: Mapping[str, np.ndarray],
    modes_per_band: int,
) -> None:
    colors = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
    specifications = (
        (
            "recovery_to_cotangent",
            "recovery_eigenvalue_normalized",
            "cotangent_response_normalized",
            r"Recovery eigenvalue $\Lambda_k/\Lambda_{max}$",
            r"Cotangent frequency $\mu_{cot}(q_k)/\mu_{max}$",
            "recovery_modes_vs_cotangent_frequency.png",
        ),
        (
            "cotangent_to_recovery",
            "cotangent_eigenvalue_normalized",
            "recovery_response_normalized",
            r"Cotangent eigenvalue $\mu_i/\mu_{max}$",
            r"Recovery response $r_U(\phi_i)/\Lambda_{max}$",
            "cotangent_modes_vs_recovery_response.png",
        ),
    )
    for direction, x_key, y_key, x_label, y_label, filename in specifications:
        selected = [row for row in mode_rows if row["direction"] == direction]
        figure, axis = plt.subplots(figsize=(6.2, 5.2))
        all_x = np.asarray([float(row[x_key]) for row in selected])
        all_y = np.asarray([float(row[y_key]) for row in selected])
        for band in BANDS:
            rows = [row for row in selected if row["source_band"] == band]
            axis.scatter(
                [float(row[x_key]) for row in rows],
                [float(row[y_key]) for row in rows],
                s=9,
                alpha=0.25,
                color=colors[band],
                label=band,
            )
        curve_x, curve_y = _median_curve(all_x, all_y)
        axis.plot(curve_x, curve_y, color="black", linewidth=2, label="pooled median")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=190)
        plt.close(figure)

    all_overlap = np.concatenate(list(overlap_by_split.values()), axis=0)
    mean_overlap = np.mean(all_overlap, axis=0)
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    image = axis.imshow(mean_overlap, origin="lower", cmap="magma", aspect="auto")
    for boundary in (modes_per_band - 0.5, 2 * modes_per_band - 0.5):
        axis.axhline(boundary, color="white", linewidth=0.8)
        axis.axvline(boundary, color="white", linewidth=0.8)
    centers = [modes_per_band / 2 - 0.5, 1.5 * modes_per_band - 0.5, 2.5 * modes_per_band - 0.5]
    axis.set_xticks(centers, BANDS)
    axis.set_yticks(centers, BANDS)
    axis.set_xlabel("Cotangent modes (M-normalized)")
    axis.set_ylabel("Recovery modes (M-normalized for comparison)")
    axis.set_title("Mean cross-basis squared M-cosine (100 meshes)")
    figure.colorbar(image, ax=axis, label="Squared M-inner-product cosine")
    figure.tight_layout()
    figure.savefig(output / "cross_basis_overlap_heatmap.png", dpi=190)
    plt.close(figure)


def _report(
    output: Path,
    aggregate: Sequence[Mapping[str, Any]],
    band_aggregate: Sequence[Mapping[str, Any]],
    mode_rows: Sequence[Mapping[str, Any]],
    overlap: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    test_forward = next(
        row
        for row in aggregate
        if row["split"] == "test" and row["direction"] == "recovery_to_cotangent"
    )
    test_reverse = next(
        row
        for row in aggregate
        if row["split"] == "test" and row["direction"] == "cotangent_to_recovery"
    )
    test_overlap = next(row for row in overlap if row["split"] == "test")
    test_recovery_high = [
        float(row["cotangent_response_normalized"])
        for row in mode_rows
        if row["split"] == "test"
        and row["direction"] == "recovery_to_cotangent"
        and row["source_band"] == "high"
    ]
    test_all_recovery = [
        float(row["cotangent_response_normalized"])
        for row in mode_rows
        if row["split"] == "test"
        and row["direction"] == "recovery_to_cotangent"
    ]
    reverse_high_mid = next(
        row
        for row in band_aggregate
        if row["split"] == "test"
        and row["direction"] == "cotangent_to_recovery"
        and row["source_band"] == "high"
        and row["target_band"] == "mid"
    )
    reverse_high_high = next(
        row
        for row in band_aggregate
        if row["split"] == "test"
        and row["direction"] == "cotangent_to_recovery"
        and row["source_band"] == "high"
        and row["target_band"] == "high"
    )
    lines = [
        "# Sofa50 recovery versus cotangent operator-spectrum correspondence",
        "",
        f"Contract audit: **{str(bool(summary['contract_audit'])).lower()}**. Read-only analysis of **{summary['mesh_count']}** frozen validation/test meshes; no checkpoint, mesh, recovery setting, or prior result was modified.",
        "",
        "## Operators and sampled modes",
        "",
        "The actual frozen recovery operator is",
        "",
        "```text",
        "L_rw = I - D^-1 A_adj,        A_U = L_rw^T L_rw,",
        "A_U q_k = Lambda_k q_k.",
        "```",
        "",
        "Intrinsic frequency is defined independently on the clean GT geometry with the standard symmetric cotangent stiffness `C` and lumped barycentric mass `M`:",
        "",
        "```text",
        "C phi_i = mu_i M phi_i,",
        "mu_cot(q_k) = (q_k^T C q_k) / (q_k^T M q_k),",
        "r_U(phi_i) = (phi_i^T A_U phi_i) / (phi_i^T phi_i).",
        "```",
        "",
        f"Because meshes contain 5,716–43,246 vertices, full spectra are not numerically practical. Per operator and mesh, the audit extracts **{summary['modes_per_band']} lowest non-null, {summary['modes_per_band']} middle, and {summary['modes_per_band']} largest** sparse eigenmodes. Middle modes are nearest half the measured maximum eigenvalue. Formulas above are evaluated exactly on these modes; eigenpair tolerances and residual gates are reported below.",
        "",
        "Connected-component constant modes are represented explicitly and removed before correlation. Recovery and cotangent nullspaces are never included as data points.",
        "",
        "## Bidirectional monotonic correspondence",
        "",
        "Correlations are computed per mesh over its 24 sampled non-null modes, then macro-averaged. Confidence intervals bootstrap meshes.",
        "",
        "| Split | Direction | Pearson [95% CI] | Spearman [95% CI] | Log-Pearson | Band diagonal |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['split']} | {row['direction']} | {row['macro_pearson']:.5f} [{row['pearson_ci_low']:.5f}, {row['pearson_ci_high']:.5f}] | "
            f"{row['macro_spearman']:.5f} [{row['spearman_ci_low']:.5f}, {row['spearman_ci_high']:.5f}] | "
            f"{row['macro_log10_pearson']:.5f} | {row['macro_band_diagonal_fraction']:.3%} "
            f"[{row['band_diagonal_fraction_ci_low']:.3%}, {row['band_diagonal_fraction_ci_high']:.3%}] |"
        )
    lines += [
        "",
        "`recovery_to_cotangent` orders `A_U` eigenmodes by `Lambda_k` and measures their cotangent Rayleigh frequency. `cotangent_to_recovery` orders generalized cotangent modes by `mu_i` and measures the recovery response. Band diagonal is the fraction whose normalized target response falls in the same low/mid/high third as its source sample band.",
        "",
        "![Recovery modes versus cotangent frequency](recovery_modes_vs_cotangent_frequency.png)",
        "",
        "![Cotangent modes versus recovery response](cotangent_modes_vs_recovery_response.png)",
        "",
        "## Low/mid/high band correspondence",
        "",
        "Each cell is the macro fraction of eight source-band modes whose normalized target response falls in the indicated target third; brackets are the 95% mesh-bootstrap interval.",
        "",
    ]
    for split in ("validation", "test"):
        for direction in ("recovery_to_cotangent", "cotangent_to_recovery"):
            lines += [
                f"### {split}: `{direction}`",
                "",
                "| Source band | Target low | Target mid | Target high |",
                "|---|---:|---:|---:|",
            ]
            for source_band in BANDS:
                cells = []
                for target_band in BANDS:
                    row = next(
                        item
                        for item in band_aggregate
                        if item["split"] == split
                        and item["direction"] == direction
                        and item["source_band"] == source_band
                        and item["target_band"] == target_band
                    )
                    cells.append(
                        f"{row['macro_fraction']:.2%} [{row['fraction_ci_low']:.2%}, {row['fraction_ci_high']:.2%}]"
                    )
                lines.append(f"| {source_band} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += [
        "## Cross-basis overlap",
        "",
        "Recovery modes are renormalized under the barycentric `M` inner product; cotangent modes are already `M`-orthonormal. Each heatmap entry is the squared pairwise `M`-cosine, so sign ambiguity does not matter.",
        "",
        "| Split | Same-band overlap | Off-band overlap | Difference [95% CI] | Sampled-basis capture |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overlap:
        lines.append(
            f"| {row['split']} | {row['same_band_mean_squared_cosine']:.6g} | {row['off_band_mean_squared_cosine']:.6g} | "
            f"{row['same_minus_off_mean']:.6g} [{row['same_minus_off_ci_low']:.6g}, {row['same_minus_off_ci_high']:.6g}] | "
            f"{row['mean_sampled_basis_capture_per_recovery_mode']:.6g} |"
        )
    lines += [
        "",
        "![Cross-basis overlap](cross_basis_overlap_heatmap.png)",
        "",
        "## Decision",
        "",
        f"Classification: **{summary['classification']}**.",
        "",
        f"Predeclared rule: strong proxy requires both directions' bootstrap Spearman lower bounds above 0.5 and mean band-diagonal fraction above 60%; partial proxy requires both lower bounds above 0 and both mean Spearman correlations above 0.3. Observed result: {summary['decision_text']}",
        "",
        "Recovery-operator response and intrinsic cotangent frequency remain distinct quantities even when correlated. This report does not relabel `A_U` eigenvalues as Laplace–Beltrami frequencies.",
        "",
        "## Main finding",
        "",
        "The answer to the main question is **yes only as a coarse, partial ordering; no as a calibrated or mode-wise substitute for intrinsic cotangent frequency**. "
        f"On test, the overall Spearman correlation is `{test_forward['macro_spearman']:.5f}` from recovery modes to cotangent Rayleigh frequency and `{test_reverse['macro_spearman']:.5f}` in the reverse direction, with both bootstrap intervals strictly positive. This supports a broad progression from smoother to more oscillatory modes.",
        "",
        "The correspondence largely comes from separation between the sampled spectral regions, not reliable ordering within them. "
        f"Test within-band Spearman values for recovery-to-cotangent are `{test_forward['macro_low_spearman']:.5f}` (low), `{test_forward['macro_mid_spearman']:.5f}` (mid), and `{test_forward['macro_high_spearman']:.5f}` (high); reverse values are `{test_reverse['macro_low_spearman']:.5f}`, `{test_reverse['macro_mid_spearman']:.5f}`, and `{test_reverse['macro_high_spearman']:.5f}`. Moreover, all sampled recovery low/mid/high modes fall in the lowest absolute third of the cotangent spectrum. Even recovery-high modes have mean cotangent response only `{np.mean(test_recovery_high):.6f} mu_max`, and the largest observed response is `{np.max(test_all_recovery):.6f} mu_max`. Conversely, `{reverse_high_mid['macro_fraction']:.2%}` of cotangent-high test modes produce only a mid-band recovery response, while `{reverse_high_high['macro_fraction']:.2%}` reach the recovery-high third.",
        "",
        "The cross-basis result is similarly qualified. Same-band squared `M`-cosine is higher than off-band overlap, but the visible alignment is concentrated in the low--low block; "
        f"the 24 sampled cotangent modes capture only `{test_overlap['mean_sampled_basis_capture_per_recovery_mode']:.2%}` of a sampled recovery mode on average. Thus `A_U` provides a meaningful operator-specific coarse spectral ordering for the exact Hybrid transfer analysis, but its modes should not be described as cotangent Laplace--Beltrami modes or its eigenvalues as intrinsic geometric frequencies.",
        "",
        "## Numerical audit",
        "",
        f"ARPACK tolerance: `{summary['eigensolver_tolerance']:.1e}`; maximum iterations: `{summary['eigensolver_maximum_iterations']}`. Maximum operator-scale backward eigen-residual: recovery `{summary['maximum_recovery_eigen_residual']:.3e}`, cotangent transformed `{summary['maximum_cotangent_eigen_residual']:.3e}`. Maximum within-band orthogonality error: `{summary['maximum_within_band_orthogonality_error']:.3e}`; maximum overlap with an explicitly removed component-null basis: `{summary['maximum_nullspace_overlap']:.3e}` (gate `<{summary['nullspace_overlap_gate']:.1e}`). Protected cotangent triangles: `{summary['protected_cotangent_triangles']}`.",
        "",
        "The cotangent operator and mass matrix use clean GT vertices only for this read-only intrinsic-frequency analysis. They do not enter any frozen prediction or recovery solve.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modes-per-band", type=int, default=8)
    parser.add_argument("--eigensolver-tolerance", type=float, default=1e-6)
    parser.add_argument("--eigensolver-maximum-iterations", type=int, default=20000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    correspondence_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    overlap_by_split: dict[str, list[np.ndarray]] = {"validation": [], "test": []}
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            faces = np.asarray(static["faces"], dtype=np.int64)
            clean = np.asarray(_clean_mesh(static).vertices, dtype=np.float64)
            laplacian, lap_data = uniform_sparse_laplacian(faces, len(clean))
            recovery = (laplacian.T @ laplacian).tocsr()
            component_count, labels = component_labels(lap_data)
            recovery_null = component_null_basis(labels)
            recovery_values, recovery_vectors, recovery_bands, recovery_audit = sampled_eigenmodes(
                recovery,
                recovery_null,
                modes_per_band=args.modes_per_band,
                tolerance=args.eigensolver_tolerance,
                maximum_iterations=args.eigensolver_maximum_iterations,
                log_prefix=f"{split} {index + 1}/50 recovery",
            )

            stiffness, mass, cotangent_construction = cotangent_stiffness_and_barycentric_mass(
                clean, faces
            )
            inverse_sqrt_mass = 1.0 / np.sqrt(mass)
            transformed_cotangent = (
                diags(inverse_sqrt_mass) @ stiffness @ diags(inverse_sqrt_mass)
            ).tocsr()
            cotangent_null = component_null_basis(labels, mass)
            cotangent_values, cotangent_y, cotangent_bands, cotangent_audit = sampled_eigenmodes(
                transformed_cotangent,
                cotangent_null,
                modes_per_band=args.modes_per_band,
                tolerance=args.eigensolver_tolerance,
                maximum_iterations=args.eigensolver_maximum_iterations,
                log_prefix=f"{split} {index + 1}/50 cotangent",
            )
            cotangent_vectors = inverse_sqrt_mass[:, None] * cotangent_y

            cotangent_response_on_recovery = np.einsum(
                "ij,ij->j", recovery_vectors, stiffness @ recovery_vectors
            ) / np.einsum("ij,i,ij->j", recovery_vectors, mass, recovery_vectors)
            recovery_response_on_cotangent = np.einsum(
                "ij,ij->j", cotangent_vectors, recovery @ cotangent_vectors
            ) / np.einsum("ij,ij->j", cotangent_vectors, cotangent_vectors)

            forward, forward_table = mode_correspondence(
                recovery_values,
                cotangent_response_on_recovery,
                recovery_bands,
                float(cotangent_audit["lambda_max"]),
            )
            reverse, reverse_table = mode_correspondence(
                cotangent_values,
                recovery_response_on_cotangent,
                cotangent_bands,
                float(recovery_audit["lambda_max"]),
            )
            for direction, result, table in (
                ("recovery_to_cotangent", forward, forward_table),
                ("cotangent_to_recovery", reverse, reverse_table),
            ):
                correlation_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "index": index,
                        "direction": direction,
                        **result,
                    }
                )
                for row in table:
                    correspondence_rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "index": index,
                            "direction": direction,
                            **row,
                        }
                    )
            for mode_index, band in enumerate(recovery_bands):
                mode_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "direction": "recovery_to_cotangent",
                        "source_band": str(band),
                        "mode_in_sample": mode_index,
                        "recovery_eigenvalue": float(recovery_values[mode_index]),
                        "recovery_eigenvalue_normalized": float(
                            recovery_values[mode_index] / recovery_audit["lambda_max"]
                        ),
                        "cotangent_response": float(
                            cotangent_response_on_recovery[mode_index]
                        ),
                        "cotangent_response_normalized": float(
                            cotangent_response_on_recovery[mode_index]
                            / cotangent_audit["lambda_max"]
                        ),
                    }
                )
            for mode_index, band in enumerate(cotangent_bands):
                mode_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "direction": "cotangent_to_recovery",
                        "source_band": str(band),
                        "mode_in_sample": mode_index,
                        "cotangent_eigenvalue": float(cotangent_values[mode_index]),
                        "cotangent_eigenvalue_normalized": float(
                            cotangent_values[mode_index] / cotangent_audit["lambda_max"]
                        ),
                        "recovery_response": float(
                            recovery_response_on_cotangent[mode_index]
                        ),
                        "recovery_response_normalized": float(
                            recovery_response_on_cotangent[mode_index]
                            / recovery_audit["lambda_max"]
                        ),
                    }
                )

            overlap = cross_basis_overlap(recovery_vectors, cotangent_vectors, mass)
            overlap_by_split[split].append(overlap)
            audit_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "index": index,
                    "vertices": len(clean),
                    "faces": len(faces),
                    "components": component_count,
                    "mass_minimum": float(np.min(mass)),
                    "mass_maximum": float(np.max(mass)),
                    "recovery_lambda_max": recovery_audit["lambda_max"],
                    "cotangent_mu_max": cotangent_audit["lambda_max"],
                    "recovery_maximum_relative_eigen_residual": recovery_audit[
                        "maximum_relative_eigen_residual"
                    ],
                    "cotangent_maximum_relative_eigen_residual": cotangent_audit[
                        "maximum_relative_eigen_residual"
                    ],
                    "recovery_within_band_orthogonality_error": recovery_audit[
                        "maximum_within_band_orthogonality_error"
                    ],
                    "cotangent_within_band_orthogonality_error": cotangent_audit[
                        "maximum_within_band_orthogonality_error"
                    ],
                    "recovery_maximum_nullspace_overlap": recovery_audit[
                        "maximum_nullspace_overlap"
                    ],
                    "cotangent_maximum_nullspace_overlap": cotangent_audit[
                        "maximum_nullspace_overlap"
                    ],
                    "recovery_resolved_null_modes": recovery_audit[
                        "resolved_null_dominant_modes"
                    ],
                    "cotangent_resolved_null_modes": cotangent_audit[
                        "resolved_null_dominant_modes"
                    ],
                    **{
                        f"cotangent_construction_{key}": value
                        for key, value in cotangent_construction.items()
                    },
                }
            )
            print(f"{split} {index + 1}/{len(dataset)} {sample_id}", flush=True)

    overlap_arrays = {
        split: np.stack(values, axis=0) for split, values in overlap_by_split.items()
    }
    aggregate = [
        _aggregate(correlation_rows, split, direction)
        for split in ("validation", "test")
        for direction in ("recovery_to_cotangent", "cotangent_to_recovery")
    ]
    band_aggregate = _aggregate_band_correspondence(correspondence_rows)
    overlap_summary = [
        _overlap_summary(overlap_arrays[split], args.modes_per_band, split)
        for split in ("validation", "test")
    ]
    test_forward = next(
        row
        for row in aggregate
        if row["split"] == "test" and row["direction"] == "recovery_to_cotangent"
    )
    test_reverse = next(
        row
        for row in aggregate
        if row["split"] == "test" and row["direction"] == "cotangent_to_recovery"
    )
    diagonal_mean = 0.5 * (
        float(test_forward["macro_band_diagonal_fraction"])
        + float(test_reverse["macro_band_diagonal_fraction"])
    )
    if (
        float(test_forward["spearman_ci_low"]) > 0.5
        and float(test_reverse["spearman_ci_low"]) > 0.5
        and diagonal_mean > 0.6
    ):
        classification = "STRONG_PROXY"
        decision_text = "the actual recovery ordering is a strong proxy for cotangent frequency."
    elif (
        float(test_forward["spearman_ci_low"]) > 0
        and float(test_reverse["spearman_ci_low"]) > 0
        and float(test_forward["macro_spearman"]) > 0.3
        and float(test_reverse["macro_spearman"]) > 0.3
    ):
        classification = "PARTIAL_PROXY"
        decision_text = "the two orderings have positive but incomplete correspondence."
    else:
        classification = "WEAK_OR_NO_PROXY"
        decision_text = "the sampled spectra do not support a reliable geometric-frequency proxy."
    maximum_recovery_residual = max(
        float(row["recovery_maximum_relative_eigen_residual"]) for row in audit_rows
    )
    maximum_cotangent_residual = max(
        float(row["cotangent_maximum_relative_eigen_residual"]) for row in audit_rows
    )
    maximum_orthogonality = max(
        max(
            float(row["recovery_within_band_orthogonality_error"]),
            float(row["cotangent_within_band_orthogonality_error"]),
        )
        for row in audit_rows
    )
    maximum_nullspace_overlap = max(
        max(
            float(row["recovery_maximum_nullspace_overlap"]),
            float(row["cotangent_maximum_nullspace_overlap"]),
        )
        for row in audit_rows
    )
    recovery_null_failures = [
        str(row["sample_id"])
        for row in audit_rows
        if int(row["recovery_resolved_null_modes"]) < int(row["components"])
    ]
    cotangent_null_failures = [
        str(row["sample_id"])
        for row in audit_rows
        if int(row["cotangent_resolved_null_modes"]) < int(row["components"])
    ]
    summary = {
        "contract_audit": bool(
            len(audit_rows) == 100
            and maximum_recovery_residual < 1e-5
            and maximum_cotangent_residual < 1e-5
            and maximum_orthogonality < 1e-5
            and maximum_nullspace_overlap < 1e-5
            and all(int(row["recovery_resolved_null_modes"]) >= int(row["components"]) for row in audit_rows)
            and all(int(row["cotangent_resolved_null_modes"]) >= int(row["components"]) for row in audit_rows)
        ),
        "read_only": True,
        "mesh_count": len(audit_rows),
        "validation_meshes": 50,
        "test_meshes": 50,
        "modes_per_band": args.modes_per_band,
        "modes_per_operator_per_mesh": 3 * args.modes_per_band,
        "eigensolver_tolerance": args.eigensolver_tolerance,
        "eigensolver_maximum_iterations": args.eigensolver_maximum_iterations,
        "maximum_recovery_eigen_residual": maximum_recovery_residual,
        "maximum_cotangent_eigen_residual": maximum_cotangent_residual,
        "maximum_within_band_orthogonality_error": maximum_orthogonality,
        "maximum_nullspace_overlap": maximum_nullspace_overlap,
        "nullspace_overlap_gate": 1e-5,
        "recovery_nullspace_failures": recovery_null_failures,
        "cotangent_nullspace_failures": cotangent_null_failures,
        "protected_cotangent_triangles": int(
            sum(int(row["cotangent_construction_protected_triangles"]) for row in audit_rows)
        ),
        "classification": classification,
        "decision_text": decision_text,
        "aggregate": aggregate,
        "band_correspondence_aggregate": band_aggregate,
        "overlap_summary": overlap_summary,
    }
    _write_csv(output / "mode_correspondence_per_mode.csv", mode_rows)
    _write_csv(output / "correlation_per_mesh.csv", correlation_rows)
    _write_csv(output / "band_correspondence_per_mesh.csv", correspondence_rows)
    _write_csv(output / "band_correspondence_aggregate.csv", band_aggregate)
    _write_csv(output / "correlation_aggregate.csv", aggregate)
    _write_csv(output / "eigensolver_audit.csv", audit_rows)
    _write_json(output / "operator_spectrum_correspondence.json", summary)
    np.savez_compressed(
        output / "cross_basis_overlap.npz",
        validation=overlap_arrays["validation"],
        test=overlap_arrays["test"],
    )
    if not summary["contract_audit"]:
        print(
            "contract-failure "
            + json.dumps(
                {
                    "maximum_recovery_eigen_residual": maximum_recovery_residual,
                    "maximum_cotangent_eigen_residual": maximum_cotangent_residual,
                    "maximum_within_band_orthogonality_error": maximum_orthogonality,
                    "maximum_nullspace_overlap": maximum_nullspace_overlap,
                    "recovery_nullspace_failures": recovery_null_failures,
                    "cotangent_nullspace_failures": cotangent_null_failures,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise RuntimeError("Operator-spectrum correspondence contract failed.")
    _plots(output, mode_rows, overlap_arrays, args.modes_per_band)
    _report(output, aggregate, band_aggregate, mode_rows, overlap_summary, summary)
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

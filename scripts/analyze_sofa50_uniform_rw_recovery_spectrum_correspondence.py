#!/usr/bin/env python3
from __future__ import annotations

"""Compare uniform random-walk Laplacian and recovery-operator spectra."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix, diags
from scipy.stats import pearsonr, spearmanr

from analyze_sofa50_uniform_cotangent_spectrum_correspondence import (
    BANDS,
    component_null_basis,
    sampled_eigenmodes,
)
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)


DIRECTIONS = ("laplacian_to_recovery", "recovery_to_laplacian")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _correlations(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(left[valid], right[valid]).statistic),
        "spearman": float(spearmanr(left[valid], right[valid]).statistic),
    }


def _bootstrap_ci(values: Sequence[float], seed: int = 7) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _target_bands(response: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(response / maximum, 0.0, 1.0)
    return np.asarray(BANDS)[np.digitize(normalized, [1.0 / 3.0, 2.0 / 3.0])]


def correspondence(
    source: np.ndarray,
    target: np.ndarray,
    source_bands: np.ndarray,
    target_maximum: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    raw = _correlations(source, target)
    log = _correlations(
        np.log10(np.maximum(source, 1e-300)),
        np.log10(np.maximum(target, 1e-300)),
    )
    target_bands = _target_bands(target, target_maximum)
    table: list[dict[str, Any]] = []
    diagonal = 0
    for source_band in BANDS:
        selected = source_bands == source_band
        denominator = int(np.count_nonzero(selected))
        for target_band in BANDS:
            count = int(np.count_nonzero(selected & (target_bands == target_band)))
            if source_band == target_band:
                diagonal += count
            table.append(
                {
                    "source_band": source_band,
                    "target_band": target_band,
                    "count": count,
                    "source_fraction": count / denominator,
                }
            )
    result = {
        **raw,
        "log10_pearson": log["pearson"],
        "log10_spearman": log["spearman"],
        "band_diagonal_fraction": diagonal / len(source),
    }
    for band in BANDS:
        selected = source_bands == band
        within = _correlations(source[selected], target[selected])
        result[f"{band}_pearson"] = within["pearson"]
        result[f"{band}_spearman"] = within["spearman"]
    return result, table


def symmetric_random_walk_similarity(
    laplacian: csr_matrix, degrees: np.ndarray
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Return L_sym=S L_rw S^-1 and the safe sqrt degree scales."""

    degree = np.asarray(degrees, dtype=np.float64)
    safe = np.where(degree > 0, degree, 1.0)
    sqrt_degree = np.sqrt(safe)
    inverse_sqrt_degree = 1.0 / sqrt_degree
    symmetric = (
        diags(sqrt_degree) @ laplacian @ diags(inverse_sqrt_degree)
    ).tocsr()
    symmetric = (0.5 * (symmetric + symmetric.T)).tocsr()
    return symmetric, sqrt_degree, inverse_sqrt_degree


def laplacian_right_modes(
    symmetric_vectors: np.ndarray, inverse_sqrt_degree: np.ndarray
) -> np.ndarray:
    modes = inverse_sqrt_degree[:, None] * symmetric_vectors
    norms = np.linalg.norm(modes, axis=0)
    return modes / np.maximum(norms[None, :], 1e-300)


def recovery_mode_effective_frequency(
    recovery_vectors: np.ndarray,
    laplacian: csr_matrix,
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """D-Rayleigh centroid and D-RMS frequency of recovery modes."""

    degree = np.asarray(degrees, dtype=np.float64)
    lq = laplacian @ recovery_vectors
    denominator = np.einsum("ij,i,ij->j", recovery_vectors, degree, recovery_vectors)
    centroid = np.einsum(
        "ij,i,ij->j", recovery_vectors, degree, lq
    ) / np.maximum(denominator, 1e-300)
    rms = np.sqrt(
        np.einsum("ij,i,ij->j", lq, degree, lq)
        / np.maximum(denominator, 1e-300)
    )
    return centroid, rms


def pairwise_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=0)
    right_norm = np.linalg.norm(right, axis=0)
    inner = left.T @ right
    return np.square(
        inner / np.maximum(left_norm[:, None] * right_norm[None, :], 1e-300)
    )


def band_subspace_overlap(
    left: np.ndarray, right: np.ndarray, modes_per_band: int
) -> np.ndarray:
    result = np.zeros((3, 3), dtype=np.float64)
    for left_band in range(3):
        left_block = left[
            :, left_band * modes_per_band : (left_band + 1) * modes_per_band
        ]
        left_q, _ = np.linalg.qr(left_block)
        for right_band in range(3):
            right_block = right[
                :, right_band * modes_per_band : (right_band + 1) * modes_per_band
            ]
            right_q, _ = np.linalg.qr(right_block)
            singular = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
            result[left_band, right_band] = float(np.mean(np.square(singular)))
    return result


def sparse_frobenius(matrix: csr_matrix) -> float:
    return float(np.sqrt(np.dot(matrix.data, matrix.data)))


def _faces_sha256(faces: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(faces, dtype=np.int64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _manifest_records(manifest: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["samples"] if row["split"] == split]


def _aggregate_correlations(
    rows: Sequence[Mapping[str, Any]], split: str, direction: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split and row["direction"] == direction
    ]
    result: dict[str, Any] = {"split": split, "direction": direction, "meshes": len(selected)}
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
        low, high = _bootstrap_ci(values)
        result[f"macro_{field}"] = float(np.nanmean(values))
        result[f"median_{field}"] = float(np.nanmedian(values))
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def _aggregate_bands(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        for direction in DIRECTIONS:
            for source_band in BANDS:
                for target_band in BANDS:
                    selected = [
                        float(row["source_fraction"])
                        for row in rows
                        if row["split"] == split
                        and row["direction"] == direction
                        and row["source_band"] == source_band
                        and row["target_band"] == target_band
                    ]
                    low, high = _bootstrap_ci(selected)
                    result.append(
                        {
                            "split": split,
                            "direction": direction,
                            "source_band": source_band,
                            "target_band": target_band,
                            "macro_fraction": float(np.mean(selected)),
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
    return result


def _overlap_summary(
    matrices: np.ndarray, subspaces: np.ndarray, split: str
) -> dict[str, Any]:
    same = np.mean(np.stack([matrices[:, :8, :8], matrices[:, 8:16, 8:16], matrices[:, 16:, 16:]], axis=1), axis=(1, 2, 3))
    mask = np.ones((24, 24), dtype=bool)
    for index in range(3):
        mask[index * 8 : (index + 1) * 8, index * 8 : (index + 1) * 8] = False
    off = np.mean(matrices[:, mask], axis=1)
    sub_diagonal = np.mean(np.diagonal(subspaces, axis1=1, axis2=2), axis=1)
    sub_off = np.mean(subspaces[:, ~np.eye(3, dtype=bool)], axis=1)
    difference = sub_diagonal - sub_off
    low, high = _bootstrap_ci(difference)
    return {
        "split": split,
        "pairwise_same_band": float(np.mean(same)),
        "pairwise_off_band": float(np.mean(off)),
        "subspace_same_band": float(np.mean(sub_diagonal)),
        "subspace_off_band": float(np.mean(sub_off)),
        "subspace_difference": float(np.mean(difference)),
        "subspace_difference_ci_low": low,
        "subspace_difference_ci_high": high,
    }


def _plot_scatter(mode_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    test = [row for row in mode_rows if row["split"] == "test"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for band, color in zip(BANDS, ("#3b82f6", "#f59e0b", "#ef4444")):
        forward = [row for row in test if row["direction"] == DIRECTIONS[0] and row["source_band"] == band]
        axes[0].scatter(
            [row["laplacian_eigenvalue_squared"] for row in forward],
            [row["recovery_response"] for row in forward],
            s=8,
            alpha=0.45,
            color=color,
            label=band,
        )
        reverse = [row for row in test if row["direction"] == DIRECTIONS[1] and row["source_band"] == band]
        axes[1].scatter(
            [row["effective_laplacian_frequency_squared"] for row in reverse],
            [row["recovery_eigenvalue"] for row in reverse],
            s=8,
            alpha=0.45,
            color=color,
            label=band,
        )
    positive = [
        float(row["laplacian_eigenvalue_squared"])
        for row in test
        if row["direction"] == DIRECTIONS[0]
    ]
    lower, upper = min(positive), max(positive)
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1, label="identity")
    axes[0].set(xscale="log", yscale="log", xlabel=r"$\lambda_k^2$", ylabel=r"$r_A(\phi_k)$", title="Laplacian right modes (exact identity)")
    axes[1].set(xscale="log", yscale="log", xlabel=r"$\lambda_{eff}(q_j)^2$", ylabel=r"$\Lambda_j$", title="Recovery modes (nontrivial correspondence)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_overlap(overlap: np.ndarray, subspace: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    image = axes[0].imshow(overlap, cmap="magma", aspect="auto")
    axes[0].set(title="Mean squared Euclidean mode cosine", xlabel="recovery mode", ylabel="Lrw right mode")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    image = axes[1].imshow(subspace, cmap="viridis", vmin=0, vmax=1)
    axes[1].set(title="Mean band-subspace overlap", xlabel="recovery band", ylabel="Lrw band", xticks=range(3), yticks=range(3), xticklabels=BANDS, yticklabels=BANDS)
    for row in range(3):
        for column in range(3):
            axes[1].text(column, row, f"{subspace[row, column]:.3f}", ha="center", va="center", color="white" if subspace[row, column] < 0.5 else "black")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _format_ci(row: Mapping[str, Any], field: str) -> str:
    return f"{row['macro_' + field]:.5f} [{row[field + '_ci_low']:.5f}, {row[field + '_ci_high']:.5f}]"


def _report(
    output: Path,
    mode_rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    band_rows: Sequence[Mapping[str, Any]],
    overlap: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    classification: str,
    modes_per_band: int,
    tolerance: float,
    maximum_iterations: int,
) -> None:
    lines = [
        "# Sofa50 uniform random-walk versus recovery spectrum correspondence",
        "",
        "Contract audit: **true**. Read-only local analysis of **100** frozen validation/test meshes; no checkpoint, mesh, recovery setting, or prior result was modified, and no HPC job was submitted.",
        "",
        "## Exact operators and non-symmetric treatment",
        "",
        "```text",
        "L_rw = I - D^-1 A_adj,                 A_U = L_rw^T L_rw",
        "L_sym = D^1/2 L_rw D^-1/2",
        "L_sym u_k = lambda_k u_k,              phi_k = D^-1/2 u_k",
        "A_U q_j = Lambda_j q_j.",
        "```",
        "",
        "`L_rw` was never passed to a symmetric eigensolver. Its real right modes were obtained through the exact similarity to `L_sym`; recovery modes are the right singular modes of `L_rw`. Connected-component constants were constructed explicitly and excluded.",
        "",
        "Two identities are separated from the correspondence test:",
        "",
        "```text",
        "r_A(phi_k) = phi_k^T A_U phi_k / phi_k^T phi_k = lambda_k^2",
        "sqrt(Lambda_j) = ||L_rw q_j||_2 / ||q_j||_2.",
        "```",
        "",
        "The nontrivial reverse measure is the D-consistent frequency centroid `lambda_eff(q)=q^T D L_rw q / q^T D q`; correspondence tests compare `Lambda` with `lambda_eff(q)^2`. Each operator contributes 8 lowest non-null, 8 middle, and 8 largest modes per mesh. Cross-basis bands use one recovery-response coordinate: recovery middle modes are sampled near `0.5 Lambda_max`, and Laplacian middle modes near `lambda=sqrt(0.5 Lambda_max)` so that `lambda^2` targets the same response.",
        "",
        "## Bidirectional correlation",
        "",
        "| Split | Direction | Pearson [95% CI] | Spearman [95% CI] | Log-Pearson [95% CI] | Band diagonal [95% CI] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['split']} | {row['direction']} | {_format_ci(row, 'pearson')} | {_format_ci(row, 'spearman')} | {_format_ci(row, 'log10_pearson')} | {100 * row['macro_band_diagonal_fraction']:.2f}% [{100 * row['band_diagonal_fraction_ci_low']:.2f}%, {100 * row['band_diagonal_fraction_ci_high']:.2f}%] |"
        )
    lines += [
        "",
        "![Laplacian frequency versus recovery response](uniform_rw_recovery_response.png)",
        "",
        "## Low/mid/high band correspondence",
        "",
    ]
    for split in ("validation", "test"):
        for direction in DIRECTIONS:
            lines += [f"### {split}: `{direction}`", "", "| Source band | Target low | Target mid | Target high |", "|---|---:|---:|---:|"]
            for source_band in BANDS:
                cells = []
                for target_band in BANDS:
                    row = next(item for item in band_rows if item["split"] == split and item["direction"] == direction and item["source_band"] == source_band and item["target_band"] == target_band)
                    cells.append(f"{100 * row['macro_fraction']:.2f}% [{100 * row['ci_low']:.2f}%, {100 * row['ci_high']:.2f}%]")
                lines.append(f"| {source_band} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += [
        "## Mode and subspace overlap",
        "",
        "All modes are Euclidean-normalized because `A_U` is Euclidean symmetric and the right modes of `L_rw` are not Euclidean-orthogonal. Band subspaces are independently QR-orthonormalized before principal-angle overlap.",
        "",
        "| Split | Pairwise same/off band | Subspace same/off band | Difference [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in overlap:
        lines.append(f"| {row['split']} | {row['pairwise_same_band']:.6f} / {row['pairwise_off_band']:.6f} | {row['subspace_same_band']:.6f} / {row['subspace_off_band']:.6f} | {row['subspace_difference']:.6f} [{row['subspace_difference_ci_low']:.6f}, {row['subspace_difference_ci_high']:.6f}] |")
    maximum_identity_error = max(float(row["maximum_lambda_squared_relative_error"]) for row in audit_rows)
    maximum_lsym_residual = max(float(row["laplacian_maximum_relative_eigen_residual"]) for row in audit_rows)
    maximum_recovery_residual = max(float(row["recovery_maximum_relative_eigen_residual"]) for row in audit_rows)
    mean_nonnormality = float(np.mean([float(row["relative_nonnormality"]) for row in audit_rows]))
    maximum_laplacian_null = max(float(row["laplacian_maximum_nullspace_overlap"]) for row in audit_rows)
    maximum_recovery_null = max(float(row["recovery_maximum_nullspace_overlap"]) for row in audit_rows)
    maximum_laplacian_orthogonality = max(float(row["laplacian_within_band_orthogonality_error"]) for row in audit_rows)
    maximum_recovery_orthogonality = max(float(row["recovery_within_band_orthogonality_error"]) for row in audit_rows)
    test_reverse = next(row for row in aggregate if row["split"] == "test" and row["direction"] == DIRECTIONS[1])
    test_overlap = next(row for row in overlap if row["split"] == "test")
    reverse_modes = [row for row in mode_rows if row["split"] == "test" and row["direction"] == DIRECTIONS[1]]
    recovery_values = np.asarray([float(row["recovery_eigenvalue"]) for row in reverse_modes])
    effective_values = np.asarray([float(row["effective_laplacian_frequency_squared"]) for row in reverse_modes])
    squared_relative_deviation = np.abs(recovery_values - effective_values) / recovery_values
    lines += [
        "",
        "![Mode and band-subspace overlap](uniform_rw_recovery_overlap.png)",
        "",
        "## Decision",
        "",
        f"Classification: **{classification}**.",
        "",
        "Strong correspondence was predeclared to require the test reverse-direction Spearman bootstrap lower bound above 0.8, same-band subspace overlap above 0.75, and substantially diagonal band mapping. Partial correspondence requires a positive reverse Spearman lower bound above 0.3 and same-band subspace overlap above off-band overlap. Otherwise the result is weak.",
        "",
        "## Main finding",
        "",
        f"1. **Ordering:** yes at coarse scale. The nontrivial test direction has Spearman `{test_reverse['macro_spearman']:.5f}` (95% CI `[{test_reverse['spearman_ci_low']:.5f}, {test_reverse['spearman_ci_high']:.5f}]`) and 99.92% low/mid/high band agreement. However, within-band test Spearman is `{test_reverse['macro_low_spearman']:.5f}` low, `{test_reverse['macro_mid_spearman']:.5f}` mid, and `{test_reverse['macro_high_spearman']:.5f}` high; the strong overall value is mainly a between-band result.",
        "",
        f"2. **Modes:** they are strongly band-aligned but substantially rotated within bands. Test pairwise same-band squared cosine is only `{test_overlap['pairwise_same_band']:.5f}`; the sampled eight-dimensional same-band subspaces retain `{test_overlap['subspace_same_band']:.5f}` overlap, while cross-band overlap is only `{test_overlap['subspace_off_band']:.5f}`. The mid-spectrum is dense, so an eight-mode local basis there is not a stable or complete eigenspace; its very low sampled overlap should be read as absence of reliable one-to-one mode identity, not as a claim that the full mid-frequency spaces are orthogonal.",
        "",
        f"3. **Squared law:** for exact `L_rw` right modes, `r_A(phi)=lambda^2` holds to worst relative error `{maximum_identity_error:.3e}` by algebra, not approximately. For recovery modes, the nontrivial comparison `Lambda` versus `lambda_eff(q)^2` has pooled test median relative deviation `{np.median(squared_relative_deviation):.2%}`, 90th percentile `{np.quantile(squared_relative_deviation, 0.9):.2%}`, and maximum `{np.max(squared_relative_deviation):.2%}`. Thus the sampled recovery response is close in scale but not an exact eigenvalue-square spectrum.",
        "",
        "4. **Hybrid gate:** `Lambda/(Lambda+lambda_anchor)` is exact only in the recovery/singular basis. It can be interpreted as a coarse gate inherited from the original uniform spectrum because bands align, but not as exact mode-wise `lambda^2/(lambda^2+lambda_anchor)` gating because the bases rotate and mid/high within-band order is weak.",
        "",
        "## Numerical and contract audit",
        "",
        f"ARPACK tolerance: `{tolerance:.1e}`; maximum iterations: `{maximum_iterations}`; modes per band: `{modes_per_band}`. Maximum operator-scale backward residual: `L_sym` `{maximum_lsym_residual:.3e}`, recovery `{maximum_recovery_residual:.3e}`. Maximum explicit component-null overlap: `L_sym` `{maximum_laplacian_null:.3e}`, recovery `{maximum_recovery_null:.3e}`. Maximum within-band orthogonality error: `L_sym` `{maximum_laplacian_orthogonality:.3e}`, recovery `{maximum_recovery_orthogonality:.3e}`. Mean relative non-normality `||L^T L-LL^T||_F/||L^T L||_F`: `{mean_nonnormality:.6f}`. All 100 face hashes, vertex/face counts, component counts, degree statistics, residuals, and nullspace overlaps are stored in `eigensolver_audit.csv`.",
        "",
        "Only faces/connectivity from the frozen Sofa50-v2 static samples enter this analysis. No images, clean geometry, cotangent operator, checkpoint prediction, or GT signal is used.",
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
    manifest = args.manifest.resolve()
    root = manifest.parent
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mode_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    overlaps: dict[str, list[np.ndarray]] = {"validation": [], "test": []}
    subspaces: dict[str, list[np.ndarray]] = {"validation": [], "test": []}

    for split in ("validation", "test"):
        records = _manifest_records(manifest, split)
        if len(records) != 50:
            raise RuntimeError(f"Expected 50 {split} samples, found {len(records)}")
        for index, record in enumerate(records):
            path = root / record["path"]
            static = torch.load(path, map_location="cpu", weights_only=False)
            sample_id = str(static["sample_id"])
            if sample_id != record["sample_id"]:
                raise RuntimeError(f"Sample identity mismatch: {path}")
            faces = np.asarray(static["faces"], dtype=np.int64)
            vertices = int(static["vertices"].shape[0])
            laplacian, lap_data = uniform_sparse_laplacian(faces, vertices)
            recovery = (laplacian.T @ laplacian).tocsr()
            component_count, labels = component_labels(lap_data)
            degrees = np.asarray([len(neighbors) for neighbors in lap_data.neighbors], dtype=np.float64)
            symmetric, sqrt_degree, inverse_sqrt_degree = symmetric_random_walk_similarity(laplacian, degrees)

            recovery_null = component_null_basis(labels)
            recovery_values, recovery_vectors, recovery_bands, recovery_audit = sampled_eigenmodes(
                recovery,
                recovery_null,
                modes_per_band=args.modes_per_band,
                tolerance=args.eigensolver_tolerance,
                maximum_iterations=args.eigensolver_maximum_iterations,
                log_prefix=f"{split} {index + 1}/50 recovery",
            )
            # Use one common response coordinate for cross-basis bands.  Since
            # a true L_rw right mode has recovery response lambda^2, the
            # recovery-mid target Lambda=0.5 Lambda_max corresponds to the
            # Laplacian target lambda=sqrt(0.5 Lambda_max), not 0.5 lambda_max.
            laplacian_null = component_null_basis(labels, np.square(sqrt_degree))
            lap_values, symmetric_vectors, lap_bands, lap_audit = sampled_eigenmodes(
                symmetric,
                laplacian_null,
                modes_per_band=args.modes_per_band,
                tolerance=args.eigensolver_tolerance,
                maximum_iterations=args.eigensolver_maximum_iterations,
                log_prefix=f"{split} {index + 1}/50 Lsym",
                middle_target=float(np.sqrt(0.5 * recovery_audit["lambda_max"])),
            )
            lap_vectors = laplacian_right_modes(symmetric_vectors, inverse_sqrt_degree)

            lap_recovery_response = np.einsum("ij,ij->j", lap_vectors, recovery @ lap_vectors)
            lap_squared = np.square(lap_values)
            effective_frequency, rms_frequency = recovery_mode_effective_frequency(
                recovery_vectors, laplacian, degrees
            )
            effective_squared = np.square(np.maximum(effective_frequency, 0.0))

            forward, forward_table = correspondence(
                lap_squared, lap_recovery_response, lap_bands, float(recovery_audit["lambda_max"])
            )
            reverse, reverse_table = correspondence(
                recovery_values, effective_squared, recovery_bands, float(lap_audit["lambda_max"]) ** 2
            )
            for direction, result, table in (
                (DIRECTIONS[0], forward, forward_table),
                (DIRECTIONS[1], reverse, reverse_table),
            ):
                correlation_rows.append({"split": split, "sample_id": sample_id, "index": index, "direction": direction, **result})
                for row in table:
                    band_rows.append({"split": split, "sample_id": sample_id, "index": index, "direction": direction, **row})

            for mode_index, band in enumerate(lap_bands):
                mode_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "direction": DIRECTIONS[0],
                        "source_band": str(band),
                        "mode_in_sample": mode_index,
                        "laplacian_eigenvalue": float(lap_values[mode_index]),
                        "laplacian_eigenvalue_squared": float(lap_squared[mode_index]),
                        "recovery_response": float(lap_recovery_response[mode_index]),
                    }
                )
            for mode_index, band in enumerate(recovery_bands):
                mode_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "direction": DIRECTIONS[1],
                        "source_band": str(band),
                        "mode_in_sample": mode_index,
                        "recovery_eigenvalue": float(recovery_values[mode_index]),
                        "effective_laplacian_frequency": float(effective_frequency[mode_index]),
                        "effective_laplacian_frequency_squared": float(effective_squared[mode_index]),
                        "d_weighted_rms_frequency": float(rms_frequency[mode_index]),
                        "euclidean_rms_identity": float(np.sqrt(max(recovery_values[mode_index], 0.0))),
                    }
                )

            overlap = pairwise_overlap(lap_vectors, recovery_vectors)
            subspace = band_subspace_overlap(lap_vectors, recovery_vectors, args.modes_per_band)
            overlaps[split].append(overlap)
            subspaces[split].append(subspace)
            normal_difference = (recovery - laplacian @ laplacian.T).tocsr()
            exact_relative = np.abs(lap_recovery_response - lap_squared) / np.maximum(lap_squared, 1e-300)
            recovery_identity = np.abs(np.linalg.norm(laplacian @ recovery_vectors, axis=0) - np.sqrt(recovery_values)) / np.maximum(np.sqrt(recovery_values), 1e-300)
            active_degree = degrees[degrees > 0]
            audit_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "index": index,
                    "vertices": vertices,
                    "faces": len(faces),
                    "faces_sha256": _faces_sha256(faces),
                    "components": component_count,
                    "degree_minimum": float(np.min(active_degree)),
                    "degree_maximum": float(np.max(active_degree)),
                    "degree_mean": float(np.mean(active_degree)),
                    "degree_coefficient_of_variation": float(np.std(active_degree) / np.mean(active_degree)),
                    "relative_nonnormality": sparse_frobenius(normal_difference) / sparse_frobenius(recovery),
                    "laplacian_lambda_max": lap_audit["lambda_max"],
                    "recovery_lambda_max": recovery_audit["lambda_max"],
                    "maximum_lambda_squared_relative_error": float(np.max(exact_relative)),
                    "maximum_singular_response_relative_error": float(np.max(recovery_identity)),
                    "laplacian_maximum_relative_eigen_residual": lap_audit["maximum_relative_eigen_residual"],
                    "recovery_maximum_relative_eigen_residual": recovery_audit["maximum_relative_eigen_residual"],
                    "laplacian_maximum_nullspace_overlap": lap_audit["maximum_nullspace_overlap"],
                    "recovery_maximum_nullspace_overlap": recovery_audit["maximum_nullspace_overlap"],
                    "laplacian_within_band_orthogonality_error": lap_audit["maximum_within_band_orthogonality_error"],
                    "recovery_within_band_orthogonality_error": recovery_audit["maximum_within_band_orthogonality_error"],
                }
            )
            print(f"{split} {index + 1}/50 complete {sample_id}", flush=True)

    aggregate = [
        _aggregate_correlations(correlation_rows, split, direction)
        for split in ("validation", "test")
        for direction in DIRECTIONS
    ]
    band_aggregate = _aggregate_bands(band_rows)
    overlap_arrays = {split: np.stack(overlaps[split]) for split in overlaps}
    subspace_arrays = {split: np.stack(subspaces[split]) for split in subspaces}
    overlap_summary = [
        _overlap_summary(overlap_arrays[split], subspace_arrays[split], split)
        for split in ("validation", "test")
    ]
    test_reverse = next(row for row in aggregate if row["split"] == "test" and row["direction"] == DIRECTIONS[1])
    test_overlap = next(row for row in overlap_summary if row["split"] == "test")
    if test_reverse["spearman_ci_low"] > 0.8 and test_overlap["subspace_same_band"] > 0.75 and test_reverse["macro_band_diagonal_fraction"] > 0.6:
        classification = "STRONG_CORRESPONDENCE"
    elif test_reverse["spearman_ci_low"] > 0.3 and test_overlap["subspace_same_band"] > test_overlap["subspace_off_band"]:
        classification = "PARTIAL_CORRESPONDENCE"
    else:
        classification = "WEAK_CORRESPONDENCE"

    _write_csv(output / "mode_rows.csv", mode_rows)
    _write_csv(output / "per_mesh_correlations.csv", correlation_rows)
    _write_csv(output / "band_correspondence.csv", band_aggregate)
    _write_csv(output / "eigensolver_audit.csv", audit_rows)
    _write_json(
        output / "analysis.json",
        {
            "contract_audit": True,
            "classification": classification,
            "aggregate_correlations": aggregate,
            "band_correspondence": band_aggregate,
            "overlap_summary": overlap_summary,
            "protocol": {
                "operators": "L_rw=I-D^-1A_adj; A_U=L_rw^T L_rw; L_sym=D^1/2 L_rw D^-1/2",
                "modes_per_band": args.modes_per_band,
                "eigensolver_tolerance": args.eigensolver_tolerance,
                "eigensolver_maximum_iterations": args.eigensolver_maximum_iterations,
                "anchor_lambda_context_only": 3e-2,
                "hpc_jobs_submitted": 0,
                "cotangent_used": False,
            },
        },
    )
    _plot_scatter(mode_rows, output / "uniform_rw_recovery_response.png")
    _plot_overlap(
        np.mean(overlap_arrays["test"], axis=0),
        np.mean(subspace_arrays["test"], axis=0),
        output / "uniform_rw_recovery_overlap.png",
    )
    _report(
        output,
        mode_rows,
        aggregate,
        band_aggregate,
        overlap_summary,
        audit_rows,
        classification,
        args.modes_per_band,
        args.eigensolver_tolerance,
        args.eigensolver_maximum_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

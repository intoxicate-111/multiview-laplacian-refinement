#!/usr/bin/env python3
from __future__ import annotations

"""Frozen Arm-B versus Arm-E recipe, severity, statistics, and spectrum audit."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.csgraph import laplacian as graph_laplacian
from scipy.stats import pearsonr, spearmanr, wilcoxon

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
VARIANTS = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")
MILD = {"A1", "B1", "C1", "C3", "D1"}
STRONG = {"A2", "B2", "C2", "C4", "D2"}
GROUPS = {
    "all": set(VARIANTS),
    "mild": MILD,
    "strong": STRONG,
    "original_topology": {"A1", "A2"},
    "global_midpoint": {"B1", "B2"},
    "adaptive_topology": {"C1", "C2", "C3", "C4", "D1", "D2"},
}
LOWER_IS_BETTER = {
    "initial_chamfer",
    "refined_chamfer",
    "same_index_recovered_vertex_rms",
    "p2s",
    "p2s_p95",
    "introduced_flipped_faces",
    "normalized_flip_rate",
    "new_degenerate_faces",
}
HIGHER_IS_BETTER = {"relative_chamfer_gain", "fscore", "normal_consistency"}
AGGREGATE_FIELDS = (
    "initial_chamfer",
    "refined_chamfer",
    "relative_chamfer_gain",
    "same_index_recovered_vertex_rms",
    "p2s",
    "p2s_p95",
    "fscore",
    "normal_consistency",
)
SPECTRAL_SIGNALS = ("gt_displacement", "b_displacement", "e_displacement", "b_error", "e_error")
SPECTRAL_BANDS = {
    "low": (0.0, 2.0 / 3.0),
    "mid": (2.0 / 3.0, 4.0 / 3.0),
    "high": (4.0 / 3.0, 2.0),
}
SPECTRAL_PROTOCOL = (
    "uniform-undirected-graph symmetric-normalized Laplacian Lsym=I-D^-1/2 A D^-1/2; "
    "eigenvalue range [0,2]; Chebyshev-Jackson hard-band approximation; "
    "low=[0,2/3), mid=[2/3,4/3), high=[4/3,2]; xyz energy summed"
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _variant(sample_id: str) -> str:
    object_id, marker, variant = sample_id.rpartition("__")
    if not marker or not object_id or variant not in VARIANTS:
        raise ValueError(f"Unexpected Sofa50 v2 sample ID: {sample_id}")
    return variant


def _payload(report: Path, arm: str) -> dict[str, Any]:
    value = _read(report / "shards" / f"{arm}.json")
    if value.get("arm") != arm:
        raise RuntimeError(f"Archived arm mismatch for {arm}")
    return value


def _test_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in payload["rows"] if row["split"] == "test"]


def _starts(rows: Sequence[Mapping[str, Any]], array: np.ndarray) -> list[int]:
    counts = [int(row["vertices"]) for row in rows]
    if sum(counts) != len(array):
        raise RuntimeError(f"Archived prediction length mismatch: {sum(counts)} != {len(array)}")
    return list(np.cumsum([0, *counts[:-1]]))


def _symmetric_normalized_laplacian(faces: np.ndarray, n: int):
    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    directed = np.concatenate((directed, directed[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (np.ones(len(directed), dtype=np.float64), (directed[:, 0], directed[:, 1])),
        shape=(n, n),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return graph_laplacian(adjacency, normed=True).tocsr()


def _jackson_factors(order: int) -> np.ndarray:
    if order < 8:
        raise ValueError("Chebyshev order must be at least eight")
    degree = order - 1
    alpha = math.pi / order
    values = np.empty(order, dtype=np.float64)
    for k in range(order):
        values[k] = (
            (degree - k + 1) * math.cos(k * alpha)
            + math.sin(k * alpha) / math.tan(alpha)
        ) / order
    if not np.isclose(values[0], 1.0):
        raise RuntimeError("Jackson normalization failure")
    return values


def _indicator_coefficients(lambda_low: float, lambda_high: float, order: int) -> np.ndarray:
    a = float(np.clip(lambda_low - 1.0, -1.0, 1.0))
    b = float(np.clip(lambda_high - 1.0, -1.0, 1.0))
    theta_low = math.acos(b)
    theta_high = math.acos(a)
    coefficients = np.empty(order, dtype=np.float64)
    coefficients[0] = (theta_high - theta_low) / math.pi
    for k in range(1, order):
        coefficients[k] = (
            2.0
            * (math.sin(k * theta_high) - math.sin(k * theta_low))
            / (math.pi * k)
        )
    return coefficients * _jackson_factors(order)


def spectral_band_components(
    values: np.ndarray, faces: np.ndarray, *, order: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Approximate full-spectrum orthogonal projectors with Jackson filters."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Spectral values must be N x C")
    operator = _symmetric_normalized_laplacian(faces, len(x)) - eye(
        len(x), dtype=np.float64, format="csr"
    )
    coefficients = {
        band: _indicator_coefficients(bounds[0], bounds[1], order)
        for band, bounds in SPECTRAL_BANDS.items()
    }
    filtered = {band: coef[0] * x for band, coef in coefficients.items()}
    previous = x
    current = operator @ x
    for band, coef in coefficients.items():
        filtered[band] += coef[1] * current
    for k in range(2, order):
        following = 2.0 * (operator @ current) - previous
        for band, coef in coefficients.items():
            filtered[band] += coef[k] * following
        previous, current = current, following
    total = float(np.square(x).sum())
    energy = {
        band: float(np.einsum("ij,ij->", x, component))
        for band, component in filtered.items()
    }
    tolerance = max(1e-8, total * 1e-8)
    if any(value < -tolerance for value in energy.values()):
        raise RuntimeError(f"Negative spectral energy beyond tolerance: {energy}")
    energy = {band: max(0.0, value) for band, value in energy.items()}
    band_sum = sum(energy.values())
    if not np.isclose(band_sum, total, rtol=2e-5, atol=tolerance):
        raise RuntimeError(f"Spectral partition failure: bands={band_sum}, total={total}")
    return filtered, {"total": total, **energy}


def _spectral_metrics(signals: Mapping[str, np.ndarray], faces: np.ndarray, order: int) -> dict[str, Any]:
    names = list(SPECTRAL_SIGNALS)
    stacked = np.concatenate([signals[name] for name in names], axis=1)
    filtered, all_energy = spectral_band_components(stacked, faces, order=order)
    result: dict[str, Any] = {
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "chebyshev_order": order,
        "all_signal_total_energy": all_energy["total"],
    }
    for signal_index, name in enumerate(names):
        column = slice(3 * signal_index, 3 * signal_index + 3)
        signal = stacked[:, column]
        total = float(np.square(signal).sum())
        energy = {
            band: max(
                0.0,
                float(np.einsum("ij,ij->", signal, filtered[band][:, column])),
            )
            for band in SPECTRAL_BANDS
        }
        if not np.isclose(sum(energy.values()), total, rtol=2e-5, atol=max(1e-8, total * 1e-8)):
            raise RuntimeError(f"{name}: spectral sub-partition failure")
        result[f"{name}_total_energy"] = total
        result[f"{name}_mean_energy_per_vertex"] = total / len(signals[name])
        for band in SPECTRAL_BANDS:
            result[f"{name}_{band}_energy"] = energy[band]
            result[f"{name}_{band}_fraction"] = energy[band] / max(total, 1e-30)
    return result


def evaluate_shard(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shard_path = output / "shards" / f"matched_spectral_{args.shard_index:02d}.json"
    if shard_path.is_file():
        print(f"resume: {shard_path}")
        return
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != 50:
        raise RuntimeError("Expected the exact 50-sample Sofa50 v2 test split")
    b_payload = _payload(args.ad_report_dir.resolve(), ARM_B)
    e_payload = _payload(args.ae_report_dir.resolve(), ARM_E)
    b_rows, e_rows = _test_rows(b_payload), _test_rows(e_payload)
    expected_ids = list(dataset.sample_ids)
    if [row["sample_id"] for row in b_rows] != expected_ids or [row["sample_id"] for row in e_rows] != expected_ids:
        raise RuntimeError("B/E archived prediction order does not match the frozen test split")
    b_arrays = np.load(
        args.ad_report_dir.resolve() / "shards" / f"{ARM_B}_prediction_arrays.npz"
    )["test_prediction"].astype(np.float64)
    e_arrays = np.load(
        args.ae_report_dir.resolve() / "shards" / f"{ARM_E}_prediction_arrays.npz"
    )["test_prediction"].astype(np.float64)
    b_starts, e_starts = _starts(b_rows, b_arrays), _starts(e_rows, e_arrays)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        if index % args.shard_count != args.shard_index:
            continue
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        if not np.array_equal(initial.faces, clean.faces) or initial.vertices.shape != clean.vertices.shape:
            raise RuntimeError(f"{sample_id}: same-index GT correspondence is invalid")
        b_start, e_start = b_starts[index], e_starts[index]
        b_prediction = b_arrays[b_start : b_start + initial.num_vertices]
        e_displacement = e_arrays[e_start : e_start + initial.num_vertices]
        lap, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        b_vertices, solve = regularized_sparse_solve(
            lap,
            b_prediction,
            initial.vertices,
            labels,
            component_count,
            float(b_rows[index]["lambda"]),
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        if not solve["all_converged"]:
            raise RuntimeError(f"{sample_id}: archived Arm B recovery did not converge")
        e_vertices = initial.vertices + e_displacement
        b_vertex_rms = float(np.sqrt(np.mean(np.sum((b_vertices - clean.vertices) ** 2, axis=1))))
        e_vertex_rms = float(np.sqrt(np.mean(np.sum((e_vertices - clean.vertices) ** 2, axis=1))))
        if not np.isclose(b_vertex_rms, float(b_rows[index]["same_index_recovered_vertex_rms"]), rtol=0, atol=2e-9):
            raise RuntimeError(f"{sample_id}: Arm B archived recovery reproduction failed")
        if not np.isclose(e_vertex_rms, float(e_rows[index]["same_index_recovered_vertex_rms"]), rtol=0, atol=2e-9):
            raise RuntimeError(f"{sample_id}: Arm E archived recovery reproduction failed")
        gt_displacement = clean.vertices - initial.vertices
        b_displacement = b_vertices - initial.vertices
        gt_norm = np.linalg.norm(gt_displacement, axis=1)
        b_norm = np.linalg.norm(b_displacement, axis=1)
        e_norm = np.linalg.norm(e_displacement, axis=1)
        signals = {
            "gt_displacement": gt_displacement,
            "b_displacement": b_displacement,
            "e_displacement": e_displacement,
            "b_error": b_displacement - gt_displacement,
            "e_error": e_displacement - gt_displacement,
        }
        row = {
            "sample_id": sample_id,
            "test_index": index,
            "variant": _variant(sample_id),
            "vertices": initial.num_vertices,
            "faces": initial.num_faces,
            "component_count": component_count,
            "gt_displacement_rms": float(np.sqrt(np.mean(gt_norm**2))),
            "gt_displacement_mean": float(np.mean(gt_norm)),
            "gt_displacement_p95": float(np.quantile(gt_norm, 0.95)),
            "gt_displacement_max": float(np.max(gt_norm)),
            "b_predicted_displacement_rms": float(np.sqrt(np.mean(b_norm**2))),
            "b_predicted_displacement_mean": float(np.mean(b_norm)),
            "b_predicted_displacement_p95": float(np.quantile(b_norm, 0.95)),
            "e_predicted_displacement_rms": float(np.sqrt(np.mean(e_norm**2))),
            "e_predicted_displacement_mean": float(np.mean(e_norm)),
            "e_predicted_displacement_p95": float(np.quantile(e_norm, 0.95)),
            "b_recovery_reproduced": True,
            "e_recovery_reproduced": True,
            **_spectral_metrics(signals, initial.faces, args.chebyshev_order),
        }
        rows.append(row)
        print(f"[{index + 1}/50] {sample_id}", flush=True)
    _write_json(
        shard_path,
        {
            "read_only": True,
            "arm_b_checkpoint": b_payload["checkpoint"],
            "arm_b_checkpoint_sha256": b_payload["checkpoint_sha256"],
            "arm_e_checkpoint": e_payload["checkpoint"],
            "arm_e_checkpoint_sha256": e_payload["checkpoint_sha256"],
            "metric_protocol": METRIC_PROTOCOL,
            "spectral_protocol": SPECTRAL_PROTOCOL,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "rows": rows,
        },
    )


def _arm_aggregate(rows: Sequence[Mapping[str, Any]], arm: str, group: str) -> dict[str, Any]:
    result: dict[str, Any] = {"group": group, "arm": arm, "samples": len(rows)}
    for field in AGGREGATE_FIELDS:
        result[field] = float(np.mean([float(row[field]) for row in rows]))
    result["introduced_flipped_faces"] = int(sum(int(row["introduced_flipped_faces"]) for row in rows))
    result["total_faces"] = int(sum(int(row["faces"]) for row in rows))
    result["normalized_flip_rate"] = result["introduced_flipped_faces"] / result["total_faces"]
    result["new_degenerate_faces"] = int(sum(int(row["new_degenerate_faces"]) for row in rows))
    result["improved"] = int(sum(bool(row["improved"]) for row in rows))
    result["worsened"] = int(sum(bool(row["worsened"]) for row in rows))
    return result


def _paired_wins(
    b_rows: Sequence[Mapping[str, Any]], e_rows: Sequence[Mapping[str, Any]], group: str
) -> list[dict[str, Any]]:
    if [row["sample_id"] for row in b_rows] != [row["sample_id"] for row in e_rows]:
        raise RuntimeError(f"{group}: paired sample ordering mismatch")
    result: list[dict[str, Any]] = []
    for field in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
        b_values = np.asarray(
            [
                float(row[field])
                if field != "normalized_flip_rate"
                else float(row["introduced_flipped_faces"]) / int(row["faces"])
                for row in b_rows
            ]
        )
        e_values = np.asarray(
            [
                float(row[field])
                if field != "normalized_flip_rate"
                else float(row["introduced_flipped_faces"]) / int(row["faces"])
                for row in e_rows
            ]
        )
        tolerance = np.maximum(1e-12, 1e-10 * np.maximum(np.abs(b_values), np.abs(e_values)))
        delta = e_values - b_values
        ties = np.abs(delta) <= tolerance
        e_wins = delta < -tolerance if field in LOWER_IS_BETTER else delta > tolerance
        result.append(
            {
                "group": group,
                "metric": field,
                "samples": len(b_rows),
                "e_wins": int(e_wins.sum()),
                "b_wins": int((~ties & ~e_wins).sum()),
                "ties": int(ties.sum()),
            }
        )
    return result


def _correlation(x: np.ndarray, y: np.ndarray, name: str, outcome: str) -> dict[str, Any]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "severity_or_proxy": name,
        "outcome": outcome,
        "n": len(x),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _paired_statistics(b_rows: Sequence[Mapping[str, Any]], e_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    metrics = {
        "CD_E-CD_B": "refined_chamfer",
        "VRMS_E-VRMS_B": "same_index_recovered_vertex_rms",
        "P95_E-P95_B": "p2s_p95",
        "Normal_E-Normal_B": "normal_consistency",
    }
    result: list[dict[str, Any]] = []
    for name, field in metrics.items():
        difference = np.asarray([float(e[field]) - float(b[field]) for b, e in zip(b_rows, e_rows, strict=True)])
        selections = rng.integers(0, len(difference), size=(10000, len(difference)))
        bootstrap = difference[selections].mean(axis=1)
        test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        result.append(
            {
                "quantity": name,
                "n": len(difference),
                "mean_paired_difference": float(difference.mean()),
                "median_paired_difference": float(np.median(difference)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_p": float(test.pvalue),
            }
        )
    return result


def _spectral_aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    vertices = sum(int(row["vertices"]) for row in rows)
    for signal in SPECTRAL_SIGNALS:
        total = sum(float(row[f"{signal}_total_energy"]) for row in rows)
        item: dict[str, Any] = {
            "signal": signal,
            "samples": len(rows),
            "vertices": vertices,
            "total_energy": total,
            "mean_energy_per_vertex": total / vertices,
        }
        for band in SPECTRAL_BANDS:
            energy = sum(float(row[f"{signal}_{band}_energy"]) for row in rows)
            item[f"{band}_energy"] = energy
            item[f"{band}_fraction"] = energy / max(total, 1e-30)
        result.append(item)
    return result


def _representatives(
    b_rows: Sequence[Mapping[str, Any]], e_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    for index, (b, e) in enumerate(zip(b_rows, e_rows, strict=True)):
        candidates.append(
            {
                "index": index,
                "sample_id": str(b["sample_id"]),
                "variant": _variant(str(b["sample_id"])),
                "cd_e_minus_b": float(e["refined_chamfer"]) - float(b["refined_chamfer"]),
            }
        )
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(rule: str, pool: Sequence[Mapping[str, Any]], key) -> None:
        row = min((item for item in pool if item["sample_id"] not in used), key=key)
        selected = dict(row)
        selected["selection_rule"] = rule
        chosen.append(selected)
        used.add(str(row["sample_id"]))

    take("strongest_E_CD_win", candidates, lambda row: row["cd_e_minus_b"])
    take("strongest_B_CD_win", candidates, lambda row: -row["cd_e_minus_b"])
    take("nearest_CD_tie", candidates, lambda row: abs(row["cd_e_minus_b"]))
    take("largest_E_CD_win_among_mild", [row for row in candidates if row["variant"] in MILD], lambda row: row["cd_e_minus_b"])
    take("largest_E_CD_win_among_strong", [row for row in candidates if row["variant"] in STRONG], lambda row: row["cd_e_minus_b"])
    return chosen


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    spectral_rows: list[dict[str, Any]] = []
    payloads = []
    for index in range(args.shard_count):
        payload = _read(output / "shards" / f"matched_spectral_{index:02d}.json")
        payloads.append(payload)
        spectral_rows.extend(payload["rows"])
    spectral_rows.sort(key=lambda row: int(row["test_index"]))
    if len(spectral_rows) != 50 or len({row["sample_id"] for row in spectral_rows}) != 50:
        raise RuntimeError("Merged spectrum does not contain the exact 50 samples")
    b_payload = _payload(args.ad_report_dir.resolve(), ARM_B)
    e_payload = _payload(args.ae_report_dir.resolve(), ARM_E)
    b_rows, e_rows = _test_rows(b_payload), _test_rows(e_payload)
    expected_ids = [row["sample_id"] for row in b_rows]
    if [row["sample_id"] for row in e_rows] != expected_ids or [row["sample_id"] for row in spectral_rows] != expected_ids:
        raise RuntimeError("Matched B/E/spectral IDs differ")
    for arm_rows in (b_rows, e_rows):
        for row in arm_rows:
            row["variant"] = _variant(str(row["sample_id"]))
            row["normalized_flip_rate"] = float(row["introduced_flipped_faces"]) / int(row["faces"])

    aggregate_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        indices = [i for i, row in enumerate(b_rows) if row["variant"] == variant]
        if len(indices) != 5:
            raise RuntimeError(f"{variant}: expected five test samples")
        b_group, e_group = [b_rows[i] for i in indices], [e_rows[i] for i in indices]
        aggregate_rows.extend((_arm_aggregate(b_group, ARM_B, variant), _arm_aggregate(e_group, ARM_E, variant)))
        paired_rows.extend(_paired_wins(b_group, e_group, variant))
    for group, variants in GROUPS.items():
        indices = [i for i, row in enumerate(b_rows) if row["variant"] in variants]
        b_group, e_group = [b_rows[i] for i in indices], [e_rows[i] for i in indices]
        aggregate_rows.extend((_arm_aggregate(b_group, ARM_B, group), _arm_aggregate(e_group, ARM_E, group)))
        paired_rows.extend(_paired_wins(b_group, e_group, group))

    severity_order = np.argsort([float(row["gt_displacement_rms"]) for row in spectral_rows], kind="stable")
    severity_bins = {}
    for label, indices in zip(("low", "medium", "high"), np.array_split(severity_order, 3), strict=True):
        for index in indices:
            severity_bins[int(index)] = label
    severity_aggregate: list[dict[str, Any]] = []
    severity_paired: list[dict[str, Any]] = []
    for label in ("low", "medium", "high"):
        indices = [index for index in range(50) if severity_bins[index] == label]
        b_group, e_group = [b_rows[i] for i in indices], [e_rows[i] for i in indices]
        severity_aggregate.extend((_arm_aggregate(b_group, ARM_B, label), _arm_aggregate(e_group, ARM_E, label)))
        severity_paired.extend(_paired_wins(b_group, e_group, label))
    for index, row in enumerate(spectral_rows):
        row["severity_bin"] = severity_bins[index]
        row["initial_chamfer"] = float(b_rows[index]["initial_chamfer"])
        row["cd_e_minus_b"] = float(e_rows[index]["refined_chamfer"]) - float(b_rows[index]["refined_chamfer"])
        row["vrms_e_minus_b"] = float(e_rows[index]["same_index_recovered_vertex_rms"]) - float(b_rows[index]["same_index_recovered_vertex_rms"])
        row["normal_e_minus_b"] = float(e_rows[index]["normal_consistency"]) - float(b_rows[index]["normal_consistency"])

    correlations = []
    for source in (
        "gt_displacement_rms",
        "gt_displacement_mean",
        "gt_displacement_p95",
        "gt_displacement_max",
        "b_predicted_displacement_rms",
        "e_predicted_displacement_rms",
        "initial_chamfer",
    ):
        x = np.asarray([float(row[source]) for row in spectral_rows])
        for outcome in ("cd_e_minus_b", "vrms_e_minus_b", "normal_e_minus_b"):
            y = np.asarray([float(row[outcome]) for row in spectral_rows])
            correlations.append(_correlation(x, y, source, outcome))

    paired_statistics = _paired_statistics(b_rows, e_rows)
    spectral_aggregate = _spectral_aggregate(spectral_rows)
    representatives = _representatives(b_rows, e_rows)
    implementation_audit = bool(
        all(payload["read_only"] for payload in payloads)
        and len({payload["arm_b_checkpoint_sha256"] for payload in payloads}) == 1
        and len({payload["arm_e_checkpoint_sha256"] for payload in payloads}) == 1
        and all(row["b_recovery_reproduced"] and row["e_recovery_reproduced"] for row in spectral_rows)
        and b_payload["parameter_count"] == e_payload["parameter_count"]
    )
    summary = {
        "implementation_audit": implementation_audit,
        "read_only": True,
        "model_retrained": False,
        "arm_b_checkpoint": b_payload["checkpoint"],
        "arm_b_checkpoint_sha256": b_payload["checkpoint_sha256"],
        "arm_e_checkpoint": e_payload["checkpoint"],
        "arm_e_checkpoint_sha256": e_payload["checkpoint_sha256"],
        "parameter_count": int(b_payload["parameter_count"]),
        "metric_protocol": METRIC_PROTOCOL,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "spectral_chebyshev_order": args.chebyshev_order,
        "aggregate": aggregate_rows,
        "paired_wins": paired_rows,
        "severity_aggregate": severity_aggregate,
        "severity_paired_wins": severity_paired,
        "correlations": correlations,
        "paired_statistics": paired_statistics,
        "spectral_aggregate": spectral_aggregate,
        "representatives": representatives,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "representative_selection.json", representatives)
    _write_csv(output / "recipe_and_group_aggregate.csv", aggregate_rows)
    _write_csv(output / "recipe_and_group_paired_wins.csv", paired_rows)
    _write_csv(output / "severity_aggregate.csv", severity_aggregate)
    _write_csv(output / "severity_paired_wins.csv", severity_paired)
    _write_csv(output / "severity_and_spectral_per_sample.csv", spectral_rows)
    _write_csv(output / "severity_correlations.csv", correlations)
    _write_csv(output / "paired_statistics.csv", paired_statistics)
    _write_csv(output / "spectral_aggregate.csv", spectral_aggregate)

    recipe_lookup = {(row["group"], row["arm"]): row for row in aggregate_rows}
    lines = [
        "# Sofa50 frozen Arm B vs Arm E matched-domain diagnostic",
        "",
        f"Implementation/read-only audit: **{str(implementation_audit).lower()}**. No model was retrained.",
        "",
        "## A1-D2 recipe breakdown",
        "",
        "| Recipe | Arm | Initial CD | Refined CD | Mean gain | Vertex RMS | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for arm in (ARM_B, ARM_E):
            row = recipe_lookup[(variant, arm)]
            lines.append(
                f"| {variant} | {arm} | {row['initial_chamfer']:.8g} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:.2%} | {row['same_index_recovered_vertex_rms']:.8g} | {row['p2s_p95']:.8g} | {row['fscore']:.8g} | {row['normal_consistency']:.8g} | {row['introduced_flipped_faces']} / {row['normalized_flip_rate']:.3%} | {row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} |"
            )
    lines.extend((
        "",
        "Full per-recipe paired wins for every requested metric are in `recipe_and_group_paired_wins.csv`.",
        "",
        "## Mild/strong and topology-family summary",
        "",
        "| Group | Arm | Refined CD | Mean gain | Vertex RMS | P2S p95 | Normal | Flip rate | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for group in ("mild", "strong", "original_topology", "global_midpoint", "adaptive_topology", "all"):
        for arm in (ARM_B, ARM_E):
            row = recipe_lookup[(group, arm)]
            lines.append(
                f"| {group} | {arm} | {row['refined_chamfer']:.8g} | {row['relative_chamfer_gain']:.2%} | {row['same_index_recovered_vertex_rms']:.8g} | {row['p2s_p95']:.8g} | {row['normal_consistency']:.8g} | {row['normalized_flip_rate']:.3%} | {row['improved']}/{row['worsened']} |"
            )
    lines.extend((
        "",
        "## Paired statistics",
        "",
        "| Quantity | Mean difference | Median | Bootstrap 95% CI | Wilcoxon p |",
        "|---|---:|---:|---:|---:|",
    ))
    for row in paired_statistics:
        lines.append(
            f"| {row['quantity']} | {row['mean_paired_difference']:+.8g} | {row['median_paired_difference']:+.8g} | [{row['bootstrap_ci95_low']:+.8g}, {row['bootstrap_ci95_high']:+.8g}] | {row['wilcoxon_p']:.6g} |"
        )
    lines.extend((
        "",
        "## Graph-frequency analysis",
        "",
        SPECTRAL_PROTOCOL + ".",
        "",
        "| Signal | Total energy | Mean/vertex | Low fraction | Mid fraction | High fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for row in spectral_aggregate:
        lines.append(
            f"| {row['signal']} | {row['total_energy']:.8g} | {row['mean_energy_per_vertex']:.8g} | {row['low_fraction']:.2%} | {row['mid_fraction']:.2%} | {row['high_fraction']:.2%} |"
        )
    lines.extend((
        "",
        "Correction severity uses equal-count rank bins of GT displacement RMS and is diagnostic-only; it is never provided to either predictor. Full bin metrics and GT-free proxy correlations are in the adjacent CSV files.",
        "",
        "The final R1/R2/R3/R4 classification is deferred until the separately contract-gated frozen OOD evaluation is merged.",
        "",
    ))
    (output / "MATCHED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"implementation_audit": implementation_audit, "samples": 50}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ad-report-dir", required=True, type=Path)
    parser.add_argument("--ae-report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index")
    if args.merge_only:
        merge(args)
    else:
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

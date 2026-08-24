#!/usr/bin/env python3
from __future__ import annotations

"""Exact sparse Laplacian sanity check and invisible-equation weight sweep."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Mesh
from mlr.learned_laplacian.evaluation import _reconstruct
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.refinement import RefinementConfig


ALPHAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
EXACT_STATES = (
    "stored_target_clean_gauge",
    "float64_target_clean_gauge",
    "float64_target_initial_gauge",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_sparse_laplacian(faces: np.ndarray, num_vertices: int) -> tuple[csr_matrix, Any]:
    data = build_uniform_laplacian_data(faces, num_vertices)
    active = np.fromiter((bool(neighbors) for neighbors in data.neighbors), dtype=bool)
    diagonal = np.flatnonzero(active)
    rows = np.concatenate((diagonal, data.rows))
    cols = np.concatenate((diagonal, data.cols))
    values = np.concatenate((np.ones(len(diagonal)), -data.weights))
    matrix = coo_matrix((values, (rows, cols)), shape=(num_vertices, num_vertices)).tocsr()
    return matrix, data


def component_labels(data: Any) -> tuple[int, np.ndarray]:
    adjacency = coo_matrix(
        (np.ones(len(data.rows), dtype=np.float64), (data.rows, data.cols)),
        shape=(data.num_vertices, data.num_vertices),
    ).tocsr()
    return connected_components(adjacency, directed=False, return_labels=True)


def component_centroids(vertices: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    centroids = np.zeros((count, 3), dtype=np.float64)
    sizes = np.bincount(labels, minlength=count).astype(np.float64)
    for axis in range(3):
        centroids[:, axis] = np.bincount(
            labels, weights=vertices[:, axis], minlength=count
        ) / sizes
    return centroids


def component_constraint(labels: np.ndarray, count: int) -> csr_matrix:
    sizes = np.bincount(labels, minlength=count).astype(np.float64)
    rows = labels
    cols = np.arange(len(labels), dtype=np.int64)
    values = 1.0 / np.sqrt(sizes[labels])
    return coo_matrix((values, (rows, cols)), shape=(count, len(labels))).tocsr()


def exact_sparse_solve(
    laplacian: csr_matrix,
    target: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    gauge_centroids: np.ndarray,
    *,
    atol: float,
    btol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    constraint = component_constraint(labels, component_count)
    sizes = np.bincount(labels, minlength=component_count).astype(np.float64)
    system = vstack((laplacian, constraint), format="csr")
    gauge_rhs = gauge_centroids * np.sqrt(sizes)[:, None]
    rhs = np.vstack((np.asarray(target, dtype=np.float64), gauge_rhs))
    solution = np.empty_like(target, dtype=np.float64)
    axes: list[dict[str, Any]] = []
    for axis in range(3):
        result = lsmr(
            system,
            rhs[:, axis],
            atol=atol,
            btol=btol,
            conlim=1e12,
            maxiter=maxiter,
        )
        solution[:, axis] = result[0]
        axes.append(
            {
                "axis": axis,
                "istop": int(result[1]),
                "iterations": int(result[2]),
                "norm_residual": float(result[3]),
                "norm_normal_residual": float(result[4]),
                "operator_norm": float(result[5]),
                "condition_estimate": float(result[6]),
                "solution_norm": float(result[7]),
            }
        )
    return solution, {
        "axes": axes,
        "maximum_iterations": max(row["iterations"] for row in axes),
        "all_converged": all(row["istop"] in (1, 2, 4, 5) for row in axes),
    }


def shift_component_gauge(
    vertices: np.ndarray,
    labels: np.ndarray,
    source_centroids: np.ndarray,
    destination_centroids: np.ndarray,
) -> np.ndarray:
    return vertices + (destination_centroids - source_centroids)[labels]


def _residual_stats(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "residual_rms": float(np.sqrt(np.mean(norms**2))),
        "residual_max": float(norms.max(initial=0.0)),
        "residual_component_max_abs": float(np.max(np.abs(values), initial=0.0)),
    }


def _vertex_error(vertices: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(vertices - clean, axis=1)
    return {
        "vertex_rms_to_clean": float(np.sqrt(np.mean(error**2))),
        "vertex_mean_to_clean": float(error.mean()),
        "vertex_max_to_clean": float(error.max(initial=0.0)),
    }


def _load_reference(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    rows: dict[tuple[str, str], float] = {}
    with path.resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(str(row["sample_id"]), str(row["arm"]))] = float(row["chamfer"])
    return rows


def _sweep_config() -> RefinementConfig:
    return RefinementConfig(
        operator_type="uniform",
        lambda_lap=1.0,
        lambda_anchor=0.01,
        lambda_edge=0.0,
        lambda_unseen_anchor=0.0,
        num_iters=200,
        learning_rate=0.01,
        robust_loss="l2",
        huber_delta=0.01,
    )


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    reference = _load_reference(args.reference_ablation_csv)
    exact_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for index in range(args.shard_index, len(dataset), args.shard_count):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        stored_target = torch.as_tensor(static["raw_laplacian_target"]).cpu().numpy().astype(np.float64)
        laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        float64_target = apply_uniform_laplacian(clean.vertices, lap_data)
        component_count, labels = component_labels(lap_data)
        clean_centroids = component_centroids(clean.vertices, labels, component_count)
        initial_centroids = component_centroids(initial.vertices, labels, component_count)
        stored_solution, stored_solver = exact_sparse_solve(
            laplacian,
            stored_target,
            labels,
            component_count,
            clean_centroids,
            atol=args.lsmr_atol,
            btol=args.lsmr_btol,
            maxiter=args.lsmr_maxiter,
        )
        float64_solution, float64_solver = exact_sparse_solve(
            laplacian,
            float64_target,
            labels,
            component_count,
            clean_centroids,
            atol=args.lsmr_atol,
            btol=args.lsmr_btol,
            maxiter=args.lsmr_maxiter,
        )
        initial_gauge_solution = shift_component_gauge(
            float64_solution, labels, clean_centroids, initial_centroids
        )
        baseline = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in (("initial", initial), ("clean", clean))
        }
        initial_cd = float(baseline["initial"]["chamfer"])
        clean_cd = float(baseline["clean"]["chamfer"])
        available = initial_cd - clean_cd
        exact_meshes = {
            "stored_target_clean_gauge": Mesh(stored_solution, initial.faces.copy()).ensure_normals(),
            "float64_target_clean_gauge": Mesh(float64_solution, initial.faces.copy()).ensure_normals(),
            "float64_target_initial_gauge": Mesh(initial_gauge_solution, initial.faces.copy()).ensure_normals(),
        }
        targets = {
            "stored_target_clean_gauge": stored_target,
            "float64_target_clean_gauge": float64_target,
            "float64_target_initial_gauge": float64_target,
        }
        solver_audits = {
            "stored_target_clean_gauge": stored_solver,
            "float64_target_clean_gauge": float64_solver,
            "float64_target_initial_gauge": float64_solver,
        }
        for state in EXACT_STATES:
            mesh = exact_meshes[state]
            geometry = _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            equation = laplacian @ mesh.vertices - targets[state]
            exact_rows.append(
                {
                    **geometry,
                    "state": state,
                    "variant": metadata.get("variant"),
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "connected_components": component_count,
                    "initial_chamfer": initial_cd,
                    "clean_chamfer": clean_cd,
                    "eta_recovery": (initial_cd - float(geometry["chamfer"])) / available,
                    **_residual_stats(equation),
                    **_vertex_error(mesh.vertices, clean.vertices),
                    "lsmr_max_iterations": int(solver_audits[state]["maximum_iterations"]),
                    "lsmr_all_converged": bool(solver_audits[state]["all_converged"]),
                }
            )

        visibility = torch.as_tensor(static["visibility_backface_and_occlusion"], dtype=torch.bool)
        if visibility.ndim != 2 or visibility.shape[1] != initial.num_vertices:
            raise RuntimeError(f"Unexpected visibility shape for {sample_id}: {tuple(visibility.shape)}")
        visible = visibility.any(dim=0).cpu().numpy()
        alpha_reference_errors: list[float] = []
        for alpha in ALPHAS:
            weight = np.where(visible, 1.0, alpha).astype(np.float64)
            result, solver_name = _reconstruct(
                initial,
                stored_target,
                np.ones(initial.num_vertices, dtype=np.float64),
                _sweep_config(),
                args.dense_vertex_limit,
                laplacian_weight=weight,
            )
            geometry = _geometry_row(
                args.dataset_arm, sample_id, f"alpha_{alpha:g}", result.mesh, clean, initial
            )
            reference_arm = "plus_visibility" if alpha == 0 else "plus_anchor" if alpha == 1 else None
            reference_error = float("nan")
            if reference_arm is not None and reference:
                reference_error = abs(float(geometry["chamfer"]) - reference[(sample_id, reference_arm)])
                alpha_reference_errors.append(reference_error)
            sweep_rows.append(
                {
                    **geometry,
                    "alpha": alpha,
                    "variant": metadata.get("variant"),
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "visible_fraction": float(visible.mean()),
                    "initial_chamfer": initial_cd,
                    "clean_chamfer": clean_cd,
                    "eta_recovery": (initial_cd - float(geometry["chamfer"])) / available,
                    "solver_name": solver_name,
                    "reference_arm": reference_arm or "",
                    "reference_chamfer_abs_error": reference_error,
                }
            )

        clean_equation = laplacian @ clean.vertices
        centroid_shift = np.linalg.norm(initial_centroids - clean_centroids, axis=1)
        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "connected_components": component_count,
            "stored_target_vs_float64_max_abs_error": float(
                np.max(np.abs(stored_target - float64_target), initial=0.0)
            ),
            "clean_equation_vs_stored_target_max_abs_error": float(
                np.max(np.abs(clean_equation - stored_target), initial=0.0)
            ),
            "clean_equation_vs_float64_target_max_abs_error": float(
                np.max(np.abs(clean_equation - float64_target), initial=0.0)
            ),
            "component_centroid_shift_mean": float(centroid_shift.mean()),
            "component_centroid_shift_max": float(centroid_shift.max(initial=0.0)),
            "stored_lsmr": stored_solver,
            "float64_lsmr": float64_solver,
            "maximum_alpha_endpoint_reference_chamfer_error": max(alpha_reference_errors, default=float("nan")),
            "same_exact_target_all_alpha": True,
            "same_anchor_l2_iterations_lr_all_alpha": True,
            "only_invisible_equation_weight_changed": True,
        }
        audit["passed"] = bool(
            audit["clean_equation_vs_float64_target_max_abs_error"] <= 1e-12
            and stored_solver["all_converged"]
            and float64_solver["all_converged"]
            and (
                not np.isfinite(audit["maximum_alpha_endpoint_reference_chamfer_error"])
                or audit["maximum_alpha_endpoint_reference_chamfer_error"] <= args.reference_tolerance
            )
        )
        audits.append(audit)
        print(
            f"{args.dataset_arm} {sample_id}: components={component_count} "
            f"exact_rms={exact_rows[-2]['vertex_rms_to_clean']:.3g} "
            f"eta0={sweep_rows[-6]['eta_recovery']:.4g} eta1={sweep_rows[-1]['eta_recovery']:.4g} "
            f"audit={audit['passed']}",
            flush=True,
        )

    payload = {
        "dataset_arm": args.dataset_arm,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "metric_protocol": METRIC_PROTOCOL,
        "alphas": list(ALPHAS),
        "exact_rows": exact_rows,
        "sweep_rows": sweep_rows,
        "audits": audits,
    }
    _write_json(output / "shards" / f"shard_{args.shard_index:02d}.json", payload)


def _aggregate(rows: Sequence[Mapping[str, Any]], group: str, value: Any) -> dict[str, Any]:
    selected = [row for row in rows if row[group] == value]
    eta = np.asarray([float(row["eta_recovery"]) for row in selected])
    return {
        group: value,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "p2s": float(np.mean([float(row["p2s"]) for row in selected])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
        "normal_consistency": float(np.mean([float(row["normal_consistency"]) for row in selected])),
        "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
        "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
        "improved_over_initial": int(sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in selected)),
        "eta_mean": float(eta.mean()),
        "eta_median": float(np.median(eta)),
        "eta_p10": float(np.quantile(eta, 0.1)),
        "eta_p90": float(np.quantile(eta, 0.9)),
        "eta_negative_count": int((eta < 0).sum()),
        "vertex_rms_to_clean_mean": float(np.mean([float(row.get("vertex_rms_to_clean", float("nan"))) for row in selected])),
        "vertex_max_to_clean_max": max((float(row.get("vertex_max_to_clean", float("nan"))) for row in selected), default=float("nan")),
        "equation_residual_rms_mean": float(np.mean([float(row.get("residual_rms", float("nan"))) for row in selected])),
        "equation_residual_max_max": max((float(row.get("residual_max", float("nan"))) for row in selected), default=float("nan")),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shards = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    exact = [row for shard in shards for row in shard["exact_rows"]]
    sweep = [row for shard in shards for row in shard["sweep_rows"]]
    audits = [row for shard in shards for row in shard["audits"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    if len(exact) != expected * len(EXACT_STATES) or len(sweep) != expected * len(ALPHAS):
        raise RuntimeError("Incomplete exact-solve/visibility-sweep shards.")
    if not all(bool(row["passed"]) for row in audits):
        raise RuntimeError("Contract audit failed.")
    exact_aggregate = [{"dataset_arm": args.dataset_arm, **_aggregate(exact, "state", state)} for state in EXACT_STATES]
    sweep_aggregate = [{"dataset_arm": args.dataset_arm, **_aggregate(sweep, "alpha", alpha)} for alpha in ALPHAS]
    eta_means = np.asarray([float(row["eta_mean"]) for row in sweep_aggregate])
    eta_medians = np.asarray([float(row["eta_median"]) for row in sweep_aggregate])
    by_sample = {
        sample_id: [next(float(row["eta_recovery"]) for row in sweep if row["sample_id"] == sample_id and float(row["alpha"]) == alpha) for alpha in ALPHAS]
        for sample_id in sorted({str(row["sample_id"]) for row in sweep})
    }
    per_sample_monotonic = {
        sample_id: bool(np.all(np.diff(values) >= -1e-12)) for sample_id, values in by_sample.items()
    }
    summary = {
        "dataset_arm": args.dataset_arm,
        "contract_audit": True,
        "test_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "exact_solve": exact_aggregate,
        "visibility_sweep": sweep_aggregate,
        "monotonicity": {
            "mean_eta_non_decreasing_with_alpha": bool(np.all(np.diff(eta_means) >= -1e-12)),
            "median_eta_non_decreasing_with_alpha": bool(np.all(np.diff(eta_medians) >= -1e-12)),
            "per_sample_eta_non_decreasing_count": int(sum(per_sample_monotonic.values())),
            "per_sample_eta_non_decreasing_total": len(per_sample_monotonic),
            "mean_eta_alpha0": float(eta_means[0]),
            "mean_eta_alpha1": float(eta_means[-1]),
            "mean_eta_gain_alpha0_to_1": float(eta_means[-1] - eta_means[0]),
        },
        "connected_components": {
            "minimum": min(int(row["connected_components"]) for row in audits),
            "maximum": max(int(row["connected_components"]) for row in audits),
        },
        "target_quantization": {
            "maximum_stored_vs_float64_error": max(float(row["stored_target_vs_float64_max_abs_error"]) for row in audits),
        },
        "maximum_alpha_endpoint_reference_chamfer_error": max(
            (float(row["maximum_alpha_endpoint_reference_chamfer_error"]) for row in audits if np.isfinite(float(row["maximum_alpha_endpoint_reference_chamfer_error"]))),
            default=float("nan"),
        ),
    }
    contract = {
        "passed": True,
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(audits),
        "same_exact_target_anchor_l2_iterations_lr": True,
        "only_invisible_equation_weight_changed_in_sweep": True,
        "exact_solve_uses_component_centroid_constraints_only_to_fix_nullspace": True,
        "manifest": shards[0]["manifest"],
        "manifest_sha256": shards[0]["manifest_sha256"],
        "metric_protocol": METRIC_PROTOCOL,
        "maximum_clean_equation_vs_float64_target_error": max(float(row["clean_equation_vs_float64_target_max_abs_error"]) for row in audits),
        "maximum_alpha_endpoint_reference_chamfer_error": summary["maximum_alpha_endpoint_reference_chamfer_error"],
    }
    _write_csv(output / "exact_solve_per_sample.csv", exact)
    _write_csv(output / "exact_solve_aggregate.csv", exact_aggregate)
    _write_csv(output / "visibility_sweep_per_sample.csv", sweep)
    _write_csv(output / "visibility_sweep_aggregate.csv", sweep_aggregate)
    _write_json(output / "per_sample_contract_audit.json", audits)
    _write_json(output / "contract_audit.json", contract)
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-ablation-csv", type=Path)
    parser.add_argument("--dense-vertex-limit", type=int, default=5000)
    parser.add_argument("--lsmr-atol", type=float, default=1e-12)
    parser.add_argument("--lsmr-btol", type=float, default=1e-12)
    parser.add_argument("--lsmr-maxiter", type=int, default=100000)
    parser.add_argument("--reference-tolerance", type=float, default=1e-12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

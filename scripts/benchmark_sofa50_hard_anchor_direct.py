#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark an undamped reduced-normal sparse factorization on real meshes."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import splu

from diagnose_sofa50_exact_solve_visibility_sweep import uniform_sparse_laplacian
from mlr.learned_laplacian.hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
    hard_anchor_sparse_recovery_lsmr,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    sizes = [int(torch.as_tensor(dataset.load_static(i)["vertices"]).shape[0]) for i in range(len(dataset))]
    ordered = sorted(range(len(dataset)), key=lambda i: sizes[i])
    rows = []
    for index in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
        static = dataset.load_static(index)
        initial = torch.as_tensor(static["vertices"]).numpy().astype(np.float64)
        target = torch.as_tensor(static["raw_laplacian_target"]).numpy().astype(np.float64)
        faces = torch.as_tensor(static["faces"]).numpy().astype(np.int64)
        laplacian, _ = uniform_sparse_laplacian(faces, len(initial))
        anchors = deterministic_component_anchor_indices(
            torch.as_tensor(static["edge_index"]), len(initial)
        ).numpy()
        mask = np.ones(len(initial), dtype=bool)
        mask[anchors] = False
        free = np.flatnonzero(mask)
        anchor_only = np.zeros_like(initial)
        anchor_only[anchors] = initial[anchors]
        reduced = laplacian[:, free].tocsr()
        normal = (reduced.T @ reduced).tocsc()
        rhs = reduced.T @ (target - laplacian @ anchor_only)
        started = time.perf_counter()
        factor = splu(
            normal,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        factor_seconds = time.perf_counter() - started
        started = time.perf_counter()
        free_solution = np.column_stack([factor.solve(rhs[:, axis]) for axis in range(3)])
        solve_seconds = time.perf_counter() - started
        solution = anchor_only.copy()
        solution[free] = free_solution
        reference, reference_audit = hard_anchor_sparse_recovery_lsmr(
            laplacian, target, initial, anchors, atol=1e-12, btol=1e-12, maxiter=100000
        )
        residual = laplacian @ solution - target
        rows.append(
            {
                "sample_id": str(static["sample_id"]),
                "vertices": len(initial),
                "anchors": len(anchors),
                "normal_nnz": int(normal.nnz),
                "factor_l_nnz": int(factor.L.nnz),
                "factor_u_nnz": int(factor.U.nnz),
                "factor_seconds": factor_seconds,
                "three_rhs_solve_seconds": solve_seconds,
                "direct_vs_lsmr_vertex_rms": float(
                    np.sqrt(np.mean(np.sum((solution - reference) ** 2, axis=1)))
                ),
                "laplacian_residual_rms": float(
                    np.sqrt(np.mean(np.sum(residual**2, axis=1)))
                ),
                "lsmr_converged": bool(reference_audit["all_converged"]),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "hidden_regularization": False,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        del factor, normal, reduced
    finite_fields = (
        "factor_seconds",
        "three_rhs_solve_seconds",
        "direct_vs_lsmr_vertex_rms",
        "laplacian_residual_rms",
        "peak_rss_mib",
    )
    result = {
        "rows": rows,
        "all_finite": all(
            np.isfinite(float(row[field])) for row in rows for field in finite_fields
        ),
        "all_reference_solves_converged": all(row["lsmr_converged"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

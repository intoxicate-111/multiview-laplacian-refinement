#!/usr/bin/env python3
from __future__ import annotations

"""Real-mesh forward/backward audit for the lambda=0 hard-anchor solver."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.learned_laplacian.hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
    differentiable_hard_anchor_sparse_recovery_with_audit,
    hard_anchor_sparse_recovery_lsmr,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    sizes = [int(torch.as_tensor(dataset.load_static(i)["vertices"]).shape[0]) for i in range(len(dataset))]
    ordered = sorted(range(len(dataset)), key=lambda index: sizes[index])
    selected_indices = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    spec = _load_spec(args.arm_b_run.resolve(), device)
    rows: list[dict[str, object]] = []
    for index in selected_indices:
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        values = _infer_recovery_arm(dataset, index, spec, device)
        prediction_cpu = values["prediction_raw"].numpy().astype(np.float64)
        initial_cpu = torch.as_tensor(static["vertices"]).numpy().astype(np.float64)
        clean_cpu = torch.as_tensor(static["clean_reference_vertices"]).numpy().astype(np.float64)
        edge_cpu = torch.as_tensor(static["edge_index"], dtype=torch.long)
        degree_cpu = torch.as_tensor(static["vertex_degree"], dtype=torch.float64)
        anchors_cpu = deterministic_component_anchor_indices(edge_cpu, len(initial_cpu))
        laplacian, lap_data = uniform_sparse_laplacian(
            torch.as_tensor(static["faces"]).numpy(), len(initial_cpu)
        )
        component_count, _ = component_labels(lap_data)
        if int(anchors_cpu.numel()) != int(component_count):
            raise RuntimeError(f"Component/anchor mismatch for {sample_id}.")
        prediction = torch.from_numpy(prediction_cpu).to(device).requires_grad_(True)
        initial = torch.from_numpy(initial_cpu).to(device)
        edge = edge_cpu.to(device)
        degree = degree_cpu.to(device)
        anchors = anchors_cpu.to(device)
        started = time.perf_counter()
        try:
            recovered, pcg = differentiable_hard_anchor_sparse_recovery_with_audit(
                prediction,
                initial,
                edge,
                degree,
                anchors,
                maximum_iterations=2048,
                tolerance=1e-4,
            )
            refine_loss = ((recovered - torch.from_numpy(clean_cpu).to(device)) ** 2).sum(-1).mean()
            refine_loss.backward()
            pcg_error = None
        except RuntimeError as error:
            pcg = None
            recovered = None
            refine_loss = None
            pcg_error = str(error)
        direct_seconds = time.perf_counter() - started
        lsmr_started = time.perf_counter()
        reference, lsmr_audit = hard_anchor_sparse_recovery_lsmr(
            laplacian,
            prediction_cpu,
            initial_cpu,
            anchors_cpu.numpy(),
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        lsmr_seconds = time.perf_counter() - lsmr_started
        if recovered is None:
            forward_rms = None
            anchor_error = None
            laplacian_residual_rms = None
            prediction_gradient_norm = None
            finite_gradient = False
        else:
            recovered_cpu = recovered.detach().cpu().numpy()
            forward_rms = float(
                np.sqrt(np.mean(np.sum((recovered_cpu - reference) ** 2, axis=1)))
            )
            anchor_error = float(
                np.max(
                    np.abs(
                        recovered_cpu[anchors_cpu.numpy()]
                        - initial_cpu[anchors_cpu.numpy()]
                    ),
                    initial=0.0,
                )
            )
            residual = laplacian @ recovered_cpu - prediction_cpu
            laplacian_residual_rms = float(
                np.sqrt(np.mean(np.sum(residual**2, axis=1)))
            )
            prediction_gradient_norm = float(torch.linalg.vector_norm(prediction.grad).cpu())
            finite_gradient = bool(
                torch.isfinite(prediction.grad).all() and prediction_gradient_norm > 0
            )
        rows.append(
            {
                "sample_id": sample_id,
                "vertices": len(initial_cpu),
                "faces": int(torch.as_tensor(static["faces"]).shape[0]),
                "connected_components": int(component_count),
                "hard_anchors": int(anchors_cpu.numel()),
                "direct_converged": bool(pcg is not None and pcg.converged),
                "direct_factorizations": None if pcg is None else int(pcg.iterations),
                "direct_relative_normal_residual": None if pcg is None else float(pcg.relative_residual),
                "direct_runtime_seconds": direct_seconds,
                "direct_error": pcg_error,
                "lsmr_converged": bool(lsmr_audit["all_converged"]),
                "lsmr_iterations": int(lsmr_audit["maximum_iterations"]),
                "lsmr_condition_estimate": float(lsmr_audit["maximum_condition_estimate"]),
                "lsmr_runtime_seconds": lsmr_seconds,
                "direct_vs_lsmr_vertex_rms": forward_rms,
                "anchor_max_abs_error": anchor_error,
                "laplacian_residual_rms": laplacian_residual_rms,
                "prediction_gradient_norm": prediction_gradient_norm,
                "finite_nonzero_prediction_gradient": finite_gradient,
                "hidden_regularization": False,
            }
        )
        del values
        torch.cuda.empty_cache()
    passed = all(
        bool(row["direct_converged"])
        and bool(row["lsmr_converged"])
        and bool(row["finite_nonzero_prediction_gradient"])
        and float(row["anchor_max_abs_error"]) == 0.0
        and float(row["direct_vs_lsmr_vertex_rms"]) <= 1e-5
        for row in rows
    )
    summary = {
        "passed": passed,
        "selection": "validation_min_median_max_vertex_count",
        "solver": "reduced_hard_anchor_undamped_normal_sparse_lu_float64",
        "reference": "reduced_column_scipy_lsmr_float64",
        "lambda": 0.0,
        "centroid_constraint": False,
        "soft_positional_penalty": False,
        "hidden_tikhonov_damping": False,
        "maximum_iterations": 2048,
        "tolerance": 1e-4,
        "samples": rows,
    }
    (output / "audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            "Hard-anchor solver preflight failed; Arm I must not start without diagnosis."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

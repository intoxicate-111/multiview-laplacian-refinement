#!/usr/bin/env python3
from __future__ import annotations

"""Read-only fixed-subset mechanism diagnostics for one stored joint checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from analyze_sofa50_frozen_vs_joint_mechanisms import (
    LAMBDA_B,
    _lap_semantic_row,
    _latent_row,
    _pcg,
    _position_semantic_row,
)
from analyze_sofa50_joint_gradient_interference import _one as gradient_one
from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _spectral_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import SPECTRAL_BANDS, SPECTRAL_PROTOCOL
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


METHODS = ("Joint_Lap", "Joint_Direct", "Joint_Hybrid")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", default="0,5,10,15,20,25,30,35,40,45")
    parser.add_argument("--chebyshev-order", type=int, default=128)
    args = parser.parse_args()

    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    run_config = _read(args.run.resolve() / "run_config.json")
    config = run_config.get("experiment_config", run_config)
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(args.checkpoint.resolve(), model, map_location=device)
    model.eval()
    if not model.hybrid_direct_head_enabled:
        raise RuntimeError("Expected a shared-backbone hybrid direct head.")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")

    latent_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    gradient_audits: list[dict[str, Any]] = []

    for progress, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            output = model(conditioned)
        direct_prediction = output.direct_vertex_displacement_prediction
        if direct_prediction is None:
            raise RuntimeError("Joint checkpoint has no direct prediction.")
        delta = output.predicted_laplacian.detach().double().cpu().numpy()
        displacement = direct_prediction.detach().double().cpu().numpy()
        direct = vertices + displacement

        laplacian, lap_data = uniform_sparse_laplacian(faces, len(vertices))
        component_count, labels = component_labels(lap_data)
        lap_vertices, lap_audit = regularized_sparse_solve(
            laplacian,
            delta,
            vertices,
            labels,
            component_count,
            LAMBDA_B,
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        if not lap_audit["all_converged"]:
            raise RuntimeError(f"{sample_id}: standalone Lap recovery failed")
        hybrid, hybrid_audit = _pcg(delta, direct, static, device)
        methods = {
            "Joint_Lap": lap_vertices,
            "Joint_Direct": direct,
            "Joint_Hybrid": hybrid,
        }
        for method, method_vertices in methods.items():
            metric = _geometry_row(
                "validation",
                sample_id,
                method,
                Mesh(method_vertices, faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            geometry_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_index": index,
                    "method": method,
                    "chamfer": float(metric["chamfer"]),
                    "p2s_p95": float(metric["p2s_p95"]),
                    "fscore": float(metric["fscore"]),
                    "normal": float(metric["normal_consistency"]),
                    "vertex_rms": float(
                        np.sqrt(np.mean(np.sum((method_vertices - clean.vertices) ** 2, axis=1)))
                    ),
                    "pcg_iterations": int(hybrid_audit["pcg_iterations"])
                    if method == "Joint_Hybrid"
                    else None,
                    "pcg_relative_residual": float(hybrid_audit["pcg_relative_residual"])
                    if method == "Joint_Hybrid"
                    else None,
                }
            )
        delta_gt = laplacian @ clean.vertices
        latent_rows.append(
            _latent_row("validation", sample_id, "Joint_Lap_Direct", delta, laplacian @ direct)
        )
        lap_rows.append(_lap_semantic_row("validation", sample_id, "Joint_Lap", delta, delta_gt))
        direct_rows.append(
            _position_semantic_row("validation", sample_id, "Joint_Direct", direct, clean.vertices)
        )
        errors = {method: values - clean.vertices for method, values in methods.items()}
        spectral_rows.extend(
            _spectral_row(
                "validation",
                sample_id,
                faces,
                {f"{method}_error": error for method, error in errors.items()},
                args.chebyshev_order,
            )
        )
        sample_gradients, gradient_audit = gradient_one(
            model, dataset, index, config, device, amp_enabled, amp_dtype, 0
        )
        gradient_rows.extend(sample_gradients)
        gradient_audits.append(gradient_audit)
        print(f"evolution {args.label} {progress}/{len(indices)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    geometry_aggregate = [
        {
            "method": method,
            "samples": len([row for row in geometry_rows if row["method"] == method]),
            "chamfer": _mean([row for row in geometry_rows if row["method"] == method], "chamfer"),
            "vertex_rms": _mean(
                [row for row in geometry_rows if row["method"] == method], "vertex_rms"
            ),
        }
        for method in METHODS
    ]
    spectral_aggregate = []
    for method in METHODS:
        selected = [row for row in spectral_rows if row["signal"] == f"{method}_error"]
        item: dict[str, Any] = {
            "method": method,
            "samples": len(selected),
            "total_energy": float(sum(float(row["total_energy"]) for row in selected)),
        }
        for band in SPECTRAL_BANDS:
            item[f"{band}_energy"] = float(sum(float(row[f"{band}_energy"]) for row in selected))
        spectral_aggregate.append(item)
    gradient_aggregate = []
    for layer in sorted({str(row["layer"]) for row in gradient_rows}):
        selected = [row for row in gradient_rows if row["layer"] == layer]
        gradient_aggregate.append(
            {
                "layer": layer,
                "samples": len(selected),
                "cosine_mean": _mean(selected, "cosine"),
                "cosine_median": float(np.median([row["cosine"] for row in selected])),
                "conflict_rate": float(np.mean([float(row["cosine"]) < 0 for row in selected])),
                "magnitude_ratio_mean": _mean(selected, "magnitude_ratio"),
                "alignment_ratio_mean": _mean(selected, "alignment_ratio"),
            }
        )
    payload = {
        "read_only": True,
        "label": args.label,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "indices": indices,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "all_solvers_converged": all(
            float(row["pcg_relative_residual"]) <= 1.05e-8 for row in gradient_audits
        ),
        "latent_aggregate": {
            field: _mean(latent_rows, field)
            for field in ("redundancy_rms", "relative_discrepancy", "cosine", "norm_ratio")
        },
        "lap_semantic_aggregate": {
            field: _mean(lap_rows, field)
            for field in ("raw_epe", "raw_rms", "raw_cosine", "top10_epe", "top1_epe")
        },
        "direct_semantic_aggregate": {
            field: _mean(direct_rows, field)
            for field in ("vertex_rms", "vertex_error_mean", "vertex_error_p95")
        },
        "geometry_aggregate": geometry_aggregate,
        "spectral_aggregate": spectral_aggregate,
        "gradient_aggregate": gradient_aggregate,
        "latent_rows": latent_rows,
        "lap_semantic_rows": lap_rows,
        "direct_semantic_rows": direct_rows,
        "geometry_rows": geometry_rows,
        "spectral_rows": spectral_rows,
        "gradient_rows": gradient_rows,
        "gradient_audits": gradient_audits,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"evolution_{args.label}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "samples": len(indices)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

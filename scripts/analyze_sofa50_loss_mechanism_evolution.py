#!/usr/bin/env python3
from __future__ import annotations

"""Read-only S0 loss-gradient evolution on fixed validation indices."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_sofa50_joint_gradient_interference import _one as shared_gradient_one
from analyze_sofa50_loss_mechanisms import _gradient_rows, _mse_vertices, _recover, _rhs_row
from diagnose_sofa50_exact_solve_visibility_sweep import uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_representation_b_vs_e import SPECTRAL_BANDS, SPECTRAL_PROTOCOL
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


LAMBDA = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--indices", default="0,5,10,15,20,25,30,35,40,45")
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    config_payload = _read(args.run.resolve() / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(args.checkpoint.resolve(), model, map_location=device)
    model.eval()
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    spectral_rows: list[dict[str, Any]] = []
    rhs_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for progress, index in enumerate(indices, 1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        faces = np.asarray(static["faces"], dtype=np.int64)
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        clean = _clean_mesh(static).vertices.astype(np.float64)
        lap, _ = uniform_sparse_laplacian(faces, len(vertices))
        delta_gt = np.asarray(lap @ clean, dtype=np.float64)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(conditioned)
        direct = output.direct_vertex_displacement_prediction
        if direct is None:
            raise RuntimeError("S0 checkpoint has no direct output")
        delta = output.predicted_laplacian.detach().double().requires_grad_(True)
        v_direct = (prepared.sample["vertices"].double() + direct.detach().double()).requires_grad_(True)
        clean_tensor = torch.as_tensor(clean, dtype=torch.float64, device=device)
        degree = prepared.sample["vertex_degree"].double()
        recovered = _recover(
            delta, v_direct, prepared.sample["edge_index"], degree,
            regularization=LAMBDA, maximum_iterations=MAXIMUM_ITERATIONS, tolerance=TOLERANCE,
        )
        loss = _mse_vertices(recovered, clean_tensor)
        g_lap, g_direct = torch.autograd.grad(loss, (delta, v_direct))
        gradients = {
            "g_S0_lap_delta": g_lap.detach().cpu().numpy(),
            "g_S0_direct_V": g_direct.detach().cpu().numpy(),
        }
        sample_spectral = _gradient_rows("validation", sample_id, faces, gradients, args.chebyshev_order)
        for row in sample_spectral:
            row["checkpoint"] = args.label
        spectral_rows.extend(sample_spectral)
        rhs = _rhs_row(
            "validation", sample_id, "S0", delta.detach(),
            torch.as_tensor(delta_gt, dtype=torch.float64, device=device), v_direct.detach(),
            clean_tensor, prepared.sample["edge_index"], degree,
        )
        rhs["checkpoint"] = args.label
        rhs_rows.append(rhs)
        sample_shared, sample_audit = shared_gradient_one(
            model, dataset, index, config, device, amp_enabled, amp_dtype, 0
        )
        for row in sample_shared:
            row["checkpoint"] = args.label
        sample_audit["checkpoint"] = args.label
        shared_rows.extend(sample_shared)
        audit_rows.append(sample_audit)
        print(f"loss-evolution {args.label} {progress}/{len(indices)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    spectral_aggregate = []
    for path in ("g_S0_lap_delta", "g_S0_direct_V"):
        selected = [row for row in spectral_rows if row["path"] == path]
        spectral_aggregate.append(
            {
                "checkpoint": args.label,
                "path": path,
                "samples": len(selected),
                "mean_gradient_norm": float(np.mean([row["gradient_norm"] for row in selected])),
                "mean_total_energy": float(np.mean([row["total_energy"] for row in selected])),
                **{f"mean_{band}_energy": float(np.mean([row[f"{band}_energy"] for row in selected])) for band in SPECTRAL_BANDS},
                **{f"mean_{band}_fraction": float(np.mean([row[f"{band}_fraction"] for row in selected])) for band in SPECTRAL_BANDS},
            }
        )
    shared_aggregate = []
    for layer in sorted({row["layer"] for row in shared_rows}):
        selected = [row for row in shared_rows if row["layer"] == layer]
        shared_aggregate.append(
            {
                "checkpoint": args.label,
                "layer": layer,
                "samples": len(selected),
                "mean_cosine": float(np.mean([row["cosine"] for row in selected])),
                "mean_lap_norm": float(np.mean([row["lap_norm"] for row in selected])),
                "mean_direct_norm": float(np.mean([row["direct_norm"] for row in selected])),
                "median_magnitude_ratio": float(np.median([row["magnitude_ratio"] for row in selected])),
            }
        )
    payload = {
        "read_only": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "label": args.label,
        "indices": indices,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "all_finite": all(row["all_finite"] for row in audit_rows),
        "maximum_pcg_relative_residual": max(row["pcg_relative_residual"] for row in audit_rows),
        "spectral_aggregate": spectral_aggregate,
        "shared_gradient_aggregate": shared_aggregate,
        "rhs_aggregate": {
            "mean_lap_rhs_norm": float(np.mean([row["lap_rhs_norm"] for row in rhs_rows])),
            "mean_direct_rhs_norm": float(np.mean([row["direct_rhs_norm"] for row in rhs_rows])),
            "mean_combined_rhs_norm": float(np.mean([row["combined_rhs_norm"] for row in rhs_rows])),
            "mean_rhs_cosine": float(np.mean([row["rhs_cosine"] for row in rhs_rows])),
            "median_rhs_cosine": float(np.median([row["rhs_cosine"] for row in rhs_rows])),
            "mean_cancellation_ratio": float(np.mean([row["cancellation_ratio"] for row in rhs_rows])),
            "median_cancellation_ratio": float(np.median([row["cancellation_ratio"] for row in rhs_rows])),
        },
        "spectral_rows": spectral_rows,
        "shared_gradient_rows": shared_rows,
        "rhs_rows": rhs_rows,
        "gradient_audits": audit_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"loss_evolution_{args.label}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "all_finite": payload["all_finite"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

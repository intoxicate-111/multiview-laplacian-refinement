#!/usr/bin/env python3
from __future__ import annotations

"""Empirical exact-operator transfer profiles for the Uniform hybrid solve."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from diagnose_sofa50_representation_b_vs_e import SPECTRAL_BANDS, SPECTRAL_PROTOCOL, spectral_band_components
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


LAMBDA = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _apply(
    signal: np.ndarray,
    static: Mapping[str, Any],
    device: torch.device,
    branch: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = torch.as_tensor(signal, dtype=torch.float64, device=device)
    zero = torch.zeros_like(value)
    delta, anchor = (value, zero) if branch == "S_delta" else (zero, value)
    output, audit = recovery_forward_audit(
        delta,
        anchor,
        torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
        torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device),
        regularization=LAMBDA,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=TOLERANCE,
    )
    if not audit.converged:
        raise RuntimeError(f"{static['sample_id']} {branch}: {audit}")
    return output.detach().cpu().numpy(), {
        "iterations": int(audit.iterations),
        "relative_residual": float(audit.relative_residual),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--indices", default="0,12,24,36,49")
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for index in indices:
        static = dataset.load_static(index)
        faces = np.asarray(static["faces"], dtype=np.int64)
        vertices = len(static["vertices"])
        rng = np.random.default_rng(7300 + index)
        for probe_index in range(args.probes):
            raw = rng.standard_normal((vertices, 3))
            bands, _ = spectral_band_components(raw, faces, order=args.chebyshev_order)
            for band in SPECTRAL_BANDS:
                signal = bands[band]
                signal_norm = float(np.linalg.norm(signal))
                signal = signal / max(signal_norm, 1e-30)
                for branch in ("S_delta", "S_direct"):
                    output, audit = _apply(signal, static, device, branch)
                    rows.append(
                        {
                            "sample_id": str(static["sample_id"]),
                            "sample_index": index,
                            "probe": probe_index,
                            "band": band,
                            "branch": branch,
                            "input_norm": 1.0,
                            "output_norm": float(np.linalg.norm(output)),
                            "empirical_gain": float(np.linalg.norm(output)),
                            **audit,
                        }
                    )
        print(f"transfer {index} {static['sample_id']}", flush=True)
    aggregate: list[dict[str, Any]] = []
    for branch in ("S_delta", "S_direct"):
        for band in SPECTRAL_BANDS:
            selected = [row for row in rows if row["branch"] == branch and row["band"] == band]
            values = np.asarray([row["empirical_gain"] for row in selected], dtype=np.float64)
            aggregate.append(
                {
                    "branch": branch,
                    "band": band,
                    "measurements": len(selected),
                    "mean_gain": float(values.mean()),
                    "standard_deviation_gain": float(values.std(ddof=1)),
                    "median_gain": float(np.median(values)),
                    "minimum_gain": float(values.min()),
                    "maximum_gain": float(values.max()),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "transfer_profile_rows.csv", rows)
    _write_csv(args.output_dir / "transfer_profile_aggregate.csv", aggregate)
    payload = {
        "contract_audit": all(row["relative_residual"] <= 1.05 * TOLERANCE for row in rows),
        "read_only": True,
        "operator": "exact random-walk L_U=I-D^-1A; no symmetry assumption",
        "maps": {
            "S_delta": "(L_U^T L_U + lambda I)^-1 L_U^T",
            "S_direct": "lambda (L_U^T L_U + lambda I)^-1",
        },
        "lambda": LAMBDA,
        "tolerance": TOLERANCE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "indices": indices,
        "probes_per_band": args.probes,
        "aggregate": aggregate,
    }
    (args.output_dir / "transfer_profile_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"contract_audit": payload["contract_audit"], "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

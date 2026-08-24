#!/usr/bin/env python3
from __future__ import annotations

"""Select beta from pilot predictions using only unified-v2 validation geometry."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


BETAS = (1e-4, 1e-3, 1e-2)
TAGS = ("1em4", "1em3", "1em2")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidate(args: argparse.Namespace) -> None:
    beta = BETAS[args.candidate_index]
    tag = TAGS[args.candidate_index]
    run = args.runs_root.resolve() / f"sofa50_v2_sparse_recovery_beta_pilot_{tag}_1k_seed7"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    config = run_config.get("experiment_config", run_config)
    if int(metrics["optimizer_steps"]) != 1000:
        raise RuntimeError(f"Incomplete beta pilot: {run}")
    configured = config["training"]["recovery_aware_geometry_loss"]
    if not configured["enabled"] or float(configured["beta"]) != beta:
        raise RuntimeError(f"Pilot beta contract mismatch: {run}")
    regularization = float(configured["lambda"])
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        prediction_path = run / "predictions" / "validation" / f"{sample_id}_raw_delta.npy"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        prediction = np.load(prediction_path).astype(np.float64)
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        recovered, solver = regularized_sparse_solve(
            laplacian,
            prediction,
            initial.vertices,
            labels,
            component_count,
            regularization,
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        initial_row = _geometry_row("v2_strong_smoothing", sample_id, "initial", initial, clean, initial)
        clean_row = _geometry_row("v2_strong_smoothing", sample_id, "clean", clean, clean, initial)
        recovered_row = _geometry_row(
            "v2_strong_smoothing", sample_id, f"beta_{beta:.0e}",
            Mesh(recovered, initial.faces.copy()).ensure_normals(), clean, initial,
        )
        initial_cd = float(initial_row["chamfer"])
        clean_cd = float(clean_row["chamfer"])
        chamfer = float(recovered_row["chamfer"])
        rows.append(
            {
                "split": "validation",
                "sample_id": sample_id,
                "beta": beta,
                "lambda": regularization,
                "chamfer": chamfer,
                "initial_chamfer": initial_cd,
                "relative_chamfer_gain": (initial_cd - chamfer) / initial_cd,
                "eta": (initial_cd - chamfer) / (initial_cd - clean_cd),
                "same_index_vertex_rms": float(
                    np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
                ),
                "p2s": float(recovered_row["p2s"]),
                "p2s_p95": float(recovered_row["p2s_p95"]),
                "fscore": float(recovered_row["fscore"]),
                "normal_consistency": float(recovered_row["normal_consistency"]),
                "introduced_flipped_faces": int(recovered_row["introduced_flipped_faces"]),
                "new_degenerate_faces": int(recovered_row["new_degenerate_faces"]),
                "lsmr_all_converged": bool(solver["all_converged"]),
                "solver_runtime_seconds": float(solver["runtime_seconds"]),
            }
        )
        print(f"beta={beta:.0e} validation {index + 1}/{len(dataset)} {sample_id}", flush=True)
    output = args.output_dir.resolve() / "candidates"
    _write_json(
        output / f"beta_{tag}.json",
        {
            "beta": beta,
            "lambda": regularization,
            "run": str(run),
            "optimizer_steps": metrics["optimizer_steps"],
            "split": "validation",
            "test_split_loaded": False,
            "rows": rows,
        },
    )


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [
        json.loads((output / "candidates" / f"beta_{tag}.json").read_text())
        for tag in TAGS
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    aggregate: list[dict[str, Any]] = []
    lambdas = {float(payload["lambda"]) for payload in payloads}
    for beta in BETAS:
        selected = [row for row in rows if float(row["beta"]) == beta]
        if len(selected) != 50:
            raise RuntimeError(f"Expected 50 validation rows for beta={beta}")
        aggregate.append(
            {
                "beta": beta,
                "samples": 50,
                "mean_chamfer": float(np.mean([row["chamfer"] for row in selected])),
                "mean_relative_chamfer_gain": float(np.mean([row["relative_chamfer_gain"] for row in selected])),
                "mean_eta": float(np.mean([row["eta"] for row in selected])),
                "mean_same_index_vertex_rms": float(np.mean([row["same_index_vertex_rms"] for row in selected])),
                "mean_normal_consistency": float(np.mean([row["normal_consistency"] for row in selected])),
                "introduced_flipped_faces": int(np.sum([row["introduced_flipped_faces"] for row in selected])),
                "new_degenerate_faces": int(np.sum([row["new_degenerate_faces"] for row in selected])),
                "improved": int(np.sum([row["chamfer"] < row["initial_chamfer"] for row in selected])),
                "worsened": int(np.sum([row["chamfer"] > row["initial_chamfer"] for row in selected])),
                "all_lsmr_converged": all(row["lsmr_all_converged"] for row in selected),
            }
        )
    chosen = min(
        aggregate,
        key=lambda row: (float(row["mean_chamfer"]), float(row["mean_same_index_vertex_rms"])),
    )
    contract = {
        "passed": bool(
            len(lambdas) == 1
            and len(rows) == 150
            and all(payload["split"] == "validation" and not payload["test_split_loaded"] for payload in payloads)
            and all(row["all_lsmr_converged"] for row in aggregate)
        ),
        "selection_split": "validation",
        "test_split_loaded": False,
        "beta_candidates_predeclared": list(BETAS),
        "selection_rule": "minimum mean unified-v2 validation Chamfer; same-index vertex RMS tie-break",
        "pilot_optimizer_steps": 1000,
        "each_pilot_from_scratch": True,
        "metric_protocol": METRIC_PROTOCOL,
    }
    summary = {
        "contract_audit": contract,
        "lambda": next(iter(lambdas)),
        "selected_beta": chosen["beta"],
        "selected_validation_metrics": chosen,
        "validation_curve": aggregate,
    }
    _write_csv(output / "validation_beta_per_sample.csv", rows)
    _write_csv(output / "validation_beta_curve.csv", aggregate)
    _write_json(output / "beta_selection.json", summary)
    lines = [
        "# Sofa50 v2 validation-only recovery-aware beta selection", "",
        f"Contract audit: **{str(contract['passed']).lower()}**.", "",
        "The test split was not loaded. All three 1k pilots started independently from seed 7.", "",
        "| Beta | Chamfer | Relative gain | Eta | Same-index RMS | Normal | Flips | Improved |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['beta']:.0e} | {row['mean_chamfer']:.9g} | {row['mean_relative_chamfer_gain']:.2%} | "
            f"{row['mean_eta']:.9g} | {row['mean_same_index_vertex_rms']:.9g} | "
            f"{row['mean_normal_consistency']:.9g} | {row['introduced_flipped_faces']} | {row['improved']}/50 |"
        )
    lines.extend(("", f"Selected beta: **{chosen['beta']:.0e}**.", ""))
    (output / "BETA_SELECTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int, choices=range(3))
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.manifest is None or args.runs_root is None or args.candidate_index is None:
            parser.error("candidate evaluation requires manifest, runs-root and candidate-index")
        evaluate_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

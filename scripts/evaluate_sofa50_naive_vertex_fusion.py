#!/usr/bin/env python3
from __future__ import annotations

"""Validation-selected scalar fusion of frozen Sofa50 B/E vertex outputs.

This is deliberately a vertex-space baseline:

    V_alpha = alpha * V_D + (1 - alpha) * V_P

The script has hard phase boundaries.  Test shards require a validation selection
lock and do not expose an alpha argument.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import (
    ARM_B,
    ARM_E,
    ARM_H,
    _inputs as matched_inputs,
    _pcg as matched_hybrid_pcg,
)
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_old_domain_specialists import pcg as old_pcg
from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ALPHAS = tuple(index / 100.0 for index in range(101))
METHOD_INITIAL = "Initial mesh"
METHOD_D = "Operator-Mediated Differential"
METHOD_P = "Direct Positional"
METHOD_NAIVE = "Naive scalar fusion"
METHOD_HYBRID = "Proposed operator Hybrid"
METHODS = (METHOD_INITIAL, METHOD_D, METHOD_P, METHOD_NAIVE, METHOD_HYBRID)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_cluster(sample_id: str) -> str:
    return sample_id.split("__", 1)[0]


def metric_row(
    split: str,
    sample_id: str,
    method: str,
    vertices: np.ndarray,
    initial: Mesh,
    clean: Mesh,
) -> dict[str, Any]:
    mesh = Mesh(np.asarray(vertices, dtype=np.float64), initial.faces.copy()).ensure_normals()
    metric = _geometry_row(split, sample_id, method, mesh, clean, initial)
    vrms = float(np.sqrt(np.mean(np.sum((mesh.vertices - clean.vertices) ** 2, axis=1))))
    return {
        "split": split,
        "sample_id": sample_id,
        "object_cluster": sample_cluster(sample_id),
        "method": method,
        "vertices": initial.num_vertices,
        "faces": initial.num_faces,
        "chamfer": float(metric["chamfer"]),
        "p2s": float(metric["p2s"]),
        "p2s_p95": float(metric["p2s_p95"]),
        "fscore": float(metric["fscore"]),
        "normal_consistency": float(metric["normal_consistency"]),
        "same_index_vertex_rms": vrms,
        "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
        "new_degenerate_faces": int(metric["new_degenerate_faces"]),
    }


def archive_metric(row: Mapping[str, Any], field: str = "refined_chamfer") -> float:
    return float(row[field])


def check_metric(label: str, actual: float, expected: float, tolerance: float) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise RuntimeError(
            f"{label}: archived metric mismatch: actual={actual:.17g}, "
            f"expected={expected:.17g}, tolerance={tolerance:g}"
        )


def selected_indices(length: int, count: int, index: int) -> list[int]:
    return list(range(index, length, count))


def preflight(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    target = output / "contract_audit.json"
    if target.exists() and not args.force:
        raise RuntimeError(f"Preflight output already exists: {target}")
    manifest = args.manifest.resolve()
    if args.domain == "matched_v2":
        b_summary = read_json(args.arm_b_report / "summary.json")
        e_summary = read_json(args.arm_e_report / "summary.json")
        hybrid = read_json(args.hybrid_summary)
        b_arrays = args.arm_b_report / "shards" / f"{ARM_B}_prediction_arrays.npz"
        e_arrays = args.arm_e_report / "shards" / f"{ARM_E}_prediction_arrays.npz"
        contract = {
            "domain": args.domain,
            "operator_mediated_checkpoint": hybrid["arm_b_checkpoint"],
            "operator_mediated_checkpoint_sha256": hybrid["arm_b_checkpoint_sha256"],
            "direct_positional_checkpoint": hybrid["arm_e_checkpoint"],
            "direct_positional_checkpoint_sha256": hybrid["arm_e_checkpoint_sha256"],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "validation_split": "validation (50 exact manifest samples)",
            "test_split": "test (50 exact manifest samples)",
            "standalone_differential_lambda": 0.01,
            "standalone_differential_anchor": "initial vertices V0",
            "proposed_hybrid_lambda": float(hybrid["lambda_hybrid_best"]),
            "proposed_hybrid_lambda_selection": hybrid["lambda_selection_metric"],
            "evaluator": METRIC_PROTOCOL,
            "cached_v_d_source": str(b_arrays.resolve()),
            "cached_v_p_source": str(e_arrays.resolve()),
            "cached_v_d": b_arrays.is_file(),
            "cached_v_p": e_arrays.is_file(),
            "source_contracts_pass": bool(
                b_summary["contract_audit"]["executable_contract_audit"]
                and e_summary["implementation_audit"]
                and hybrid["contract_audit"]
            ),
        }
    else:
        specialist = read_json(args.specialist_summary)
        hybrid = read_json(args.hybrid_summary)
        frozen = read_json(args.frozen_test_summary)
        contract = {
            "domain": args.domain,
            "operator_mediated_checkpoint": specialist["arm_b_checkpoint"],
            "operator_mediated_checkpoint_sha256": specialist["arm_b_checkpoint_sha256"],
            "direct_positional_checkpoint": specialist["arm_e_checkpoint"],
            "direct_positional_checkpoint_sha256": specialist["arm_e_checkpoint_sha256"],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "validation_split": "validation (25 exact manifest samples; five source objects)",
            "test_split": "test (25 exact manifest samples; five source objects)",
            "standalone_differential_lambda": 0.01,
            "standalone_differential_anchor": "initial vertices V0",
            "proposed_hybrid_lambda": float(hybrid["selected_lambda"]),
            "proposed_hybrid_lambda_selection": hybrid["selection_metric"],
            "evaluator": specialist["metric_protocol"],
            "cached_v_d_source": str(args.specialist_predictions.resolve()),
            "cached_v_p_source": str(args.specialist_predictions.resolve()),
            "cached_test_mesh_source": str(args.frozen_test_mesh_root.resolve()),
            "cached_v_d": args.specialist_predictions.is_file(),
            "cached_v_p": args.specialist_predictions.is_file(),
            "cached_test_meshes": args.frozen_test_mesh_root.is_dir(),
            "source_contracts_pass": bool(
                specialist["contract_audit"]
                and hybrid["contract_audit"]
                and frozen["contract_audit"]
            ),
        }
    contract.update(
        {
            "models_retrained": False,
            "alpha_grid": list(ALPHAS),
            "alpha_selection_split": "validation",
            "alpha_selection_metric": "macro mean unified surface Chamfer",
            "alpha_tie_break": "smallest alpha among exact float64 minima",
            "test_alpha_override_available": False,
        }
    )
    contract["contract_audit"] = bool(
        contract["source_contracts_pass"]
        and contract["cached_v_d"]
        and contract["cached_v_p"]
        and contract["proposed_hybrid_lambda_selection"]
        in {"mean refined Chamfer", "macro_mean_unified_surface_chamfer"}
    )
    write_json(target, contract)
    if not contract["contract_audit"]:
        raise RuntimeError("Naive-fusion preflight failed")
    print(json.dumps(contract, indent=2, sort_keys=True))


def matched_context(args: argparse.Namespace, split: str) -> dict[str, Any]:
    bundle = matched_inputs(args, split)
    dataset, _, _, b_rows, e_rows, b_array, e_array, b_starts, e_starts = bundle
    return {
        "dataset": dataset,
        "b_rows": b_rows,
        "e_rows": e_rows,
        "b_array": b_array,
        "e_array": e_array,
        "b_starts": b_starts,
        "e_starts": e_starts,
    }


def matched_vertices(
    context: Mapping[str, Any], index: int, *, include_hybrid: bool
) -> tuple[Mesh, Mesh, np.ndarray, np.ndarray, np.ndarray | None, dict[str, float]]:
    dataset = context["dataset"]
    static = dataset.load_static(index)
    initial = Mesh(
        np.asarray(static["vertices"], dtype=np.float64),
        np.asarray(static["faces"], dtype=np.int64),
    ).ensure_normals()
    clean = _clean_mesh(static)
    count = initial.num_vertices
    b_prediction = context["b_array"][
        context["b_starts"][index] : context["b_starts"][index] + count
    ]
    e_displacement = context["e_array"][
        context["e_starts"][index] : context["e_starts"][index] + count
    ]
    direct = initial.vertices + e_displacement
    lap, lap_data = uniform_sparse_laplacian(initial.faces, count)
    component_count, labels = component_labels(lap_data)
    differential, audit = regularized_sparse_solve(
        lap,
        b_prediction,
        initial.vertices,
        labels,
        component_count,
        0.01,
        atol=1e-12,
        btol=1e-12,
        maxiter=100000,
    )
    if not audit["all_converged"]:
        raise RuntimeError(f"{static['sample_id']}: differential LSMR failed")
    hybrid_vertices = None
    hybrid_residual = float("nan")
    if include_hybrid:
        hybrid_vertices, solver = matched_hybrid_pcg(
            b_prediction, direct, static, 0.03, torch.device(args_device(context))
        )
        if not solver["pcg_converged"]:
            raise RuntimeError(f"{static['sample_id']}: hybrid PCG failed")
        hybrid_residual = float(solver["pcg_relative_residual"])
    return (
        initial,
        clean,
        differential,
        direct,
        hybrid_vertices,
        {"hybrid_pcg_relative_residual": hybrid_residual},
    )


def args_device(context: Mapping[str, Any]) -> str:
    return str(context.get("device", "cpu"))


def old_validation_context(args: argparse.Namespace) -> dict[str, Any]:
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    archive = np.load(args.specialist_predictions.resolve())
    ids = archive["sample_ids"].tolist()
    if ids != list(dataset.sample_ids) or len(dataset) != 25:
        raise RuntimeError("Old-domain validation prediction IDs/order mismatch")
    rows = []
    with args.specialist_per_sample.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    b_rows = {row["sample_id"]: row for row in rows if row["arm"] == "B_recovery_aware"}
    e_rows = {row["sample_id"]: row for row in rows if row["arm"] == "E_direct_vertex"}
    return {
        "dataset": dataset,
        "archive": archive,
        "offsets": archive["offsets"].astype(np.int64),
        "b_rows": b_rows,
        "e_rows": e_rows,
        "device": args.device,
    }


def old_validation_vertices(
    context: Mapping[str, Any], index: int
) -> tuple[Mesh, Mesh, np.ndarray, np.ndarray]:
    dataset = context["dataset"]
    static = dataset.load_static(index)
    initial = Mesh(
        np.asarray(static["vertices"], dtype=np.float64),
        np.asarray(static["faces"], dtype=np.int64),
    ).ensure_normals()
    clean = _clean_mesh(static)
    start, stop = int(context["offsets"][index]), int(context["offsets"][index + 1])
    b_prediction = context["archive"]["b_prediction"][start:stop].astype(np.float64)
    direct = initial.vertices + context["archive"]["e_displacement"][start:stop].astype(np.float64)
    differential, audit = old_pcg(
        b_prediction, initial.vertices, static, 0.01, torch.device(context["device"])
    )
    if not audit["pcg_converged"] or float(audit["pcg_relative_residual"]) > 1.05e-8:
        raise RuntimeError(f"{static['sample_id']}: old-domain differential PCG failed")
    return initial, clean, differential, direct


def validation_shard(args: argparse.Namespace) -> None:
    contract = read_json(args.output_dir / "contract_audit.json")
    if not contract.get("contract_audit") or contract["domain"] != args.domain:
        raise RuntimeError("Missing or invalid preflight contract")
    target = args.output_dir / "shards" / f"validation_{args.shard_index:02d}.json"
    if target.exists() and not args.force:
        raise RuntimeError(f"Validation shard already exists: {target}")
    if args.domain == "matched_v2":
        context = matched_context(args, "validation")
        context["device"] = args.device
    else:
        context = old_validation_context(args)
    dataset = context["dataset"]
    rows: list[dict[str, Any]] = []
    endpoint_audits: list[dict[str, Any]] = []
    for progress, index in enumerate(
        selected_indices(len(dataset), args.shard_count, args.shard_index), start=1
    ):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        if args.domain == "matched_v2":
            initial, clean, differential, direct, _, _ = matched_vertices(
                context, index, include_hybrid=False
            )
            archived_b = context["b_rows"][index]
            archived_e = context["e_rows"][index]
            metric_tolerance = 2e-8
        else:
            initial, clean, differential, direct = old_validation_vertices(context, index)
            archived_b = context["b_rows"][sample_id]
            archived_e = context["e_rows"][sample_id]
            metric_tolerance = 2e-8
        alpha_zero = 0.0 * differential + 1.0 * direct
        alpha_one = 1.0 * differential + 0.0 * direct
        endpoint_audits.append(
            {
                "sample_id": sample_id,
                "alpha_0_max_abs_vs_v_p": float(np.max(np.abs(alpha_zero - direct))),
                "alpha_1_max_abs_vs_v_d": float(np.max(np.abs(alpha_one - differential))),
            }
        )
        for alpha in ALPHAS:
            vertices = alpha * differential + (1.0 - alpha) * direct
            row = metric_row("validation", sample_id, METHOD_NAIVE, vertices, initial, clean)
            row["alpha"] = alpha
            rows.append(row)
            if alpha == 0.0:
                check_metric(
                    f"{sample_id}/alpha=0",
                    row["chamfer"],
                    archive_metric(archived_e),
                    metric_tolerance,
                )
            elif alpha == 1.0:
                check_metric(
                    f"{sample_id}/alpha=1",
                    row["chamfer"],
                    archive_metric(archived_b),
                    metric_tolerance,
                )
        print(
            f"{args.domain} validation shard={args.shard_index} "
            f"{progress} {sample_id}",
            flush=True,
        )
    write_json(
        target,
        {
            "contract_audit": True,
            "domain": args.domain,
            "split": "validation",
            "test_accessed": False,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "alpha_grid": list(ALPHAS),
            "endpoint_audits": endpoint_audits,
            "rows": rows,
        },
    )


def aggregate_alpha(rows: Sequence[Mapping[str, Any]], alpha: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row["alpha"]) == alpha]
    faces = sum(int(row["faces"]) for row in selected)
    return {
        "alpha": alpha,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
        "fscore": float(np.mean([float(row["fscore"]) for row in selected])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in selected])
        ),
        "same_index_vertex_rms": float(
            np.mean([float(row["same_index_vertex_rms"]) for row in selected])
        ),
        "introduced_flipped_faces": int(
            sum(int(row["introduced_flipped_faces"]) for row in selected)
        ),
        "normalized_flip_rate": float(
            sum(int(row["introduced_flipped_faces"]) for row in selected) / faces
        ),
        "new_degenerate_faces": int(
            sum(int(row["new_degenerate_faces"]) for row in selected)
        ),
    }


def validation_plot(path: Path, aggregate: Sequence[Mapping[str, Any]], selected: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([float(row["alpha"]) for row in aggregate])
    y = np.asarray([float(row["chamfer"]) for row in aggregate])
    chosen = int(np.flatnonzero(x == selected)[0])
    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)
    ax.plot(x, y, color="#275D8C", linewidth=1.8)
    ax.scatter([selected], [y[chosen]], color="#B4413E", s=28, zorder=3)
    ax.annotate(
        rf"$\alpha^*={selected:.2f}$",
        (selected, y[chosen]),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel(r"Naive vertex blend $\alpha$")
    ax.set_ylabel("Validation CD")
    ax.grid(alpha=0.22, linewidth=0.6)
    fig.savefig(path, dpi=300, transparent=False, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), transparent=False, facecolor="white")
    plt.close(fig)


def merge_validation(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    lock_path = output / "selection_lock.json"
    if lock_path.exists() and not args.force:
        raise RuntimeError(f"Selection lock already exists: {lock_path}")
    payloads = [
        read_json(output / "shards" / f"validation_{index:02d}.json")
        for index in range(args.shard_count)
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    expected = 50 if args.domain == "matched_v2" else 25
    if len(sample_ids) != expected or len(rows) != expected * len(ALPHAS):
        raise RuntimeError(
            f"Expected {expected} samples and {expected * len(ALPHAS)} rows; "
            f"found {len(sample_ids)} and {len(rows)}"
        )
    if len({(row["sample_id"], float(row["alpha"])) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate validation sample/alpha row")
    aggregates = [aggregate_alpha(rows, alpha) for alpha in ALPHAS]
    selected_row = min(aggregates, key=lambda row: (float(row["chamfer"]), float(row["alpha"])))
    selected = float(selected_row["alpha"])
    endpoint = [audit for payload in payloads for audit in payload["endpoint_audits"]]
    endpoint_max = max(
        max(float(row["alpha_0_max_abs_vs_v_p"]), float(row["alpha_1_max_abs_vs_v_d"]))
        for row in endpoint
    )
    contract = read_json(output / "contract_audit.json")
    lock = {
        "contract_audit": bool(
            all(payload["contract_audit"] for payload in payloads)
            and not any(payload["test_accessed"] for payload in payloads)
            and endpoint_max == 0.0
            and contract["contract_audit"]
        ),
        "domain": args.domain,
        "selection_split": "validation",
        "test_accessed": False,
        "selection_metric": "macro mean unified surface Chamfer",
        "tie_break": "smallest alpha among exact float64 minima",
        "alpha_grid": list(ALPHAS),
        "selected_alpha": selected,
        "selected_validation_metrics": selected_row,
        "endpoint_max_abs_error": endpoint_max,
        "samples": expected,
        "operator_mediated_checkpoint_sha256": contract[
            "operator_mediated_checkpoint_sha256"
        ],
        "direct_positional_checkpoint_sha256": contract[
            "direct_positional_checkpoint_sha256"
        ],
        "manifest_sha256": contract["manifest_sha256"],
        "standalone_differential_lambda": contract["standalone_differential_lambda"],
        "proposed_hybrid_lambda": contract["proposed_hybrid_lambda"],
        "validation_sweep_csv_sha256": None,
    }
    write_csv(output / "validation_alpha_sweep.csv", aggregates)
    write_csv(output / "validation_alpha_sweep_per_mesh.csv", rows)
    lock["validation_sweep_csv_sha256"] = sha256_file(output / "validation_alpha_sweep.csv")
    write_json(lock_path, lock)
    validation_plot(output / "validation_alpha_cd.png", aggregates, selected)
    if not lock["contract_audit"]:
        raise RuntimeError("Validation selection contract failed")
    print(json.dumps(lock, indent=2, sort_keys=True))


def old_test_context(args: argparse.Namespace) -> dict[str, Any]:
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    frozen = read_json(args.frozen_test_summary)
    archived = {
        (str(row["sample_id"]), str(row["method"])): row
        for row in frozen["rows"]
        if str(row["method"]).startswith("Old-domain") or row["method"] == "Initial mesh"
    }
    return {"dataset": dataset, "archived": archived}


def old_test_vertices(
    args: argparse.Namespace, context: Mapping[str, Any], index: int
) -> tuple[Mesh, Mesh, np.ndarray, np.ndarray, np.ndarray]:
    static = context["dataset"].load_static(index)
    sample_id = str(static["sample_id"])
    initial = Mesh(
        np.asarray(static["vertices"], dtype=np.float64),
        np.asarray(static["faces"], dtype=np.int64),
    ).ensure_normals()
    clean = _clean_mesh(static)
    root = args.frozen_test_mesh_root / sample_id
    paths = {
        METHOD_D: root / "old-domain_arm_b.obj",
        METHOD_P: root / "old-domain_arm_e.obj",
        METHOD_HYBRID: root / "old-domain_frozen_bpluse.obj",
    }
    loaded = {method: load_mesh(path) for method, path in paths.items()}
    for method, mesh in loaded.items():
        if not np.array_equal(mesh.faces, initial.faces) or mesh.vertices.shape != initial.vertices.shape:
            raise RuntimeError(f"{sample_id}/{method}: cached mesh contract mismatch")
    return (
        initial,
        clean,
        loaded[METHOD_D].vertices.astype(np.float64),
        loaded[METHOD_P].vertices.astype(np.float64),
        loaded[METHOD_HYBRID].vertices.astype(np.float64),
    )


def test_shard(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    lock = read_json(output / "selection_lock.json")
    if not lock.get("contract_audit") or lock["domain"] != args.domain:
        raise RuntimeError("Missing or invalid validation selection lock")
    if sha256_file(output / "validation_alpha_sweep.csv") != lock["validation_sweep_csv_sha256"]:
        raise RuntimeError("Validation sweep changed after alpha lock")
    target = output / "shards" / f"test_{args.shard_index:02d}.json"
    if target.exists():
        raise RuntimeError(f"Test shard already exists; refusing a second test open: {target}")
    alpha = float(lock["selected_alpha"])
    if args.domain == "matched_v2":
        context = matched_context(args, "test")
        context["device"] = args.device
    else:
        context = old_test_context(args)
    dataset = context["dataset"]
    rows: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []
    for progress, index in enumerate(
        selected_indices(len(dataset), args.shard_count, args.shard_index), start=1
    ):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        if args.domain == "matched_v2":
            initial, clean, differential, direct, hybrid, solver = matched_vertices(
                context, index, include_hybrid=True
            )
            assert hybrid is not None
            archived = {
                METHOD_D: context["b_rows"][index],
                METHOD_P: context["e_rows"][index],
            }
            metric_tolerance = 2e-8
        else:
            initial, clean, differential, direct, hybrid = old_test_vertices(
                args, context, index
            )
            solver = {}
            archived = {
                METHOD_D: context["archived"][(sample_id, "Old-domain Arm B")],
                METHOD_P: context["archived"][(sample_id, "Old-domain Arm E")],
                METHOD_HYBRID: context["archived"][(sample_id, "Old-domain Frozen B+E")],
            }
            metric_tolerance = 2e-7
        naive = alpha * differential + (1.0 - alpha) * direct
        meshes = {
            METHOD_INITIAL: initial.vertices,
            METHOD_D: differential,
            METHOD_P: direct,
            METHOD_NAIVE: naive,
            METHOD_HYBRID: hybrid,
        }
        initial_cd = None
        for method in METHODS:
            row = metric_row("test", sample_id, method, meshes[method], initial, clean)
            if method == METHOD_INITIAL:
                initial_cd = float(row["chamfer"])
            row["alpha"] = alpha if method == METHOD_NAIVE else None
            rows.append(row)
            if method in archived:
                check_metric(
                    f"{sample_id}/{method}",
                    row["chamfer"],
                    archive_metric(archived[method], "chamfer" if args.domain != "matched_v2" else "refined_chamfer"),
                    metric_tolerance,
                )
        assert initial_cd is not None
        for row in rows[-len(METHODS) :]:
            row["initial_chamfer"] = initial_cd
            row["improved"] = float(row["chamfer"]) < initial_cd
            row["worsened"] = float(row["chamfer"]) > initial_cd
        reproduction.append({"sample_id": sample_id, **solver})
        print(
            f"{args.domain} frozen test shard={args.shard_index} "
            f"{progress} {sample_id}",
            flush=True,
        )
    write_json(
        target,
        {
            "contract_audit": True,
            "domain": args.domain,
            "split": "test",
            "test_opened_once": True,
            "selected_alpha": alpha,
            "selection_lock_sha256": sha256_file(output / "selection_lock.json"),
            "rows": rows,
            "reproduction": reproduction,
        },
    )


def aggregate_method(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    faces = sum(int(row["faces"]) for row in selected)
    return {
        "method": method,
        "samples": len(selected),
        "chamfer": float(np.mean([float(row["chamfer"]) for row in selected])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
        "fscore": float(np.mean([float(row["fscore"]) for row in selected])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in selected])
        ),
        "same_index_vertex_rms": float(
            np.mean([float(row["same_index_vertex_rms"]) for row in selected])
        ),
        "improved": int(sum(bool(row["improved"]) for row in selected)),
        "worsened": int(sum(bool(row["worsened"]) for row in selected)),
        "introduced_flipped_faces": int(
            sum(int(row["introduced_flipped_faces"]) for row in selected)
        ),
        "normalized_flip_rate": float(
            sum(int(row["introduced_flipped_faces"]) for row in selected) / faces
        ),
    }


def paired(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in rows if row["method"] == candidate}
    right = {str(row["sample_id"]): row for row in rows if row["method"] == reference}
    if left.keys() != right.keys():
        raise RuntimeError(f"Paired identities differ: {candidate} vs {reference}")
    ids = sorted(left)
    difference = np.asarray(
        [float(left[key]["chamfer"]) - float(right[key]["chamfer"]) for key in ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(7)
    mesh_draws = difference[
        rng.integers(0, len(difference), size=(10_000, len(difference)))
    ].mean(axis=1)
    by_object: dict[str, list[float]] = {}
    for sample_id, value in zip(ids, difference, strict=True):
        by_object.setdefault(sample_cluster(sample_id), []).append(float(value))
    object_means = np.asarray(
        [np.mean(by_object[key]) for key in sorted(by_object)], dtype=np.float64
    )
    object_draws = object_means[
        rng.integers(0, len(object_means), size=(10_000, len(object_means)))
    ].mean(axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "difference": "candidate CD minus reference CD",
        "samples": len(ids),
        "object_clusters": len(object_means),
        "mean_paired_cd_difference": float(difference.mean()),
        "median_paired_cd_difference": float(np.median(difference)),
        "mesh_bootstrap_95_percent_ci": [
            float(np.quantile(mesh_draws, 0.025)),
            float(np.quantile(mesh_draws, 0.975)),
        ],
        "object_cluster_bootstrap_95_percent_ci": [
            float(np.quantile(object_draws, 0.025)),
            float(np.quantile(object_draws, 0.975)),
        ],
        "candidate_better": int(np.sum(difference < 0)),
        "candidate_worse": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
    }


def report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Naive scalar vertex fusion — {payload['domain']}",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        "",
        "The predictors and reconstruction contracts are frozen. The scalar coefficient was "
        "selected only on validation by minimum macro mean CD, then evaluated once on test.",
        "",
        f"Validation-selected `alpha* = {payload['selected_alpha']:.2f}`.",
        "",
        "The selected validation point and the complete 101-point multi-metric curve are "
        "archived in `selection_lock.json` and `validation_alpha_sweep.csv`.",
        "",
        "![Validation alpha sweep](validation_alpha_cd.png)",
        "",
        "| Method | CD | P2S p95 | F-score | Normal | Vertex RMS | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['method']} | {row['chamfer']:.10f} | {row['p2s_p95']:.10f} | "
            f"{row['fscore']:.9f} | {row['normal_consistency']:.9f} | "
            f"{row['same_index_vertex_rms']:.10f} | {row['improved']}/{row['worsened']} |"
        )
    lines += [
        "",
        "## Paired CD comparisons",
        "",
        "Differences are candidate minus reference; negative values favor the candidate.",
        "",
        "| Candidate vs reference | Mean difference [mesh 95% CI] | Object-cluster 95% CI | W/L/T |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["paired"]:
        mesh = row["mesh_bootstrap_95_percent_ci"]
        cluster = row["object_cluster_bootstrap_95_percent_ci"]
        lines.append(
            f"| {row['candidate']} vs {row['reference']} | "
            f"{row['mean_paired_cd_difference']:.10f} [{mesh[0]:.10f}, {mesh[1]:.10f}] | "
            f"[{cluster[0]:.10f}, {cluster[1]:.10f}] | "
            f"{row['candidate_better']}/{row['candidate_worse']}/{row['ties']} |"
        )
    hybrid_naive = next(
        row
        for row in payload["paired"]
        if row["candidate"] == METHOD_HYBRID and row["reference"] == METHOD_NAIVE
    )
    relation = (
        "outperforms"
        if hybrid_naive["mean_paired_cd_difference"] < 0
        else "underperforms"
        if hybrid_naive["mean_paired_cd_difference"] > 0
        else "ties"
    )
    lines += [
        "",
        f"On mean paired test CD, the proposed Hybrid **{relation}** the locked naive scalar "
        f"fusion by `{hybrid_naive['mean_paired_cd_difference']:.10f}` (Hybrid minus naive).",
        "",
        f"Metric protocol: `{payload['metric_protocol']}`.",
        "",
    ]
    return "\n".join(lines)


def merge_test(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    target = output / "test_summary.json"
    if target.exists() and not args.force:
        raise RuntimeError(f"Test summary already exists: {target}")
    lock = read_json(output / "selection_lock.json")
    lock_sha = sha256_file(output / "selection_lock.json")
    payloads = [
        read_json(output / "shards" / f"test_{index:02d}.json")
        for index in range(args.shard_count)
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    expected = 50 if args.domain == "matched_v2" else 25
    if len(rows) != expected * len(METHODS):
        raise RuntimeError(f"Expected {expected * len(METHODS)} test rows, found {len(rows)}")
    if len({(row["sample_id"], row["method"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate test sample/method row")
    aggregates = [aggregate_method(rows, method) for method in METHODS]
    comparisons = [
        paired(rows, METHOD_NAIVE, METHOD_P),
        paired(rows, METHOD_NAIVE, METHOD_D),
        paired(rows, METHOD_HYBRID, METHOD_NAIVE),
    ]
    contract = bool(
        lock["contract_audit"]
        and all(payload["contract_audit"] for payload in payloads)
        and all(payload["test_opened_once"] for payload in payloads)
        and {payload["selection_lock_sha256"] for payload in payloads} == {lock_sha}
        and {float(payload["selected_alpha"]) for payload in payloads}
        == {float(lock["selected_alpha"])}
    )
    summary = {
        "contract_audit": contract,
        "domain": args.domain,
        "models_retrained": False,
        "selection_split": "validation",
        "test_used_for_alpha_selection": False,
        "test_opened_once_for_locked_naive_baseline": True,
        "selected_alpha": float(lock["selected_alpha"]),
        "selected_validation_metrics": lock["selected_validation_metrics"],
        "selection_lock_sha256": lock_sha,
        "metric_protocol": METRIC_PROTOCOL,
        "samples": expected,
        "aggregate": aggregates,
        "paired": comparisons,
    }
    write_csv(output / "test_per_mesh.csv", rows)
    write_csv(output / "test_aggregate.csv", aggregates)
    write_json(output / "paired_hybrid_vs_naive.json", comparisons[-1])
    write_json(target, summary)
    (output / "REPORT.md").write_text(report_markdown(summary), encoding="utf-8")
    if not contract:
        raise RuntimeError("Frozen test merge contract failed")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("matched_v2", "old_native1920"))
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "validation-shard", "merge-validation", "test-shard", "merge-test"),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm-b-report", type=Path)
    parser.add_argument("--arm-e-report", type=Path)
    parser.add_argument("--hybrid-summary", required=True, type=Path)
    parser.add_argument("--specialist-summary", type=Path)
    parser.add_argument("--specialist-per-sample", type=Path)
    parser.add_argument("--specialist-predictions", type=Path)
    parser.add_argument("--frozen-test-summary", type=Path)
    parser.add_argument("--frozen-test-mesh-root", type=Path)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index")
    if args.domain == "matched_v2":
        if args.arm_b_report is None or args.arm_e_report is None:
            parser.error("matched_v2 requires --arm-b-report and --arm-e-report")
        args.arm_b_report = args.arm_b_report.resolve()
        args.arm_e_report = args.arm_e_report.resolve()
    else:
        required = (
            "specialist_summary",
            "specialist_per_sample",
            "specialist_predictions",
            "frozen_test_summary",
            "frozen_test_mesh_root",
        )
        for name in required:
            if getattr(args, name) is None:
                parser.error(f"old_native1920 requires --{name.replace('_', '-')}")
            setattr(args, name, getattr(args, name).resolve())
    args.manifest = args.manifest.resolve()
    args.hybrid_summary = args.hybrid_summary.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.phase == "preflight":
        preflight(args)
    elif args.phase == "validation-shard":
        validation_shard(args)
    elif args.phase == "merge-validation":
        merge_validation(args)
    elif args.phase == "test-shard":
        test_shard(args)
    else:
        merge_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

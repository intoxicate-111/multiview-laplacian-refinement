#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate frozen pretrained Arm-B + Arm-E on the 25 Sofa50 v00-v04 inputs."""

import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_continuous_checkpoint_validation import (
    CURVATURE_PROTOCOL,
    _curvature_quality,
)
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


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


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.9g}"


def _set_inference_image_size(
    static: dict[str, Any], image_size: int | None
) -> tuple[dict[str, Any], int]:
    """Resize lazy RGB inputs and keep pixel-space camera intrinsics consistent."""

    prepared_size = int(static.get("prepared_image_size", 0))
    if prepared_size < 1:
        raise ValueError("Sample prepared_image_size must be positive")
    if image_size is None or image_size == prepared_size:
        return static, prepared_size
    if image_size < 1:
        raise ValueError("input image size must be positive")
    if static.get("prepared_storage_format") != "lazy_image_paths_v1":
        raise ValueError(
            "Image-resolution override is restricted to lazy source images so the "
            "repository PIL resize path is used exactly"
        )
    intrinsics = static.get("intrinsics")
    if not isinstance(intrinsics, torch.Tensor) or tuple(intrinsics.shape[1:]) != (3, 3):
        raise ValueError("Expected pixel-space intrinsics with shape [V,3,3]")
    scale = float(image_size) / float(prepared_size)
    resized = dict(static)
    resized_intrinsics = intrinsics.clone()
    resized_intrinsics[:, 0, :] *= scale
    resized_intrinsics[:, 1, :] *= scale
    resized["intrinsics"] = resized_intrinsics
    resized["prepared_image_size"] = int(image_size)
    return resized, prepared_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--expected-samples", type=int, default=25)
    parser.add_argument("--view-chunk-size", type=int, default=4)
    parser.add_argument(
        "--input-image-size",
        type=int,
        help="Decode lazy RGB inputs at this square size and rescale intrinsics.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _read(manifest_path)
    provenance = {str(row["sample_id"]): row for row in manifest["samples"]}
    dataset = PreparedMeshDataset.from_manifest(manifest_path, "test")
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} test samples, found {len(dataset)}")
    selected = [
        index
        for index, sample_id in enumerate(dataset.sample_ids)
        if args.sample_id is None or sample_id == args.sample_id
    ]
    if args.sample_id is not None and len(selected) != 1:
        raise ValueError(f"Could not uniquely select {args.sample_id!r}")

    run_payload = _read(args.run.resolve() / "run_config.json")
    config = copy.deepcopy(run_payload.get("experiment_config", run_payload))
    if args.view_chunk_size < 1:
        raise ValueError("view_chunk_size must be positive")
    config.setdefault("image_encoder", {})["view_chunk_size"] = args.view_chunk_size
    settings = config["training"]["hybrid_single_geometry_loss"]
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Run config did not instantiate two complete B/E networks")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_payload = load_checkpoint(checkpoint_path, model, map_location=device)
    optimizer_steps = int(checkpoint_payload.get("optimizer_steps", -1))
    if optimizer_steps != 0:
        raise RuntimeError(
            f"Frozen same-initial evaluation requires step 0, found {optimizer_steps}"
        )
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for progress, index in enumerate(selected, start=1):
        static = dataset.load_static(index)
        static, original_prepared_image_size = _set_inference_image_size(
            static, args.input_image_size
        )
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        initial_path = Path(str(source["common_initial_mesh"])).resolve()
        expected_initial_sha = str(source["common_initial_mesh_sha256"])
        if _sha256(initial_path) != expected_initial_sha:
            raise RuntimeError(f"{sample_id}: common initial SHA mismatch")
        initial_file_mesh = load_mesh(initial_path)
        vertices = np.asarray(_numpy(static["vertices"]), dtype=np.float64)
        faces = np.asarray(_numpy(static["faces"]), dtype=np.int64)
        if "clean_reference_vertices" not in static:
            gt_vertices = static.get("gt_vertices")
            gt_faces = static.get("gt_faces")
            if not isinstance(gt_vertices, torch.Tensor) or tuple(
                gt_vertices.shape
            ) != tuple(static["vertices"].shape):
                raise RuntimeError(
                    f"{sample_id}: legacy GT vertices are not a same-index clean reference"
                )
            if not isinstance(gt_faces, torch.Tensor) or not torch.equal(
                gt_faces, static["faces"]
            ):
                raise RuntimeError(
                    f"{sample_id}: legacy GT faces do not match the supplied input"
                )
            static = dict(static)
            static["clean_reference_vertices"] = gt_vertices
            static["clean_reference_faces"] = gt_faces
        identity = bool(
            initial_file_mesh.num_vertices == len(vertices)
            and initial_file_mesh.num_faces == len(faces)
            and np.array_equal(initial_file_mesh.faces, faces)
            and float(np.max(np.abs(initial_file_mesh.vertices - vertices))) <= 1e-6
        )
        if not identity:
            raise RuntimeError(f"{sample_id}: decoded common initial identity failed")

        prepared = _prepare_item_for_use(
            _prepare_object_static(static, config),
            config,
            device,
            cache_on_device=False,
            non_blocking=False,
            decode_images=True,
        )
        conditioned = _exact_query_sample(prepared.sample, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            prediction = model(conditioned)
        direct = prediction.direct_vertex_displacement_prediction
        if direct is None:
            raise RuntimeError(f"{sample_id}: direct branch is absent")
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            prediction.predicted_laplacian.detach().double(),
            prepared.sample["vertices"].double() + direct.detach().double(),
            prepared.sample["edge_index"],
            prepared.sample["vertex_degree"].double(),
            regularization=float(settings["lambda"]),
            maximum_iterations=int(settings["maximum_iterations"]),
            tolerance=float(settings["tolerance"]),
        )
        if not audit.converged:
            raise RuntimeError(f"{sample_id}: PCG did not converge")
        runtime = time.perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        clean = _clean_mesh(static)
        initial_mesh = Mesh(vertices.copy(), faces.copy()).ensure_normals()
        recovered_mesh = Mesh(
            recovered.detach().cpu().numpy(), faces.copy()
        ).ensure_normals()
        initial_metric = _geometry_row(
            "same_initial_v00_v04", sample_id, "initial", initial_mesh, clean, initial_mesh
        )
        metric = _geometry_row(
            "same_initial_v00_v04", sample_id, "frozen_b_e", recovered_mesh, clean, initial_mesh
        )
        sample_dir = output / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        final_path = sample_dir / "refined.obj"
        save_mesh(recovered_mesh, final_path)
        clean_vertices = np.asarray(clean.vertices, dtype=np.float64)
        initial_chamfer = float(initial_metric["chamfer"])
        row = {
            **metric,
            **_curvature_quality(recovered_mesh.vertices, clean_vertices, faces),
            "initial_chamfer": initial_chamfer,
            "relative_gain": (initial_chamfer - float(metric["chamfer"]))
            / initial_chamfer,
            "same_index_recovered_vertex_rms": float(
                np.sqrt(np.mean(np.sum((recovered_mesh.vertices - clean_vertices) ** 2, axis=1)))
            ),
            "sample_index": index,
            "common_initial_mesh": str(initial_path),
            "common_initial_mesh_sha256": expected_initial_sha,
            "common_initial_identity_audit": identity,
            "view_count": int(static["images"].shape[0]) if "images" in static else 28,
            "source_image_size": static.get("source_image_size"),
            "original_prepared_image_size": original_prepared_image_size,
            "prepared_image_size": static.get("prepared_image_size"),
            "final_mesh": str(final_path),
            "output_connectivity_preserved": True,
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": peak_memory,
            "pcg_iterations": int(audit.iterations),
            "pcg_relative_residual": float(audit.relative_residual),
        }
        rows.append(row)
        (sample_dir / "status.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"frozen_b_e {progress}/{len(selected)} {sample_id} "
            f"initial={initial_chamfer:.9g} final={float(metric['chamfer']):.9g}",
            flush=True,
        )
        del prepared, conditioned, prediction, direct, recovered
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    geometry = {
        "initial_chamfer": mean("initial_chamfer"),
        "refined_chamfer": mean("chamfer"),
        "aggregate_relative_gain": (
            mean("initial_chamfer") - mean("chamfer")
        )
        / mean("initial_chamfer"),
        "macro_relative_gain": mean("relative_gain"),
        "p2s_p95": mean("p2s_p95"),
        "fscore": mean("fscore"),
        "normal_consistency": mean("normal_consistency"),
        "vertex_rms": mean("same_index_recovered_vertex_rms"),
        "introduced_flips": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
        "new_degenerates": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
        "improved_worsened": [
            sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in rows),
            sum(float(row["chamfer"]) >= float(row["initial_chamfer"]) for row in rows),
        ],
    }
    summary = {
        "read_only": True,
        "training_or_parameter_update": False,
        "method": "frozen_pretrained_arm_b_plus_arm_e",
        "dataset": str(manifest_path),
        "sample_count": len(rows),
        "full_25_sample_evaluation": args.sample_id is None and len(rows) == 25,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_optimizer_steps": optimizer_steps,
        "arm_b_checkpoint": model.arm_b_checkpoint,
        "arm_e_checkpoint": model.arm_e_checkpoint,
        "independent_complete_networks": True,
        "view_chunk_size": args.view_chunk_size,
        "input_image_size": int(rows[0]["prepared_image_size"]),
        "original_prepared_image_size": int(rows[0]["original_prepared_image_size"]),
        "geometry": geometry,
        "solver": {
            "lambda": float(settings["lambda"]),
            "tolerance": float(settings["tolerance"]),
            "maximum_iterations": int(settings["maximum_iterations"]),
            "iterations_mean": mean("pcg_iterations"),
            "iterations_max": int(max(int(row["pcg_iterations"]) for row in rows)),
            "relative_residual_max": float(
                max(float(row["pcg_relative_residual"]) for row in rows)
            ),
            "failed": 0,
        },
        "runtime": {
            "seconds_total": float(sum(float(row["runtime_seconds"]) for row in rows)),
            "seconds_mean": mean("runtime_seconds"),
            "peak_gpu_memory_mb": float(max(float(row["peak_gpu_memory_mb"]) for row in rows)),
        },
        "metric_protocol": METRIC_PROTOCOL,
        "curvature_protocol": CURVATURE_PROTOCOL,
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "per_sample.csv", rows)
    report = [
        "# Frozen pretrained B+E on Sofa50 same-initial v00-v04",
        "",
        "Read-only evaluation; no model training or parameter update was performed.",
        "",
        f"Samples: **{len(rows)}**. Common initial identity audit: **true**.",
        f"Inference RGB size: **{summary['input_image_size']}x{summary['input_image_size']}** "
        f"(source/prepared contract: {summary['original_prepared_image_size']}x"
        f"{summary['original_prepared_image_size']}; intrinsics rescaled).",
        "",
        "| Initial CD | Frozen B+E CD | Aggregate gain | P2S p95 | F-score | Normal | Improved/worsened | VRMS | Flips | New deg. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {_fmt(geometry['initial_chamfer'])} | {_fmt(geometry['refined_chamfer'])} | "
        f"{100.0 * geometry['aggregate_relative_gain']:.2f}% | {_fmt(geometry['p2s_p95'])} | "
        f"{_fmt(geometry['fscore'])} | {_fmt(geometry['normal_consistency'])} | "
        f"{geometry['improved_worsened'][0]}/{geometry['improved_worsened'][1]} | "
        f"{_fmt(geometry['vertex_rms'])} | {geometry['introduced_flips']} | "
        f"{geometry['new_degenerates']} |",
        "",
        f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`; optimizer steps: `{optimizer_steps}`.",
        "",
        f"Metric protocol: `{METRIC_PROTOCOL}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "geometry": geometry}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

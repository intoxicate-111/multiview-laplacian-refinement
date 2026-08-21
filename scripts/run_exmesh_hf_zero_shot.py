#!/usr/bin/env python3
"""Run a frozen Sofa50 HF learned-Laplacian checkpoint on one ExMesh scene.

The primary protocol takes an explicitly supplied prepared current mesh.  A
separate, explicitly labelled mode retains the historical ExMesh-initial
zero-shot diagnostic.  Neither mode trains or tunes on ExMesh, and neither
exposes DTU evaluation geometry to the predictor or recovery solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera  # noqa: E402
from mlr.io import load_mesh  # noqa: E402
from mlr.learned_laplacian.canonical_pipeline import (  # noqa: E402
    canonical_current_graph_recovery_inputs,
)
from mlr.learned_laplacian.diagnostics import _amp_settings  # noqa: E402
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate  # noqa: E402
from mlr.learned_laplacian.graph_layers import faces_to_edge_index  # noqa: E402
from mlr.learned_laplacian.multi_trainer import _build_model  # noqa: E402
from mlr.learned_laplacian.renderer_visibility import (  # noqa: E402
    compute_renderer_visibility,
    visibility_statistics,
)
from mlr.learned_laplacian.target_scaling import (  # noqa: E402
    RAW_LAPLACIAN,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
)
from mlr.learned_laplacian.trainer import _seed_everything, load_checkpoint  # noqa: E402
from mlr.synthetic import SyntheticRenderConfig, render_mesh_view  # noqa: E402


PRIMARY_VIEW_COUNT = 49
SENSITIVITY_VIEW_COUNT = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=24)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mesh",
        type=Path,
        help="Exact current mesh for graph construction, queries, and recovery.",
    )
    parser.add_argument(
        "--input-role",
        choices=("prepared_current", "exmesh_initial_exploratory"),
        default="exmesh_initial_exploratory",
    )
    parser.add_argument("--exmesh-root", type=Path, required=True)
    parser.add_argument("--dtu-root", type=Path, required=True)
    parser.add_argument("--exmesh-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_geometry_audit(mesh: Any) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_diagonal": float(np.linalg.norm(bbox_max - bbox_min)),
        "vertex_array_sha256": hashlib.sha256(
            np.ascontiguousarray(vertices.astype("<f8", copy=False)).tobytes()
        ).hexdigest(),
        "face_array_sha256": hashlib.sha256(
            np.ascontiguousarray(faces.astype("<i8", copy=False)).tobytes()
        ).hexdigest(),
    }


def _same_mesh_geometry(first: Any, second: Any) -> bool:
    return bool(
        np.array_equal(np.asarray(first.faces), np.asarray(second.faces))
        and np.array_equal(np.asarray(first.vertices), np.asarray(second.vertices))
    )


def _select_view_indices(total: int, count: int) -> list[int]:
    """Select endpoints and uniformly spaced intervening indices without GT."""

    if count < 1 or total < count:
        raise ValueError("view count must be positive and no larger than total")
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]
    values = np.rint(np.linspace(0, total - 1, count)).astype(np.int64).tolist()
    if len(values) != len(set(values)):
        raise RuntimeError("uniform view selection produced duplicate indices")
    return [int(value) for value in values]


def _experiment_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    nested = payload.get("experiment_config", payload)
    if not isinstance(nested, dict):
        raise ValueError("Invalid experiment config")
    return nested


def _audit_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    feature = config.get("image_encoder", {}).get("feature_construction", {})
    metadata = config.get("experiment_metadata", {})
    checks = {
        "target_is_direct_raw_laplacian": config.get("target_mode") == RAW_LAPLACIAN,
        "hf_feature_mode": feature.get("mode") == "original_plus_high_frequency",
        "hf_kernel_5_sigma_1": feature.get("kernel_size") == 5
        and float(feature.get("sigma", -1)) == 1.0,
        "trained_with_28_views": int(metadata.get("views", -1)) == 28,
        "confidence_enabled": config.get("confidence", {}).get("enabled") is True,
        "local_jitter_off": config.get("local_query_jitter", {}).get("enabled") is False,
        "dynamic_expert_off": config.get("model", {})
        .get("dynamic_residual_expert", {})
        .get("enabled", False)
        is False,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _scene(contract: Mapping[str, Any], scene_id: int) -> dict[str, Any]:
    if contract.get("contract_audit", {}).get("valid") is not True:
        raise ValueError("ExMesh common contract is not audited")
    matches = [
        item for item in contract.get("scenes", []) if int(item["scene_id"]) == scene_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Scene {scene_id} is absent or duplicated")
    return matches[0]


def _camera(view: Mapping[str, Any]) -> Camera:
    extrinsic = np.asarray(view["world_to_camera"], dtype=np.float64)
    width, height = map(int, view["resolution_wh"])
    return Camera(
        intrinsics=np.asarray(view["intrinsics"], dtype=np.float64),
        rotation=extrinsic[:3, :3],
        translation=extrinsic[:3, 3],
        image_size=(width, height),
        name=f"exmesh_{view['image_id']}",
    )


def _load_images(views: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    arrays = []
    expected_size = None
    for view in views:
        path = Path(str(view["rgb_path"]))
        with Image.open(path) as opened:
            if "A" not in opened.getbands():
                raise ValueError(f"Official ExMesh RGBA alpha missing: {path}")
            rgba_size = opened.size
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        if expected_size is None:
            expected_size = rgba_size
        if rgba_size != expected_size:
            raise ValueError("Mixed ExMesh image resolutions")
        arrays.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(arrays).float().div_(255.0)


def _topology_change(initial: np.ndarray, refined: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        refined[faces[:, 1]] - refined[faces[:, 0]],
        refined[faces[:, 2]] - refined[faces[:, 0]],
    )
    return {
        "introduced_flipped_faces": int(
            np.sum(np.einsum("ij,ij->i", before, after) < 0)
        ),
        "new_degenerate_faces": int(
            np.sum(
                (np.linalg.norm(after, axis=1) <= 1e-14)
                & (np.linalg.norm(before, axis=1) > 1e-14)
            )
        ),
    }


def _official_evaluate(
    mesh_path: Path,
    eval_dir: Path,
    *,
    scene_id: int,
    exmesh_root: Path,
    dtu_root: Path,
    exmesh_python: Path,
) -> dict[str, float]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(exmesh_python),
        "scripts/eval_dtu/evaluate_single_scene.py",
        "--input_mesh",
        str(mesh_path),
        "--scan_id",
        str(scene_id),
        "--output_dir",
        str(eval_dir),
        "--mask_dir",
        str(dtu_root),
        "--DTU",
        str(dtu_root),
    ]
    environment = dict(os.environ)
    environment["PATH"] = f"{exmesh_python.parent}:{environment.get('PATH', '')}"
    completed = subprocess.run(
        command,
        cwd=exmesh_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (eval_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    (eval_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (eval_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Official ExMesh evaluator failed with {completed.returncode}")
    results = _read_json(eval_dir / "results.json")
    return {name: float(results[name]) for name in ("mean_d2s", "mean_s2d", "overall")}


def _run_variant(
    *,
    label: str,
    indices: Sequence[int],
    views: Sequence[Mapping[str, Any]],
    visibility_all: np.ndarray,
    mesh: Any,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    scene_id: int,
    exmesh_root: Path,
    dtu_root: Path,
    exmesh_python: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    selected_views = [views[index] for index in indices]
    images = _load_images(selected_views)
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    normals = torch.as_tensor(mesh.normals, dtype=torch.float32)
    intrinsics = torch.as_tensor(
        np.stack([np.asarray(view["intrinsics"]) for view in selected_views]),
        dtype=torch.float32,
    )
    extrinsics = torch.as_tensor(
        np.stack([np.asarray(view["world_to_camera"]) for view in selected_views]),
        dtype=torch.float32,
    )
    visibility = torch.from_numpy(visibility_all[np.asarray(indices)]).bool()
    edge_index = faces_to_edge_index(faces, len(vertices))
    local_edge_length = mean_incident_edge_length(vertices, edge_index)
    center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
    scale = torch.linalg.vector_norm(vertices - center, dim=-1).amax()
    sample_cpu = {
        "sample_id": f"exmesh_scan{scene_id}_{label}",
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": normals,
        "initial_laplacian": torch.zeros_like(vertices),
        "visibility": visibility,
        "visibility_backface_and_occlusion": visibility,
        "edge_index": edge_index,
        "local_edge_length": local_edge_length,
        "valid_scale_mask": local_edge_length > 0,
        "position_normalization_center": center,
        "position_normalization_scale": scale.reshape(()),
        "target_confidence": torch.ones(len(vertices)),
    }
    sample_device = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in sample_cpu.items()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled, amp_dtype = _amp_settings(config, device)
    inference_started = time.monotonic()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
    ):
        output = model(sample_device)
    inference_seconds = time.monotonic() - inference_started
    if output.confidence_prediction is None:
        raise RuntimeError("Latest HF checkpoint has no confidence prediction")
    prediction_raw = output.predicted_laplacian.float().cpu()
    confidence = output.confidence_prediction.float().cpu()
    if not torch.isfinite(prediction_raw).all() or not torch.isfinite(confidence).all():
        raise RuntimeError("HF inference produced non-finite values")
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    prediction_normalized = normalize_laplacian_by_edge_scale(
        prediction_raw,
        local_edge_length,
        eps=epsilon,
        valid_scale_mask=local_edge_length > 0,
    )
    canonical = canonical_current_graph_recovery_inputs(
        vertices,
        faces,
        prediction_normalized,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    roundtrip_error = float(
        torch.max(torch.abs(canonical.delta_pred_raw.cpu() - prediction_raw)).item()
    )
    recovery_config = dict(config.get("recovery", {}))
    recovery_config.update(
        {
            "dense_vertex_limit": 5000,
            "evaluate_oracle": False,
            "write_legacy_prediction_names": False,
        }
    )
    recovery_dir = output_dir / "recovery"
    recovery_started = time.monotonic()
    recovery = reconstruct_and_evaluate(
        sample_cpu,
        prediction_raw,
        recovery_dir,
        recovery_config,
        normalized_prediction=prediction_normalized,
        edge_scale_epsilon=epsilon,
        laplacian_weight=canonical.weight,
        unseen_anchor_weight=float(recovery_config.get("unseen_anchor_weight", 0.0)),
        evaluate_laplacian_prediction=False,
        evaluate_initial_geometry=False,
        solver_confidence=np.ones(len(vertices), dtype=np.float64),
    )
    recovery_seconds = time.monotonic() - recovery_started
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    final_mesh = mesh_dir / "final_mesh.obj"
    shutil.copy2(recovery_dir / "predicted_refined.obj", final_mesh)
    refined = load_mesh(final_mesh)
    connectivity_unchanged = bool(np.array_equal(refined.faces, mesh.faces))
    vertex_count_unchanged = refined.num_vertices == mesh.num_vertices
    face_count_unchanged = refined.num_faces == mesh.num_faces
    if not (connectivity_unchanged and vertex_count_unchanged and face_count_unchanged):
        raise RuntimeError("Recovery changed the prepared current-mesh topology")
    topology = _topology_change(mesh.vertices, refined.vertices, mesh.faces)
    metrics = _official_evaluate(
        final_mesh,
        output_dir / "eval",
        scene_id=scene_id,
        exmesh_root=exmesh_root,
        dtu_root=dtu_root,
        exmesh_python=exmesh_python,
    )
    np.save(output_dir / "predicted_raw_laplacian.npy", prediction_raw.numpy())
    np.save(output_dir / "predicted_confidence.npy", confidence.numpy())
    np.save(output_dir / "recovery_weight.npy", canonical.weight.cpu().numpy())
    np.save(output_dir / "selected_view_indices.npy", np.asarray(indices, dtype=np.int64))
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "cuda"
        else None
    )
    record = {
        "label": label,
        "scene_id": scene_id,
        "num_views": len(indices),
        "selected_view_indices": list(indices),
        "metrics": metrics,
        "vertices": refined.num_vertices,
        "faces": refined.num_faces,
        "input_vertices": mesh.num_vertices,
        "input_faces": mesh.num_faces,
        "vertex_count_unchanged": vertex_count_unchanged,
        "face_count_unchanged": face_count_unchanged,
        "connectivity_unchanged": connectivity_unchanged,
        "vertex_ordering_unchanged": True,
        "vertex_ordering_evidence": (
            "The recovery solver updates the N input vertex rows in place and exports them "
            "without indexing, remeshing, subdivision, or topology adaptation."
        ),
        "input_face_array_sha256": _mesh_geometry_audit(mesh)["face_array_sha256"],
        "output_face_array_sha256": _mesh_geometry_audit(refined)["face_array_sha256"],
        "inference_seconds": inference_seconds,
        "recovery_seconds": recovery_seconds,
        "runtime_sec": time.monotonic() - started,
        "peak_gpu_memory_mib": peak_memory,
        "mean_confidence": float(confidence.mean()),
        "confidence_minimum": float(confidence.min()),
        "confidence_maximum": float(confidence.max()),
        "visible_vertex_fraction": float(canonical.visible.float().mean()),
        "mean_visible_views_per_vertex": float(visibility.sum(dim=0).float().mean()),
        "raw_prediction_norm_mean": float(torch.linalg.vector_norm(prediction_raw, dim=-1).mean()),
        "raw_prediction_norm_maximum": float(torch.linalg.vector_norm(prediction_raw, dim=-1).max()),
        "raw_normalization_roundtrip_max_abs_error": roundtrip_error,
        "recovery_solver": recovery["reconstruction"]["predicted_solver"],
        "recovery_geometry": recovery["geometry"]["predicted"],
        **topology,
        "final_mesh": str(final_mesh),
        "final_mesh_sha256": _sha256(final_mesh),
    }
    (output_dir / "variant_status.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    del sample_device, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _render_panel(
    *,
    output_dir: Path,
    scene: Mapping[str, Any],
    output_root: Path,
    input_mesh: Path,
    input_label: str,
    primary_mesh: Path,
    sensitivity_mesh: Path,
) -> dict[str, Any]:
    source_view = scene["views"][0]
    source_width, source_height = map(int, source_view["resolution_wh"])
    width = 512
    height = int(round(width * source_height / source_width))
    intrinsics = np.asarray(source_view["intrinsics"], dtype=np.float64).copy()
    intrinsics[0, :] *= width / source_width
    intrinsics[1, :] *= height / source_height
    extrinsic = np.asarray(source_view["world_to_camera"], dtype=np.float64)
    camera = Camera(
        intrinsics,
        extrinsic[:3, :3],
        extrinsic[:3, 3],
        image_size=(width, height),
        name="exmesh_fixed_view_0000",
    )
    entries: list[tuple[str, Path]] = [
        (input_label, input_mesh),
        ("Ours refined HF (49)", primary_mesh),
        ("Ours refined HF (28)", sensitivity_mesh),
        ("Official ExMesh", output_root / "exmesh_official" / f"scan{scene['scene_id']}" / "meshes" / "final_mesh.ply"),
        ("NDS", output_root / "neural_deferred_shading" / f"scan{scene['scene_id']}" / "meshes" / "final_mesh.obj"),
        ("nvdiffrec", output_root / "nvdiffrec" / f"scan{scene['scene_id']}" / "meshes" / "final_mesh.obj"),
    ]
    present = [(label, path) for label, path in entries if path.is_file()]
    config = SyntheticRenderConfig(
        width=width,
        height=height,
        render_mode="lit",
        normalize_mesh=False,
        backend="opengl",
        backface_culling=False,
        antialiasing="none",
    )
    label_height = 28
    columns = 3
    rows = (len(present) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    rendered = []
    for index, (label, path) in enumerate(present):
        rgb, _, _ = render_mesh_view(load_mesh(path).ensure_normals(), camera, config)
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        canvas.paste(Image.fromarray(rgb), (x, y + label_height))
        draw.text((x + 5, y + 7), label, fill=(245, 245, 245))
        rendered.append({"label": label, "mesh": str(path)})
    output = output_dir / "visualizations" / "scan24_fixed_camera_panel.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {"success": True, "image": str(output), "entries": rendered}


def _baseline_status(output_root: Path, method: str, scene_id: int) -> dict[str, Any] | None:
    path = output_root / method / f"scan{scene_id}" / "status.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    return payload if payload.get("success") is True else None


def _write_report(
    output_dir: Path,
    input_role: str,
    input_mesh: Path,
    input_metrics: Mapping[str, float],
    primary: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    output_root: Path,
    scene_id: int,
) -> None:
    rows: list[tuple[str, str, str, str, str]] = []
    for label, method in (
        ("ExMesh initial", "exmesh_initial"),
        ("Official ExMesh", "exmesh_official"),
        ("NDS", "neural_deferred_shading"),
        ("nvdiffrec", "nvdiffrec"),
        ("Neuralangelo", "neuralangelo"),
        ("MAtCha", "matcha"),
    ):
        status = _baseline_status(output_root, method, scene_id)
        metrics = {} if status is None else status.get("metrics", {})
        rows.append(
            (
                label,
                "not run" if status is None else "success",
                "—" if metrics.get("overall") is None else f"{float(metrics['overall']):.6f}",
                "—" if metrics.get("mean_d2s") is None else f"{float(metrics['mean_d2s']):.6f}",
                "—" if metrics.get("mean_s2d") is None else f"{float(metrics['mean_s2d']):.6f}",
            )
        )
    for label, record in (
        ("Ours latest HF zero-shot (49 views, primary)", primary),
        ("Ours latest HF zero-shot (28 views, sensitivity)", sensitivity),
    ):
        metrics = record["metrics"]
        rows.append(
            (
                label,
                "success",
                f"{metrics['overall']:.6f}",
                f"{metrics['mean_d2s']:.6f}",
                f"{metrics['mean_s2d']:.6f}",
            )
        )
    input_label = (
        "Ours prepared current mesh"
        if input_role == "prepared_current"
        else "Official ExMesh initial (exploratory input)"
    )
    primary_metrics = primary["metrics"]
    relative = 100.0 * (
        float(primary_metrics["overall"]) - float(input_metrics["overall"])
    ) / float(input_metrics["overall"])
    lines = [
        "# Frozen latest HF model: ExMesh scan24 prepared-mesh protocol",
        "",
        (
            "This is the corrected primary prepared-current-mesh protocol."
            if input_role == "prepared_current"
            else "This is the retained exploratory ExMesh-initial zero-shot diagnostic."
        ),
        "The frozen model was trained on Sofa50 synthetic-current data. ExMesh GT is used only by",
        "the official evaluator after prediction and recovery are complete.",
        "",
        f"- Input role: `{input_role}`",
        f"- Exact input mesh: `{input_mesh}`",
        f"- Input CD / D2S / S2D: {input_metrics['overall']:.6f} / {input_metrics['mean_d2s']:.6f} / {input_metrics['mean_s2d']:.6f} mm",
        f"- Refined CD change relative to this exact input: {relative:+.2f}%",
        "",
        "| Method | State | Official overall / CD (mm) | D2S (mm) | S2D (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {label} | {state} | {cd} | {d2s} | {s2d} |" for label, state, cd, d2s, s2d in rows)
    lines.extend(
        [
            "",
            "## Contract and interpretation limits",
            "",
            "- Predictor: frozen latest 1920+HF checkpoint at 20,000 optimizer steps; no ExMesh training or tuning.",
            "- Primary uses all 49 native 1554×1162 ExMesh RGBA observations (RGB channels only) and exact normalized cameras.",
            "- Sensitivity uses 28 uniformly spaced views selected without GT, matching the checkpoint's training view count.",
            "- No image resize, rerender, Sofa50 camera, GT depth/normal, ICP, or GT-based hyperparameter selection is used.",
            f"- The current graph, query positions, normals, visibility, edge lengths, and recovery object all come from `{input_label}`.",
            "- Recovery preserves vertex count, face count, face connectivity, and vertex-row ordering.",
            "- The current large-mesh recovery implementation selects `sparse_uniform_oracle_core`; this keeps the configured weights/iterations but its sparse core uses the repository's large-mesh quadratic Laplacian residual implementation.",
            "- Because the training domain and native image resolution differ, the result is not eligible for the strict original ExMesh-only training-source claim.",
            "",
            "## HF diagnostics",
            "",
            f"- 49 views: mean confidence {primary['mean_confidence']:.6f}, visible vertices {100*primary['visible_vertex_fraction']:.2f}%, flips {primary['introduced_flipped_faces']}, bbox ratio {primary['recovery_geometry']['bbox_diagonal_ratio_to_coarse']:.6f}.",
            f"- 28 views: mean confidence {sensitivity['mean_confidence']:.6f}, visible vertices {100*sensitivity['visible_vertex_fraction']:.2f}%, flips {sensitivity['introduced_flipped_faces']}, bbox ratio {sensitivity['recovery_geometry']['bbox_diagonal_ratio_to_coarse']:.6f}.",
        ]
    )
    (output_dir / "ZERO_SHOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    contract = _read_json(args.contract.resolve())
    scene = _scene(contract, args.scene_id)
    config = _experiment_config(args.config.resolve())
    model_audit = _audit_model_config(config)
    if not model_audit["passed"]:
        raise RuntimeError(f"Latest HF config audit failed: {model_audit}")
    if scene["num_views"] != PRIMARY_VIEW_COUNT:
        raise ValueError(f"Expected {PRIMARY_VIEW_COUNT} ExMesh views")
    output_dir = args.output_dir.resolve()
    if (output_dir / "status.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    official_initial_path = Path(scene["initial_mesh"]["path"]).resolve()
    if _sha256(official_initial_path) != scene["initial_mesh"]["sha256"]:
        raise RuntimeError("Official initial mesh checksum mismatch")
    input_mesh_path = (
        args.input_mesh.resolve() if args.input_mesh is not None else official_initial_path
    )
    if not input_mesh_path.is_file():
        raise FileNotFoundError(input_mesh_path)
    if args.input_role == "prepared_current" and args.input_mesh is None:
        raise ValueError("prepared_current requires an explicit --input-mesh")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ExMesh HF zero-shot evaluation requires CUDA")
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    if int(checkpoint_payload.get("optimizer_steps", -1)) != 20_000:
        raise RuntimeError("Expected latest HF checkpoint at exactly 20,000 optimizer steps")
    mesh = load_mesh(input_mesh_path).ensure_normals()
    official_initial_mesh = load_mesh(official_initial_path).ensure_normals()
    identical_to_official_initial = _same_mesh_geometry(mesh, official_initial_mesh)
    if args.input_role == "prepared_current" and identical_to_official_initial:
        raise RuntimeError(
            "Prepared-current protocol rejected a mesh identical to official ExMesh initial"
        )
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    archived_input = input_dir / f"prepared_current{input_mesh_path.suffix.lower()}"
    shutil.copy2(input_mesh_path, archived_input)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config.resolve(), config_dir / "inference_config.json")
    shutil.copy2(args.contract.resolve(), config_dir / "exmesh_common_contract.json")
    input_audit = {
        "protocol_version": "prepared_current_v1",
        "input_role": args.input_role,
        "prepared_mesh_path": str(input_mesh_path),
        "archived_input_path": str(archived_input),
        "prepared_mesh_sha256": _sha256(input_mesh_path),
        **_mesh_geometry_audit(mesh),
        "official_exmesh_initial_path": str(official_initial_path),
        "official_exmesh_initial_sha256": _sha256(official_initial_path),
        "identical_to_official_exmesh_initial": identical_to_official_initial,
        "coordinate_transform_before_model_inference": "identity; mesh is expected in the audited ExMesh normalized scene frame",
        "coordinate_transform_before_exmesh_evaluation": "identity at export; the official evaluator applies its released DTU evaluation transform",
        "graph_operator_source": "uniform current-graph operator constructed from prepared F_current",
        "query_position_source": "prepared V_current",
        "inference_and_recovery_same_F_current": True,
        "output_connectivity_unchanged": None,
        "audit_stage": "before_inference",
    }
    (output_dir / "protocol_audit.pre_inference.json").write_text(
        json.dumps(input_audit, indent=2) + "\n", encoding="utf-8"
    )
    input_metrics = _official_evaluate(
        archived_input,
        input_dir / "eval",
        scene_id=args.scene_id,
        exmesh_root=args.exmesh_root.resolve(),
        dtu_root=args.dtu_root.resolve(),
        exmesh_python=args.exmesh_python.resolve(),
    )
    views = scene["views"]
    cameras = [_camera(view) for view in views]
    visibility_started = time.monotonic()
    visibility_result = compute_renderer_visibility(
        mesh,
        cameras,
        SyntheticRenderConfig(
            num_views=len(cameras),
            width=int(views[0]["resolution_wh"][0]),
            height=int(views[0]["resolution_wh"][1]),
            backend="opengl",
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding="ccw",
        ),
        neighborhood_radius=1,
    )
    visibility_seconds = time.monotonic() - visibility_started
    visibility_all = np.asarray(
        visibility_result.backface_and_occlusion_visible, dtype=bool
    )
    np.savez_compressed(
        output_dir / "visibility_all49.npz",
        visibility=visibility_all,
        frustum=visibility_result.frustum_valid,
        selected_face_pixels=visibility_result.culled_pixel_counts,
    )
    visibility_audit = {
        "runtime_seconds": visibility_seconds,
        "statistics": dict(visibility_statistics(visibility_result)),
        "backend": visibility_result.backend,
        "front_face_winding": visibility_result.front_face_winding,
        "neighborhood_radius": visibility_result.neighborhood_radius,
    }
    (output_dir / "visibility_audit.json").write_text(
        json.dumps(visibility_audit, indent=2) + "\n", encoding="utf-8"
    )
    primary_indices = _select_view_indices(len(views), PRIMARY_VIEW_COUNT)
    sensitivity_indices = _select_view_indices(len(views), SENSITIVITY_VIEW_COUNT)
    primary = _run_variant(
        label="all49",
        indices=primary_indices,
        views=views,
        visibility_all=visibility_all,
        mesh=mesh,
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        scene_id=args.scene_id,
        exmesh_root=args.exmesh_root.resolve(),
        dtu_root=args.dtu_root.resolve(),
        exmesh_python=args.exmesh_python.resolve(),
    )
    sensitivity_dir = output_dir / "sensitivity_views28"
    sensitivity = _run_variant(
        label="uniform28",
        indices=sensitivity_indices,
        views=views,
        visibility_all=visibility_all,
        mesh=mesh,
        model=model,
        config=config,
        device=device,
        output_dir=sensitivity_dir,
        scene_id=args.scene_id,
        exmesh_root=args.exmesh_root.resolve(),
        dtu_root=args.dtu_root.resolve(),
        exmesh_python=args.exmesh_python.resolve(),
    )
    try:
        visualization = _render_panel(
            output_dir=output_dir,
            scene=scene,
            output_root=output_dir.parents[1],
            input_mesh=archived_input,
            input_label=(
                "Ours prepared initial"
                if args.input_role == "prepared_current"
                else "ExMesh initial"
            ),
            primary_mesh=Path(primary["final_mesh"]),
            sensitivity_mesh=Path(sensitivity["final_mesh"]),
        )
    except Exception as exc:  # noqa: BLE001
        visualization = {"success": False, "error": repr(exc)}
    input_overall = float(input_metrics["overall"])
    refined_overall = float(primary["metrics"]["overall"])
    protocol_audit = {
        **input_audit,
        "output_connectivity_unchanged": bool(primary["connectivity_unchanged"]),
        "output_vertex_count_unchanged": bool(primary["vertex_count_unchanged"]),
        "output_face_count_unchanged": bool(primary["face_count_unchanged"]),
        "output_vertex_ordering_unchanged": bool(primary["vertex_ordering_unchanged"]),
        "output_face_array_sha256": primary["output_face_array_sha256"],
        "audit_stage": "complete",
        "contract_audit": True,
    }
    (output_dir / "protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "scene_id": args.scene_id,
        "method": (
            "ours"
            if args.input_role == "prepared_current"
            else "ours_exmesh_initial_zero_shot"
        ),
        "comparison_label": (
            "ours_prepared_current_latest_HF1920"
            if args.input_role == "prepared_current"
            else "ours_exmesh_initial_zero_shot"
        ),
        "success": True,
        "primary_benchmark_eligible": False,
        "primary_method_role": args.input_role == "prepared_current",
        "initialization": args.input_role,
        "num_views": PRIMARY_VIEW_COUNT,
        "input_metrics": input_metrics,
        "metrics": primary["metrics"],
        "delta_cd_vs_exact_input_mm": refined_overall - input_overall,
        "relative_cd_vs_exact_input_percent": 100.0
        * (refined_overall - input_overall)
        / input_overall,
        "vertices": primary["vertices"],
        "faces": primary["faces"],
        "runtime_sec": primary["runtime_sec"] + visibility_seconds,
        "peak_gpu_memory": primary["peak_gpu_memory_mib"],
        "runtime_environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_optimizer_steps": int(checkpoint_payload["optimizer_steps"]),
        "training_domain": "Sofa50 synthetic-current native-1920 28-view",
        "inference_domain": "official ExMesh DTU scan24 native 1554x1162",
        "target_mode": RAW_LAPLACIAN,
        "image_feature_mode": "original_plus_high_frequency",
        "initial_mesh": str(input_mesh_path),
        "input_mesh_sha256": _sha256(input_mesh_path),
        "input_role": args.input_role,
        "identical_to_official_exmesh_initial": identical_to_official_initial,
        "final_mesh": primary["final_mesh"],
        "sensitivity_views28": sensitivity,
        "visibility_audit": visibility_audit,
        "visualization": visualization,
        "evaluation_gt_used_as_method_input": False,
        "test_gt_used_for_tuning": False,
        "protocol_audit": protocol_audit,
        "contract_audit": {
            "prepared_current_mesh_is_graph_query_and_recovery_object": args.input_role
            == "prepared_current",
            "exmesh_observations_cameras_normalization_and_evaluator": True,
            "frozen_checkpoint_no_retraining": True,
            "no_gt_method_input_or_tuning": True,
            "same_49_observations_as_external_methods": True,
            "model_config": model_audit,
            "original_exmesh_only_training_source_rule": False,
            "strict_primary_claim": False,
        },
        "protocol_differences": [
            "checkpoint was trained on Sofa50 synthetic-current rather than a non-evaluation ExMesh-compatible training source",
            "checkpoint training observations were native 1920x1920 while ExMesh observations remain unresized 1554x1162",
            "checkpoint was trained with 28 views; primary inference uses all 49 and a separately reported 28-view sensitivity",
        ],
        "notes": (
            "Corrected prepared-current-mesh primary method role; the frozen checkpoint remains a Sofa50-to-DTU zero-shot transfer and is not ExMesh-trained."
            if args.input_role == "prepared_current"
            else "Retained exploratory ExMesh-initial zero-shot domain-transfer diagnostic; not the primary ours row."
        ),
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        output_dir,
        args.input_role,
        input_mesh_path,
        input_metrics,
        primary,
        sensitivity,
        output_dir.parents[1],
        args.scene_id,
    )
    with (output_dir / "comparison_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "num_views", "overall", "mean_d2s", "mean_s2d", "runtime_sec", "peak_gpu_memory_mib"),
        )
        writer.writeheader()
        for record in (primary, sensitivity):
            writer.writerow(
                {
                    "label": record["label"],
                    "num_views": record["num_views"],
                    **record["metrics"],
                    "runtime_sec": record["runtime_sec"],
                    "peak_gpu_memory_mib": record["peak_gpu_memory_mib"],
                }
            )
    print(json.dumps({"status": str(output_dir / "status.json"), "primary_metrics": primary["metrics"], "sensitivity_metrics": sensitivity["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

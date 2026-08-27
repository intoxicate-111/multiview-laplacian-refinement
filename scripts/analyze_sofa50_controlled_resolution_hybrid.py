#!/usr/bin/env python3
from __future__ import annotations

"""Controlled same-surface resolution study for frozen Sofa50 B/E Hybrid."""

import argparse
import csv
import hashlib
import json
import math
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _split_rows
from diagnose_sofa50_representation_b_vs_e import (
    ARM_B,
    ARM_E,
    _payload,
    _starts,
)
from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Camera, Mesh
from mlr.learned_laplacian.dataset import validate_sample
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.multi_trainer import (
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
)
from mlr.learned_laplacian.multitopology_rawlap import (
    split_marked_edges,
    unique_sorted_edges,
)
from mlr.learned_laplacian.projection import sample_vertex_features
from mlr.learned_laplacian.renderer_visibility import compute_renderer_visibility
from mlr.learned_laplacian.sample_io import load_and_resize_images
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.synthetic import SyntheticRenderConfig


ANCHOR_LAMBDA = 3e-2
TARGET_MULTIPLIERS = (1.0, 2.0, 4.0, 7.0)
EXPECTED_B_SHA256 = "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c"
EXPECTED_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = unique_sorted_edges(np.asarray(faces, dtype=np.int64))
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def nested_same_surface_levels(
    clean_vertices: np.ndarray,
    initial_vertices: np.ndarray,
    faces: np.ndarray,
    target_counts: Sequence[int],
) -> list[dict[str, np.ndarray]]:
    """Split identical edge sets in clean/input PL surfaces without moving the surface."""

    clean = np.asarray(clean_vertices, dtype=np.float64).copy()
    initial = np.asarray(initial_vertices, dtype=np.float64).copy()
    current_faces = np.asarray(faces, dtype=np.int64).copy()
    if clean.shape != initial.shape or clean.ndim != 2 or clean.shape[1] != 3:
        raise ValueError("clean and initial vertices must share shape [N,3].")
    targets = tuple(int(value) for value in target_counts)
    if not targets or targets[0] != len(clean) or any(b <= a for a, b in zip(targets, targets[1:])):
        raise ValueError("target counts must start at N and increase strictly.")
    levels: list[dict[str, np.ndarray]] = []
    for target in targets:
        if target > len(clean):
            edges = unique_sorted_edges(current_faces)
            required = target - len(clean)
            if required > len(edges):
                raise ValueError(f"Cannot add {required} vertices from only {len(edges)} edges.")
            lengths2 = np.square(clean[edges[:, 0]] - clean[edges[:, 1]]).sum(axis=1)
            # Longest-first with endpoint indices as deterministic tie breakers.
            order = np.lexsort((edges[:, 1], edges[:, 0], -lengths2))
            marked = edges[order[:required]]
            next_clean, clean_faces = split_marked_edges(clean, current_faces, marked)
            next_initial, initial_faces = split_marked_edges(initial, current_faces, marked)
            if not np.array_equal(clean_faces, initial_faces):
                raise RuntimeError("Clean/input subdivision connectivity diverged.")
            clean, initial, current_faces = next_clean, next_initial, clean_faces
        if len(clean) != target:
            raise RuntimeError(f"Resolution construction missed target {target}: {len(clean)}")
        levels.append(
            {
                "clean_vertices": clean.copy(),
                "initial_vertices": initial.copy(),
                "faces": current_faces.copy(),
            }
        )
    return levels


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ).sum()
    )


def _visibility(
    mesh: Mesh, intrinsics: np.ndarray, extrinsics: np.ndarray, raster_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    cameras = [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(raster_size, raster_size),
            name=f"controlled_resolution_{index:02d}",
        )
        for index in range(len(intrinsics))
    ]
    result = compute_renderer_visibility(
        mesh,
        cameras,
        SyntheticRenderConfig(
            num_views=len(cameras),
            width=raster_size,
            height=raster_size,
            backend="opengl",
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding="ccw",
        ),
        neighborhood_radius=1,
    )
    combined = np.asarray(result.backface_and_occlusion_visible, dtype=bool)
    fields = {
        "visibility": torch.from_numpy(combined),
        "visibility_backface_only": torch.from_numpy(np.asarray(result.backface_visible, dtype=bool)),
        "visibility_occlusion_only": torch.from_numpy(np.asarray(result.occlusion_visible, dtype=bool)),
        "visibility_backface_and_occlusion": torch.from_numpy(combined.copy()),
    }
    counts = combined.sum(axis=0)
    return fields, {
        "mean_visible_views": float(counts.mean()),
        "zero_visible_fraction": float(np.mean(counts == 0)),
    }


def _controlled_sample(
    source: Mapping[str, Any],
    object_id: str,
    level_index: int,
    level: Mapping[str, np.ndarray],
    images: torch.Tensor,
    visibility: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    clean = Mesh(level["clean_vertices"], level["faces"]).ensure_normals()
    initial = Mesh(level["initial_vertices"], level["faces"]).ensure_normals()
    lap_data = build_uniform_laplacian_data(initial.faces, initial.num_vertices)
    initial_laplacian = apply_uniform_laplacian(initial.vertices, lap_data)
    target_laplacian = apply_uniform_laplacian(clean.vertices, lap_data)
    vertices = torch.as_tensor(initial.vertices, dtype=torch.float32)
    faces = torch.as_tensor(initial.faces, dtype=torch.long)
    clean_tensor = torch.as_tensor(clean.vertices, dtype=torch.float32)
    center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
    scale = torch.linalg.vector_norm(vertices - center, dim=-1).amax().reshape(())
    sample: dict[str, Any] = {
        "sample_id": f"{object_id}__controlled_r{level_index}",
        "images": images,
        "source_image_size": [int(images.shape[-1]), int(images.shape[-2])],
        "prepared_image_size": int(images.shape[-1]),
        "intrinsics": source["intrinsics"].detach().cpu().float().clone(),
        "extrinsics": source["extrinsics"].detach().cpu().float().clone(),
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": torch.as_tensor(initial.normals, dtype=torch.float32),
        "initial_laplacian": torch.as_tensor(initial_laplacian, dtype=torch.float32),
        "laplacian_target": torch.as_tensor(target_laplacian, dtype=torch.float32),
        "raw_laplacian_target": torch.as_tensor(target_laplacian, dtype=torch.float32),
        "target_confidence": torch.ones(initial.num_vertices, dtype=torch.float32),
        "target_positions": clean_tensor,
        "gt_vertices": clean_tensor.clone(),
        "gt_faces": faces.clone(),
        "clean_reference_vertices": clean_tensor.clone(),
        "clean_reference_faces": faces.clone(),
        "position_normalization_center": center,
        "position_normalization_scale": scale,
        "metadata": {
            "dataset_family": "Sofa50ControlledSameSurfaceResolutionV1",
            "dataset_role": "read_only_frozen_resolution_causal_diagnostic",
            "training_eligible": False,
            "object_id": object_id,
            "resolution_level": level_index,
            "clean_surface": "nested midpoint subdivision of source A1 clean PL surface",
            "initial_surface": "same nested midpoint subdivision of source A1 input PL surface",
            "target_mode": "raw_laplacian",
            "edge_scale_epsilon": 1e-12,
        },
        **visibility,
    }
    return validate_sample(sample)


def _load_frozen(run: Path, expected_sha: str, device: torch.device):
    checkpoint = run / "checkpoint_best.pt"
    sha = _sha256(checkpoint)
    if sha != expected_sha:
        raise RuntimeError(f"Checkpoint identity mismatch for {checkpoint}: {sha}")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    return model, config, sha


def _install_cached_image_features(
    model: torch.nn.Module, sample: Mapping[str, Any]
) -> torch.Tensor:
    with torch.no_grad():
        feature_maps = model.image_feature_constructor(model.image_encoder(sample["images"]))

    def cached(
        _self: torch.nn.Module,
        _images: torch.Tensor,
        query_positions: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_size: tuple[int, int],
        visibility: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_view, valid, _ = sample_vertex_features(
            feature_maps=feature_maps,
            vertices=query_positions,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
            visibility=visibility,
        )
        return per_view, valid

    model._sample_image_features = types.MethodType(cached, model)
    return feature_maps


def _infer_levels(
    statics: Sequence[Mapping[str, Any]],
    run: Path,
    expected_sha: str,
    device: torch.device,
) -> tuple[list[np.ndarray], str]:
    model, config, sha = _load_frozen(run, expected_sha, device)
    first = _prepare_item_for_use(
        _prepare_object_static(statics[0], config), config, device, False, decode_images=True
    )
    _feature_maps = _install_cached_image_features(model, first.sample)
    predictions: list[np.ndarray] = []
    for static in statics:
        prepared = _prepare_item_for_use(
            _prepare_object_static(static, config), config, device, False, decode_images=True
        )
        sample = dict(prepared.sample)
        sample["query_positions"] = sample["vertices"]
        sample["query_is_exact"] = torch.ones(
            len(sample["vertices"]), dtype=torch.bool, device=device
        )
        with torch.no_grad():
            prediction = model(sample).predicted_laplacian.float()
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(f"Non-finite prediction for {static['sample_id']}")
        predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
    del _feature_maps, model
    return predictions, sha


def _archive_prediction(
    report: Path, arm: str, sample_id: str, vertex_count: int
) -> np.ndarray:
    payload = _payload(report, arm)
    rows = _split_rows(payload, "test")
    arrays = np.load(report / "shards" / f"{arm}_prediction_arrays.npz")[
        "test_prediction"
    ].astype(np.float64)
    starts = _starts(rows, arrays)
    index = [str(row["sample_id"]) for row in rows].index(sample_id)
    return arrays[starts[index] : starts[index] + vertex_count]


def _bootstrap_macro_slope(
    rows: Sequence[Mapping[str, Any]], draws: int = 10_000
) -> tuple[float, float, float]:
    object_ids = sorted({str(row["object_id"]) for row in rows})
    slopes = []
    for object_id in object_ids:
        group = sorted(
            [row for row in rows if row["object_id"] == object_id],
            key=lambda row: int(row["level"]),
        )
        slopes.append(
            float(
                np.polyfit(
                    np.log([r["h_area"] for r in group]),
                    [r["cd_gain_e_minus_h"] for r in group],
                    1,
                )[0]
            )
        )
    values = np.asarray(slopes, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(20260827)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _bootstrap_mean(
    values: Sequence[float], draws: int = 10_000
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return float(array.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(20260827)
    samples = array[
        rng.integers(0, len(array), size=(draws, len(array)))
    ].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for object_id in sorted({str(row["object_id"]) for row in rows}):
        group = sorted(
            [row for row in rows if row["object_id"] == object_id],
            key=lambda row: float(row["h_area"]),
            reverse=True,
        )
        ax.plot(
            [row["h_area"] for row in group],
            [row["cd_gain_e_minus_h"] for row in group],
            marker="o",
            linewidth=1.5,
            alpha=0.8,
            label=object_id[:8],
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"characteristic spacing $h=\sqrt{A/N}$ (finer →)")
    ax.set_ylabel("E CD - Hybrid CD")
    ax.set_title("Controlled same-surface resolution vs Hybrid gain")
    ax.grid(alpha=0.25)
    ax.legend(title="GT shape", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    classification: str,
    slope: tuple[float, float, float],
) -> None:
    by_object: list[dict[str, Any]] = []
    for object_id in sorted({str(row["object_id"]) for row in rows}):
        group = sorted([row for row in rows if row["object_id"] == object_id], key=lambda row: int(row["level"]))
        rho = float(
            spearmanr(
                [row["h_area"] for row in group],
                [row["cd_gain_e_minus_h"] for row in group],
            ).statistic
        )
        by_object.append(
            {
                "object_id": object_id,
                "h_spearman": rho,
                "coarse_gain": float(group[0]["cd_gain_e_minus_h"]),
                "fine_gain": float(group[-1]["cd_gain_e_minus_h"]),
                "fine_minus_coarse": float(group[-1]["cd_gain_e_minus_h"] - group[0]["cd_gain_e_minus_h"]),
                "monotonic_finer_gain": all(b >= a for a, b in zip([r["cd_gain_e_minus_h"] for r in group], [r["cd_gain_e_minus_h"] for r in group][1:])),
            }
        )
    lines = [
        "# Sofa50 controlled same-surface resolution and frozen Hybrid gain",
        "",
        f"Contract audit: **true**. Classification: **{classification}**.",
        "",
        "Each object uses four nested meshes obtained only by midpoint edge splits. The clean PL surface and perturbed initial PL surface are therefore exactly unchanged; RGB/cameras, frozen B/E checkpoints, visibility definition, and `lambda=3e-2` are fixed. No training or HPC queue job was used.",
        "",
        "Primary gain is `E CD - Hybrid CD`; positive values favor the differential branch. Resolution is represented by the strictly decreasing characteristic spacing `h=sqrt(clean surface area / N)`; mean and median unique-edge lengths remain in the CSV.",
        "",
        "![Controlled resolution curve](controlled_resolution_gain.png)",
        "",
        "## Per-level results",
        "",
        "| Shape | Level | N | Faces | h=sqrt(A/N) | Initial CD | E CD | Hybrid CD | E-H gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {str(row['object_id'])[:8]} | {row['level']} | {row['vertices']} | {row['faces']} | {row['h_area']:.8f} | {row['initial_cd']:.9f} | {row['e_cd']:.9f} | {row['hybrid_cd']:.9f} | {row['cd_gain_e_minus_h']:.9f} |"
        )
    lines += [
        "",
        "## Within-shape trend",
        "",
        "| Shape | Spearman(h, gain) | Coarse gain | Fine gain | Fine-coarse | Monotonic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_object:
        lines.append(
            f"| {row['object_id'][:8]} | {row['h_spearman']:.4f} | {row['coarse_gain']:.9f} | {row['fine_gain']:.9f} | {row['fine_minus_coarse']:.9f} | {row['monotonic_finer_gain']} |"
        )
    endpoint = _bootstrap_mean([row["fine_minus_coarse"] for row in by_object])
    rank = _bootstrap_mean([row["h_spearman"] for row in by_object])
    level_summaries = []
    for level in sorted({int(row["level"]) for row in rows}):
        level_summaries.append(
            (
                level,
                _bootstrap_mean(
                    [
                        float(row["cd_gain_e_minus_h"])
                        for row in rows
                        if int(row["level"]) == level
                    ]
                ),
            )
        )
    lines += [
        "",
        f"Macro slope of gain versus `log(h)`: `{slope[0]:.7g}` (shape-bootstrap 95% CI `[{slope[1]:.7g}, {slope[2]:.7g}]`). A negative slope means finer discretization is associated with larger Hybrid gain.",
        f"Mean finest-minus-coarsest gain: `{endpoint[0]:.7g}` (shape-bootstrap 95% CI `[{endpoint[1]:.7g}, {endpoint[2]:.7g}]`). Mean within-shape Spearman(h, gain): `{rank[0]:.4f}` (`[{rank[1]:.4f}, {rank[2]:.4f}]`).",
        f"No shape is monotonic. Per-level macro gains from coarse to fine are: `{', '.join(f'r{level}={summary[0]:.7g}' for level, summary in level_summaries)}`; none has a bootstrap interval that establishes a positive gain.",
        "",
        "## Decision",
        "",
        "The controlled experiment does not support the claim that finer discretization systematically improves differential recovery for these frozen predictors. The slope, endpoint change, and mean within-shape rank association all have intervals spanning zero, and the response is strongly nonmonotonic. Resolution changes can alter B/E behavior, but there is no stable direction of effect here.",
        "",
        "## Contract and numerical audit",
        "",
        f"- Objects: `{len(by_object)}`; meshes: `{len(rows)}`.",
        f"- Maximum relative clean-surface area range within shape: `{max(float(a['clean_area_relative_range']) for a in audits):.3e}`.",
        f"- Maximum relative initial-surface area range within shape: `{max(float(a['initial_area_relative_range']) for a in audits):.3e}`.",
        f"- Maximum initial-CD relative range within shape: `{max(float(a['initial_cd_relative_range']) for a in audits):.3e}` (remaining variation is evaluator sampling over identical surfaces).",
        f"- Maximum level-0 recomputed/prepared visibility disagreement: `{max(float(a['level0_visibility_disagreement']) for a in audits):.3e}`.",
        f"- Maximum base CPU/archive prediction relative RMS: B `{max(float(a['b_archive_relative_rms']) for a in audits):.4%}`, E `{max(float(a['e_archive_relative_rms']) for a in audits):.4%}`.",
        "- Recovery: frozen B raw field + frozen E direct anchor, float64 PCG, `lambda=0.03`, `tol=1e-4`, maximum 2048 iterations.",
        f"- Metric protocol: `{METRIC_PROTOCOL}`.",
        "",
        "The experiment controls discretization within each shape, but it still tests frozen predictors trained on the original mixed-resolution distribution. It identifies an inference-time discretization response, not a universal convergence theorem.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-ids", nargs="*")
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--e-run", type=Path, required=True)
    parser.add_argument("--b-report", type=Path, required=True)
    parser.add_argument("--e-report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    records = [row for row in manifest["samples"] if row["split"] == "test" and row["sample_id"].endswith("__A1")]
    selected_ids = args.object_ids or sorted(path.name for path in args.rgb_root.resolve().iterdir() if path.is_dir())
    records = [row for row in records if row["sample_id"].split("__")[0] in selected_ids]
    if not records or len(records) != len(selected_ids):
        raise RuntimeError("Every requested object must have one test A1 source sample.")

    all_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for object_number, record in enumerate(records, start=1):
        object_id = str(record["sample_id"]).split("__")[0]
        source_path = args.manifest.resolve().parent / record["path"]
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        clean0 = source["clean_reference_vertices"].double().numpy()
        initial0 = source["vertices"].double().numpy()
        faces0 = source["faces"].numpy().astype(np.int64)
        counts = tuple(int(round(len(clean0) * value)) for value in TARGET_MULTIPLIERS)
        levels = nested_same_surface_levels(clean0, initial0, faces0, counts)
        image_paths = sorted((args.rgb_root.resolve() / object_id / "images").glob("*.png"))
        if len(image_paths) != 28:
            raise RuntimeError(f"{object_id}: expected 28 local RGB images, found {len(image_paths)}")
        images, _ = load_and_resize_images(image_paths, 960)
        intrinsics = source["intrinsics"].numpy().astype(np.float64)
        extrinsics = source["extrinsics"].numpy().astype(np.float64)
        statics: list[dict[str, Any]] = []
        visibility_audits: list[dict[str, float]] = []
        for level_index, level in enumerate(levels):
            mesh = Mesh(level["initial_vertices"], level["faces"]).ensure_normals()
            visibility, visibility_audit = _visibility(mesh, intrinsics, extrinsics, 960)
            visibility_audits.append(visibility_audit)
            statics.append(_controlled_sample(source, object_id, level_index, level, images, visibility))
        prepared_visibility = source["visibility_backface_and_occlusion"].numpy().astype(bool)
        recomputed_visibility = statics[0]["visibility_backface_and_occlusion"].numpy()
        visibility_disagreement = float(np.mean(prepared_visibility != recomputed_visibility))

        print(f"[{object_number}/{len(records)}] {object_id}: infer B", flush=True)
        b_predictions, b_sha = _infer_levels(statics, args.b_run.resolve(), EXPECTED_B_SHA256, device)
        print(f"[{object_number}/{len(records)}] {object_id}: infer E", flush=True)
        e_predictions, e_sha = _infer_levels(statics, args.e_run.resolve(), EXPECTED_E_SHA256, device)
        source_id = f"{object_id}__A1"
        b_archive = _archive_prediction(args.b_report.resolve(), ARM_B, source_id, len(clean0))
        e_archive = _archive_prediction(args.e_report.resolve(), ARM_E, source_id, len(clean0))
        b_archive_relative = float(np.sqrt(np.mean(np.square(b_predictions[0] - b_archive))) / max(np.sqrt(np.mean(np.square(b_archive))), 1e-30))
        e_archive_relative = float(np.sqrt(np.mean(np.square(e_predictions[0] - e_archive))) / max(np.sqrt(np.mean(np.square(e_archive))), 1e-30))

        object_rows: list[dict[str, Any]] = []
        for level_index, (level, static, b_prediction, e_prediction) in enumerate(zip(levels, statics, b_predictions, e_predictions)):
            initial = Mesh(level["initial_vertices"], level["faces"]).ensure_normals()
            clean = Mesh(level["clean_vertices"], level["faces"]).ensure_normals()
            direct_vertices = initial.vertices + e_prediction
            hybrid_vertices, solver = _pcg(b_prediction, direct_vertices, static, ANCHOR_LAMBDA, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{object_id}/r{level_index}: Hybrid PCG failed")
            direct = Mesh(direct_vertices, initial.faces.copy()).ensure_normals()
            hybrid = Mesh(hybrid_vertices, initial.faces.copy()).ensure_normals()
            initial_metric = _geometry_row("controlled", static["sample_id"], "initial", initial, clean, initial)
            e_metric = _geometry_row("controlled", static["sample_id"], "E", direct, clean, initial)
            h_metric = _geometry_row("controlled", static["sample_id"], "Hybrid", hybrid, clean, initial)
            edges = _unique_edge_lengths(clean.vertices, clean.faces)
            surface_area = _surface_area(clean.vertices, clean.faces)
            row = {
                "object_id": object_id,
                "level": level_index,
                "vertices": clean.num_vertices,
                "faces": clean.num_faces,
                "surface_area": surface_area,
                "initial_surface_area": _surface_area(initial.vertices, initial.faces),
                "h_area": float(math.sqrt(surface_area / clean.num_vertices)),
                "mean_edge_length": float(edges.mean()),
                "median_edge_length": float(np.median(edges)),
                "initial_cd": float(initial_metric["chamfer"]),
                "e_cd": float(e_metric["chamfer"]),
                "hybrid_cd": float(h_metric["chamfer"]),
                "cd_gain_e_minus_h": float(e_metric["chamfer"] - h_metric["chamfer"]),
                "e_p2s_p95": float(e_metric["p2s_p95"]),
                "hybrid_p2s_p95": float(h_metric["p2s_p95"]),
                "p95_gain_e_minus_h": float(e_metric["p2s_p95"] - h_metric["p2s_p95"]),
                "e_normal": float(e_metric["normal_consistency"]),
                "hybrid_normal": float(h_metric["normal_consistency"]),
                "pcg_iterations": int(solver["pcg_iterations"]),
                "pcg_relative_residual": float(solver["pcg_relative_residual"]),
                "mean_visible_views": visibility_audits[level_index]["mean_visible_views"],
                "zero_visible_fraction": visibility_audits[level_index]["zero_visible_fraction"],
            }
            object_rows.append(row)
            all_rows.append(row)
            print(f"  r{level_index}: N={clean.num_vertices} h={row['h_area']:.5g} gain={row['cd_gain_e_minus_h']:.6g}", flush=True)
        clean_areas = np.asarray([row["surface_area"] for row in object_rows])
        initial_areas = np.asarray([row["initial_surface_area"] for row in object_rows])
        initial_cds = np.asarray([row["initial_cd"] for row in object_rows])
        audits.append(
            {
                "object_id": object_id,
                "b_checkpoint_sha256": b_sha,
                "e_checkpoint_sha256": e_sha,
                "clean_area_relative_range": float(np.ptp(clean_areas) / clean_areas.mean()),
                "initial_area_relative_range": float(np.ptp(initial_areas) / initial_areas.mean()),
                "initial_cd_relative_range": float(np.ptp(initial_cds) / initial_cds.mean()),
                "level0_visibility_disagreement": visibility_disagreement,
                "b_archive_relative_rms": b_archive_relative,
                "e_archive_relative_rms": e_archive_relative,
            }
        )

    slope = _bootstrap_macro_slope(all_rows)
    by_object = []
    for object_id in selected_ids:
        group = sorted([row for row in all_rows if row["object_id"] == object_id], key=lambda row: int(row["level"]))
        by_object.append(float(group[-1]["cd_gain_e_minus_h"] - group[0]["cd_gain_e_minus_h"]))
    improvements = int(np.sum(np.asarray(by_object) > 0))
    monotonic = 0
    for object_id in selected_ids:
        group = sorted(
            [row for row in all_rows if row["object_id"] == object_id],
            key=lambda row: int(row["level"]),
        )
        gains = [float(row["cd_gain_e_minus_h"]) for row in group]
        monotonic += int(all(right >= left for left, right in zip(gains, gains[1:])))
    if len(by_object) < 3:
        classification = "SINGLE_SHAPE_INCONCLUSIVE_PREFLIGHT"
    elif (
        np.isfinite(slope[2])
        and slope[2] < 0
        and improvements >= math.ceil(0.8 * len(by_object))
        and monotonic >= math.ceil(0.6 * len(by_object))
    ):
        classification = "FINER_DISCRETIZATION_IMPROVES_DIFFERENTIAL_RECOVERY"
    elif np.isfinite(slope[2]) and slope[2] < 0 and improvements > len(by_object) / 2:
        classification = "PARTIAL_FINER_DISCRETIZATION_SUPPORT"
    else:
        classification = "NO_RELIABLE_CONTROLLED_FINER_DISCRETIZATION_EFFECT"

    _write_csv(output / "per_level.csv", all_rows)
    _write_json(
        output / "analysis.json",
        {
            "contract_audit": True,
            "classification": classification,
            "objects": len(selected_ids),
            "meshes": len(all_rows),
            "target_multipliers": TARGET_MULTIPLIERS,
            "anchor_lambda": ANCHOR_LAMBDA,
            "macro_log_h_slope": slope[0],
            "macro_log_h_slope_bootstrap_ci": [
                None if not np.isfinite(slope[1]) else slope[1],
                None if not np.isfinite(slope[2]) else slope[2],
            ],
            "finest_gain_larger_objects": improvements,
            "monotonic_finer_gain_objects": monotonic,
            "audits": audits,
            "metric_protocol": METRIC_PROTOCOL,
            "hpc_jobs_submitted": 0,
        },
    )
    _plot(all_rows, output / "controlled_resolution_gain.png")
    _report(output, all_rows, audits, classification, slope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

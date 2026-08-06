#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Camera, Mesh
from mlr.learned_laplacian.dataset import load_prepared_sample, save_prepared_sample
from mlr.learned_laplacian.renderer_visibility import (
    compute_renderer_visibility,
    mesh_topology_orientation_diagnostics,
    visibility_statistics,
)
from mlr.synthetic import SyntheticRenderConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute renderer-native Sofa visibility without reading depth maps or "
            "regenerating coarse/expanded meshes."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "opengl"), default="opengl")
    parser.add_argument("--front-face-winding", choices=("ccw", "cw"), default="ccw")
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help="Restrict preparation to one manifest split.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    dataset_root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("samples")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty samples list.")
    if args.split is not None:
        records = [record for record in records if record.get("split") == args.split]
        if not records:
            raise ValueError(f"Manifest contains no samples for split {args.split!r}.")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive.")
        records = records[: args.limit]
    output_dir = (args.output_dir or dataset_root / "renderer_visibility").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for index, record in enumerate(records, start=1):
        sample_path = Path(record["path"])
        if not sample_path.is_absolute():
            sample_path = dataset_root / sample_path
        sample = load_prepared_sample(
            sample_path, materialize_images=False, dataset_root=dataset_root
        )
        sample_id = str(sample["sample_id"])
        artifact_path = output_dir / f"{sample_id}.npz"
        if artifact_path.exists() and not args.overwrite:
            raise FileExistsError(f"Visibility artifact already exists: {artifact_path}")
        mesh = Mesh(
            sample["vertices"].detach().cpu().numpy(),
            sample["faces"].detach().cpu().numpy(),
        ).ensure_normals()
        image_size = int(sample["prepared_image_size"])
        cameras = _cameras(sample, image_size)
        config = SyntheticRenderConfig(
            num_views=len(cameras),
            width=image_size,
            height=image_size,
            backend=args.backend,
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding=args.front_face_winding,
        )
        result = compute_renderer_visibility(
            mesh,
            cameras,
            config,
            neighborhood_radius=args.neighborhood_radius,
        )
        np.savez_compressed(
            artifact_path,
            frustum_valid=result.frustum_valid,
            visibility_backface_only=result.backface_visible,
            visibility_occlusion_only=result.occlusion_visible,
            visibility_backface_and_occlusion=result.backface_and_occlusion_visible,
        )
        stats = dict(visibility_statistics(result))
        stats.update(mesh_topology_orientation_diagnostics(mesh.faces))
        stats.update(
            {
                "sample_id": sample_id,
                "sample_path": str(sample_path),
                "artifact_path": str(artifact_path),
                "vertices": mesh.num_vertices,
                "faces": mesh.num_faces,
            }
        )
        summaries.append(stats)
        if args.attach:
            sample["visibility_backface_only"] = torch.from_numpy(
                result.backface_visible
            )
            sample["visibility_occlusion_only"] = torch.from_numpy(
                result.occlusion_visible
            )
            sample["visibility_backface_and_occlusion"] = torch.from_numpy(
                result.backface_and_occlusion_visible
            )
            sample["visibility"] = sample["visibility_backface_and_occlusion"]
            metadata = dict(sample.get("metadata", {}))
            metadata["renderer_visibility"] = {
                "definition": "depth_tested_face_id_incident_face_neighborhood",
                "artifact_path": str(artifact_path),
                "backend": args.backend,
                "front_face_winding": args.front_face_winding,
                "neighborhood_radius": args.neighborhood_radius,
                "depth_image_used": False,
                "mesh_identity": (
                    "visibility was rasterized from this sample's vertices and faces; "
                    "no GT depth, GT visibility, or correspondence was used"
                    if metadata.get("query_geometry_role") == "expanded_initial_raw"
                    else "visibility remains attached to the source GT vertex while its "
                    "query position may receive a small training perturbation"
                ),
            }
            sample["metadata"] = metadata
            save_prepared_sample(sample, sample_path)
        print(
            f"[{index}/{len(records)}] {sample_id} "
            f"final={stats['final_visible_ratio']:.4f} "
            f"zero={stats['zero_visible_vertex_ratio']:.4f}",
            flush=True,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 0


def _cameras(sample: dict, image_size: int) -> list[Camera]:
    intrinsics = sample["intrinsics"].detach().cpu().numpy()
    extrinsics = sample["extrinsics"].detach().cpu().numpy()
    return [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(image_size, image_size),
            name=f"view_{index:04d}",
        )
        for index in range(intrinsics.shape[0])
    ]


if __name__ == "__main__":
    raise SystemExit(main())

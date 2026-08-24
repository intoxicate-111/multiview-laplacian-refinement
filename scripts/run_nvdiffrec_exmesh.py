#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.cpp_extension


def _restore_cpp_extension_import_contract() -> None:
    """Keep extensions importable by name after Torch 2.13 JIT loading.

    The pinned nvdiffrec checkout intentionally discards the object returned by
    ``cpp_extension.load`` and immediately imports it again by ``name``.  Newer
    Torch loads the module but does not guarantee that second import contract.
    Registering the exact returned module restores the historical behavior
    without modifying the extension or the official nvdiffrec checkout.
    """

    original_load = torch.utils.cpp_extension.load
    if getattr(original_load, "_mlr_import_contract", False):
        return

    def load_and_register(*args, **kwargs):
        module = original_load(*args, **kwargs)
        name = kwargs.get("name", args[0] if args else module.__name__)
        sys.modules[str(name)] = module
        return module

    load_and_register._mlr_import_contract = True
    torch.utils.cpp_extension.load = load_and_register


def _add_xatlas_uvs(config_path: Path) -> None:
    """Add the UV atlas required by official DLMesh without changing geometry."""

    import xatlas

    config = json.loads(config_path.read_text(encoding="utf-8"))
    obj_path = Path(config["base_mesh"])
    lines = obj_path.read_text(encoding="utf-8").splitlines()
    if any(line.startswith("vt ") for line in lines):
        return
    vertices = np.asarray(
        [[float(value) for value in line.split()[1:4]] for line in lines if line.startswith("v ")],
        dtype=np.float32,
    )
    face_line_indices = [index for index, line in enumerate(lines) if line.startswith("f ")]
    faces = np.asarray(
        [
            [int(token.split("/", 1)[0]) - 1 for token in lines[index].split()[1:4]]
            for index in face_line_indices
        ],
        dtype=np.uint32,
    )
    _, texture_faces, uvs = xatlas.parametrize(vertices, faces)
    texture_faces = np.asarray(texture_faces, dtype=np.uint32)
    if texture_faces.shape != faces.shape:
        raise RuntimeError(
            f"xatlas changed face indexing shape: {texture_faces.shape} != {faces.shape}"
        )
    first_face = face_line_indices[0]
    uv_lines = [f"vt {float(uv[0]):.9g} {float(uv[1]):.9g}" for uv in uvs]
    lines[first_face:first_face] = uv_lines
    face_line_indices = [index + len(uv_lines) for index in face_line_indices]
    for face_number, line_index in enumerate(face_line_indices):
        tokens = lines[line_index].split()[1:]
        rewritten = []
        for corner, token in enumerate(tokens):
            components = token.split("/")
            position = components[0]
            normal = components[2] if len(components) > 2 else ""
            texture = str(int(texture_faces[face_number, corner]) + 1)
            rewritten.append(
                f"{position}/{texture}/{normal}" if normal else f"{position}/{texture}"
            )
        lines[line_index] = "f " + " ".join(rewritten)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official nvdiffrec with an exact ExMesh camera data adapter."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nvdiffrec-root", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _restore_cpp_extension_import_contract()
    sys.path.insert(0, str(args.nvdiffrec_root))
    try:
        config_index = remainder.index("--config")
        config_path = Path(remainder[config_index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("The exact-camera wrapper requires --config PATH.") from exc
    _add_xatlas_uvs(config_path)

    from dataset import dataset_nerf
    class DatasetExMeshExactCamera(dataset_nerf.DatasetNERF):
        def _parse_frame(self, cfg, idx):
            frame = cfg["frames"][idx]
            img = dataset_nerf._load_img(
                str(Path(self.base_dir) / frame["file_path"])
            )
            mv = torch.tensor(frame["world_to_camera_opengl"], dtype=torch.float32)
            k = np.asarray(frame["intrinsics"], dtype=np.float64)
            width, height = map(float, frame["resolution_wh"])
            near, far = map(float, self.FLAGS.cam_near_far)
            fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            proj = torch.tensor(
                [
                    [2 * fx / width, 0, 1 - 2 * cx / width, 0],
                    [0, -2 * fy / height, 1 - 2 * cy / height, 0],
                    [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                    [0, 0, -1, 0],
                ],
                dtype=torch.float32,
            )
            campos = torch.linalg.inv(mv)[:3, 3]
            mvp = proj @ mv
            return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]

    dataset_nerf.DatasetNERF = DatasetExMeshExactCamera
    sys.argv = [str(args.nvdiffrec_root / "train.py"), *remainder]
    runpy.run_path(str(args.nvdiffrec_root / "train.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

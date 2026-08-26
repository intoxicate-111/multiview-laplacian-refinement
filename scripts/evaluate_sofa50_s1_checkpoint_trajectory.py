#!/usr/bin/env python3
from __future__ import annotations

"""Read-only full-validation Hybrid Chamfer for one stored S1 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import differentiable_regularized_sparse_recovery_with_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_payload = _read(args.run.resolve() / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    settings = config["training"]["hybrid_single_geometry_loss"]
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(args.checkpoint.resolve(), model, map_location=device)
    model.eval()
    if not model.split_geometry_towers_enabled:
        raise RuntimeError("Checkpoint is not S1 split-geometry")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    rows = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(conditioned)
        direct = output.direct_vertex_displacement_prediction
        if direct is None:
            raise RuntimeError("Missing S1 direct output")
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            output.predicted_laplacian.detach().double(),
            prepared.sample["vertices"].double() + direct.detach().double(),
            prepared.sample["edge_index"], prepared.sample["vertex_degree"].double(),
            regularization=float(settings["lambda"]),
            maximum_iterations=int(settings["maximum_iterations"]),
            tolerance=float(settings["tolerance"]),
        )
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        metric = _geometry_row("validation", str(static["sample_id"]), args.label, Mesh(recovered.detach().cpu().numpy(), faces.copy()).ensure_normals(), clean, initial)
        rows.append({"sample_id": str(static["sample_id"]), "sample_index": index, "chamfer": float(metric["chamfer"]), "pcg_iterations": int(audit.iterations), "pcg_relative_residual": float(audit.relative_residual)})
        print(f"S1 trajectory {args.label} {index + 1}/{len(dataset)}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "read_only": True, "label": args.label, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()), "samples": len(rows),
        "validation_hybrid_chamfer": float(np.mean([row["chamfer"] for row in rows])),
        "pcg_iterations_mean": float(np.mean([row["pcg_iterations"] for row in rows])),
        "pcg_iterations_max": int(max(row["pcg_iterations"] for row in rows)),
        "pcg_relative_residual_max": float(max(row["pcg_relative_residual"] for row in rows)),
        "metric_protocol": METRIC_PROTOCOL, "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"trajectory_{args.label}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "validation_hybrid_chamfer": payload["validation_hybrid_chamfer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

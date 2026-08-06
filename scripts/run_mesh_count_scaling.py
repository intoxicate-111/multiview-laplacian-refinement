#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.scaling_diagnostics import run_mesh_count_scaling


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-manifest", required=True, type=Path)
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        help=(
            "Optional prepared coarse/expanded-query manifest. Omit it for a "
            "GT-query-only dataset such as raw sofa50."
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mesh-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--exposures-per-mesh", type=int, default=500)
    parser.add_argument("--accumulation-meshes", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_mesh_count_scaling(args.gt_manifest, args.expanded_manifest, args.config, args.output_dir, mesh_counts=args.mesh_counts, exposures_per_mesh=args.exposures_per_mesh, accumulation_meshes=args.accumulation_meshes, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

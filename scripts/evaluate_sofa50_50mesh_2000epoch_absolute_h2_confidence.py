#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.canonical_experiment import (
    run_canonical_experiment_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the canonical Sofa50 run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence"
        ),
    )
    parser.add_argument(
        "--gt-manifest",
        type=Path,
        default=Path(
            "/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
        ),
    )
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        default=Path(
            "/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    summary = run_canonical_experiment_evaluation(
        args.run_dir,
        args.gt_manifest,
        args.expanded_manifest,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "checkpoints": len(summary["checkpoint_metrics"]),
                "expanded_rows": len(summary["expanded_validation"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

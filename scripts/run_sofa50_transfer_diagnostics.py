#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.sofa50_transfer_diagnostics import (
    run_sofa50_transfer_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Sofa50 expanded-query gap and coordinate normalization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/learned_laplacian/"
            "train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
        ),
    )
    parser.add_argument(
        "--canonical-run-dir",
        type=Path,
        default=Path(
            "runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence"
        ),
    )
    parser.add_argument(
        "--gt-manifest",
        type=Path,
        default=Path(
            "/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/"
            "multiview_960/gt_query_manifest.json"
        ),
    )
    parser.add_argument(
        "--expanded-manifest",
        type=Path,
        default=Path(
            "/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/"
            "multiview_960/expanded_inference_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/learned_laplacian/sofa50_transfer_gap_diagnostics"
        ),
    )
    args = parser.parse_args()
    summary = run_sofa50_transfer_diagnostics(
        args.config,
        args.canonical_run_dir,
        args.gt_manifest,
        args.expanded_manifest,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "expanded_meshes": summary["expanded_validation_mesh_count"],
                "query_gap_clear": summary[
                    "query_gap_clearly_exceeds_training_field"
                ],
                "mesh_anomalies": summary["mesh_normalization"]["anomaly_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.current_graph_identity import (
    run_current_graph_identity_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the one-mesh Sofa50 current-graph identity diagnostic."
    )
    parser.add_argument(
        "--canonical-run-dir",
        type=Path,
        default=Path(
            "runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence"
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
            "runs/learned_laplacian/sofa50_current_graph_identity_diagnostic"
        ),
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Validation sample ID; defaults to the first manifest validation sample.",
    )
    parser.add_argument(
        "--no-unit-weight-control",
        action="store_true",
        help="Run only the primary visibility-times-confidence variant.",
    )
    args = parser.parse_args()
    summary = run_current_graph_identity_diagnostic(
        args.canonical_run_dir,
        args.expanded_manifest,
        args.output_dir,
        sample_id=args.sample_id,
        include_unit_weight_control=not args.no_unit_weight_control,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "selected_sample_id": summary["selected_sample_id"],
                "case": summary["case"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

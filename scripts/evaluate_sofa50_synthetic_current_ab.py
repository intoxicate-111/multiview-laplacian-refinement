from __future__ import annotations

import argparse
from pathlib import Path

from mlr.learned_laplacian.synthetic_current_comparison import (
    run_synthetic_current_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen GT-query A and new current-query B on one C-query test set."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a-checkpoint", type=Path, required=True)
    parser.add_argument("--a-config", type=Path, required=True)
    parser.add_argument("--b-checkpoint", type=Path, required=True)
    parser.add_argument("--b-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_synthetic_current_comparison(
        args.manifest,
        args.a_checkpoint,
        args.a_config,
        args.b_checkpoint,
        args.b_config,
        args.output_dir,
        device=args.device,
    )
    for name in ("A", "B"):
        row = result["aggregate"][name]
        print(
            f"{name}: mse={row['normalized_mse']:.8g} "
            f"vector_l2={row['vector_l2']:.8g} "
            f"cosine={row['global_cosine']:.8g} "
            f"chamfer={row['reconstruction_chamfer']:.8g} "
            f"improved={row['improved_over_initial']}/{row['sample_count']}"
        )


if __name__ == "__main__":
    main()

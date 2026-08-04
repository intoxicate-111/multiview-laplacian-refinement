from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.sample_io import prepare_single_object_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one learned-Laplacian training sample.")
    parser.add_argument("--dataset", required=True, type=Path, help="Synthetic dataset.json path.")
    parser.add_argument("--coarse-mesh", required=True, type=Path)
    parser.add_argument("--gt-mesh", type=Path, help="Defaults to dataset.json mesh_path.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, help="Optional square RGB resize with adjusted intrinsics.")
    parser.add_argument("--operator", default="uniform", choices=["uniform", "cotangent"])
    parser.add_argument("--distance-confidence-scale", type=float)
    parser.add_argument("--coarse-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target-mode",
        choices=["raw_laplacian", "edge_scale_normalized_laplacian"],
        default="raw_laplacian",
    )
    parser.add_argument("--edge-scale-epsilon", type=float, default=1e-12)
    args = parser.parse_args()
    sample = prepare_single_object_sample(
        dataset_path=args.dataset,
        coarse_mesh_path=args.coarse_mesh,
        gt_mesh_path=args.gt_mesh,
        output_path=args.output,
        image_size=args.image_size,
        operator_type=args.operator,
        distance_confidence_scale=args.distance_confidence_scale,
        coarse_noise_std=args.coarse_noise_std,
        seed=args.seed,
        target_mode=args.target_mode,
        edge_scale_epsilon=args.edge_scale_epsilon,
    )
    print(
        f"Saved {args.output}: views={sample['images'].shape[0]} "
        f"vertices={sample['vertices'].shape[0]} faces={sample['faces'].shape[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

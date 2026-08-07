#!/usr/bin/env bash
set -euo pipefail

MANIFEST="$HOME/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
BASE_CONFIG="configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"
OUTPUT_ROOT="runs/learned_laplacian/sofa50_image_resolution_ablation_50000step"

STEPS=50000
DEVICE="cuda"

for ARM in F0 F1 F2; do
    echo "========================================"
    echo "Starting ${ARM} - ${STEPS} optimizer steps"
    echo "========================================"

    PYTHONPATH=src conda run --no-capture-output -n test \
    python scripts/run_sofa50_image_resolution_ablation.py \
        --manifest "$MANIFEST" \
        --base-config "$BASE_CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --arm "$ARM" \
        --max-optimizer-steps "$STEPS" \
        --device "$DEVICE"

    echo "Finished ${ARM}"
done

echo "========================================"
echo "Running final analysis"
echo "========================================"

PYTHONPATH=src conda run --no-capture-output -n test \
python scripts/run_sofa50_image_resolution_ablation.py \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --analyze \
    --device "$DEVICE"

echo "Done."
echo "Report: $OUTPUT_ROOT/analysis/REPORT.md"
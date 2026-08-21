#!/bin/bash

#SBATCH --job-name=s50_gt_raw_eval
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
ROOT="${REPO}/runs/learned_laplacian/sofa50_gt_query_direct_raw_zero_shot_20k_seed7"
NEW_RUN="${ROOT}/B_gt_query_direct_raw_hf_2gpu"
GT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
CURRENT_MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json"
NORMALIZED_GT_RUN="${REPO}/runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4/views_28"
CURRENT_HF_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_image_feature_ablation_20k_seed7"
CURRENT_HF_RUN="${CURRENT_HF_ROOT}/C_original_plus_high_frequency_2gpu"
CURRENT_HF_ANALYSIS="${CURRENT_HF_ROOT}/analysis"
OUTPUT="${ROOT}/analysis"

for path in \
    "${NEW_RUN}/checkpoint_best.pt" \
    "${NEW_RUN}/metrics.json" \
    "${GT_MANIFEST}" \
    "${CURRENT_MANIFEST}" \
    "${NORMALIZED_GT_RUN}/checkpoint_best.pt" \
    "${CURRENT_HF_RUN}/checkpoint_best.pt" \
    "${CURRENT_HF_ANALYSIS}/recovery_per_sample.csv"
do
    test -e "${path}"
done
if [[ -e "${OUTPUT}/aggregate_summary.json" ]]; then
    echo "Refusing to overwrite completed evaluation: ${OUTPUT}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=7
export PYTHONPATH="${REPO}/src"
"${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_gt_raw_zero_shot_transfer.py \
        --new-run "${NEW_RUN}" \
        --gt-manifest "${GT_MANIFEST}" \
        --current-manifest "${CURRENT_MANIFEST}" \
        --normalized-gt-run "${NORMALIZED_GT_RUN}" \
        --current-hf-run "${CURRENT_HF_RUN}" \
        --current-hf-analysis "${CURRENT_HF_ANALYSIS}" \
        --output-dir "${OUTPUT}" \
        --device cuda \
        --seed 7

#!/bin/bash
#SBATCH --job-name=s50_naive_val_o
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
ROOT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1"
SPECIALISTS="${ROOT}/validation_selection/specialists"
OUTPUT="${ROOT}/naive_vertex_fusion_v1"
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=7
cd "${REPO}"

"${PYTHON}" scripts/evaluate_sofa50_naive_vertex_fusion.py \
  --domain old_native1920 --phase validation-shard \
  --manifest /networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json \
  --hybrid-summary "${ROOT}/validation_selection/frozen_lambda/lambda_selection.json" \
  --specialist-summary "${SPECIALISTS}/validation_specialist_summary.json" \
  --specialist-per-sample "${SPECIALISTS}/validation_specialist_per_sample.csv" \
  --specialist-predictions "${SPECIALISTS}/validation_specialist_predictions.npz" \
  --frozen-test-summary "${ROOT}/frozen_final_test/frozen_test_summary.json" \
  --frozen-test-mesh-root "${ROOT}/frozen_final_test/refined_meshes" \
  --output-dir "${OUTPUT}" --shard-count 5 --shard-index "${SLURM_ARRAY_TASK_ID}" --device cpu

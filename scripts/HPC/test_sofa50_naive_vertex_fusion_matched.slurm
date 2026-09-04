#!/bin/bash
#SBATCH --job-name=s50_naive_test_m
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/naive_vertex_fusion_v1"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=7
cd "${REPO}"

"${PYTHON}" scripts/evaluate_sofa50_naive_vertex_fusion.py \
  --domain matched_v2 --phase test-shard \
  --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
  --arm-b-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final" \
  --arm-e-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final" \
  --hybrid-summary "${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1/matched_summary.json" \
  --output-dir "${OUTPUT}" --shard-count 10 --shard-index "${SLURM_ARRAY_TASK_ID}" --device cpu

#!/bin/bash

#SBATCH --job-name=s50_ct_internal
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
ROOT="${REPO}/reports/sofa50_multitopology_rawlap500_v2"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8

"${PYTHON}" scripts/audit_sofa50_continuous_b_e_step0.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --config configs/learned_laplacian/train_sofa50_v2_continuous_b_e_hybrid_4blackwell.json \
    --arm-b-report "${ROOT}/recovery_aware_two_arm_ablation/final" \
    --arm-e-report "${ROOT}/direct_vertex_arm_e_extension/final" \
    --frozen-report "${ROOT}/frozen_hybrid_recovery_v1" \
    --tight-reference "${ROOT}/continuous_b_e_hybrid/preflight/tight_reproduction_validation.json" \
    --split validation --limit 1 \
    --output "${ROOT}/continuous_b_e_hybrid/preflight/internal_reproduction_validation_sample0.json" \
    --device cuda

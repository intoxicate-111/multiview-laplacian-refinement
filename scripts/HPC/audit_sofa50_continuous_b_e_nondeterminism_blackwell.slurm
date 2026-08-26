#!/bin/bash

#SBATCH --job-name=s50_ct_noise5
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
ROOT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/preflight"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=16
export PYTHONNOUSERSITE=1
unset PYTHONHOME

"${PYTHON}" scripts/audit_sofa50_continuous_b_e_nondeterminism.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --config configs/learned_laplacian/train_sofa50_v2_continuous_b_e_hybrid_4blackwell.json \
    --tight-reference "${ROOT}/tight_reproduction_validation.json" \
    --output "${ROOT}/nondeterminism_envelope_validation_5runs.json" \
    --realizations 5 --device cuda

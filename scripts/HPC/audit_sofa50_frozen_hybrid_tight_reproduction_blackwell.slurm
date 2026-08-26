#!/bin/bash

#SBATCH --job-name=s50_ct_tight_gate
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
B_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final"
E_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final"
FROZEN_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/preflight"
SPLITS=(validation test)
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"

for path in "${PYTHON}" "${MANIFEST}" "${B_REPORT}" "${E_REPORT}" "${FROZEN_REPORT}"; do
    test -e "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONHASHSEED=7 OMP_NUM_THREADS=8

"${PYTHON}" scripts/audit_sofa50_frozen_hybrid_tight_reproduction.py \
    --manifest "${MANIFEST}" \
    --arm-b-report "${B_REPORT}" \
    --arm-e-report "${E_REPORT}" \
    --frozen-report "${FROZEN_REPORT}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}/tight_reproduction_${SPLIT}.json" \
    --device cuda

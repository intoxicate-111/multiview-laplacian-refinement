#!/usr/bin/env bash
#SBATCH --job-name=s50_loss_mech
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02,gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
AB_REPORT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final
E_REPORT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final
S0_RUN=${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

if (( SLURM_ARRAY_TASK_ID < 5 )); then
    SPLIT=validation
    SHARD=${SLURM_ARRAY_TASK_ID}
else
    SPLIT=test
    SHARD=$((SLURM_ARRAY_TASK_ID - 5))
fi

"${PYTHON}" scripts/analyze_sofa50_loss_mechanisms.py \
    --phase shard \
    --manifest "${MANIFEST}" \
    --arm-ab-report "${AB_REPORT}" \
    --arm-e-report "${E_REPORT}" \
    --s0-run "${S0_RUN}" \
    --output-dir "${OUTPUT}" \
    --split "${SPLIT}" \
    --shard-count 5 \
    --shard-index "${SHARD}" \
    --chebyshev-order 128 \
    --device cuda

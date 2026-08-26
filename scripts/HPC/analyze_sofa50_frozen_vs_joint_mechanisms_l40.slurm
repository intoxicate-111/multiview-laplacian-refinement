#!/bin/bash

#SBATCH --job-name=s50_mech_geometry
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
B_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final"
E_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final"
JOINT_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1"

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 5 ]]; then
    SPLIT="validation"
    SHARD="${SLURM_ARRAY_TASK_ID}"
else
    SPLIT="test"
    SHARD="$((SLURM_ARRAY_TASK_ID - 5))"
fi

mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 PYTHONHASHSEED=7
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"${PYTHON}" scripts/analyze_sofa50_frozen_vs_joint_mechanisms.py \
    --phase shard --manifest "${MANIFEST}" --arm-b-report "${B_REPORT}" \
    --arm-e-report "${E_REPORT}" --joint-run "${JOINT_RUN}" \
    --output-dir "${OUTPUT}" --split "${SPLIT}" --shard-count 5 \
    --shard-index "${SHARD}" --chebyshev-order 128 --device cuda

#!/bin/bash
#SBATCH --job-name=s50_be_ood
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%8
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
B_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7"
E_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7"
OUTPUT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1/ood"
TASK="${SLURM_ARRAY_TASK_ID}"
if [[ "${TASK}" -eq 0 ]]; then
  NAME="legacy_v1"; MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json"; SHARDS=1; INDEX=0
elif [[ "${TASK}" -eq 1 ]]; then
  NAME="unseen_recipes_v1"; MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json"; SHARDS=1; INDEX=0
else
  NAME="future2000_x5"; MANIFEST="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"; SHARDS=8; INDEX="$((TASK - 2))"
fi
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}"
nvidia-smi
"${PYTHON}" scripts/evaluate_frozen_sofa50_b_e_ood.py \
  --domain-name "${NAME}" --manifest "${MANIFEST}" --arm-b-run "${B_RUN}" \
  --arm-e-run "${E_RUN}" --output-dir "${OUTPUT}" --shard-count "${SHARDS}" \
  --shard-index "${INDEX}" --device cuda

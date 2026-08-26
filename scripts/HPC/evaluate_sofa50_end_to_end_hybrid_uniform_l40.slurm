#!/bin/bash

#SBATCH --job-name=s50_uniform_eval
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CODE_REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
RUN="${REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/end_to_end_hybrid_single_loss/evaluation"

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    DOMAIN="matched_v2"
    MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
    SPLITS=(validation test)
elif [[ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]]; then
    DOMAIN="legacy_v1"
    MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json"
    SPLITS=(test)
else
    DOMAIN="unseen_recipes_v1"
    MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json"
    SPLITS=(test)
fi

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${CODE_REPO}"
export PYTHONPATH="${CODE_REPO}/src:${CODE_REPO}/scripts" OMP_NUM_THREADS=8 PYTHONHASHSEED=7
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"${PYTHON}" "${CODE_REPO}/scripts/evaluate_sofa50_uniform_vs_cotangent_hybrid.py" \
    --arm uniform --domain "${DOMAIN}" --run "${RUN}" \
    --manifest "${MANIFEST}" --output-dir "${OUTPUT}" --splits "${SPLITS[@]}" \
    --device cuda

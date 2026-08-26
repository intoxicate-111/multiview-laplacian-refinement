#!/bin/bash

#SBATCH --job-name=s50_u_c_eval
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
MAIN_REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
U_RUN="${MAIN_REPO}/runs/learned_laplacian/sofa50_v2_end_to_end_hybrid_single_loss_lambda3e-2_20k_seed7"
C_RUN="${REPO}/runs/learned_laplacian/sofa50_v2_cotangent_hybrid_selected_20k_seed7"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/uniform_vs_cotangent_single_loss/evaluation"

TASK="${SLURM_ARRAY_TASK_ID}"
if [[ "${TASK}" -lt 3 ]]; then
    ARM="uniform"; RUN="${U_RUN}"
else
    ARM="cotangent"; RUN="${C_RUN}"; TASK="$((TASK - 3))"
fi
if [[ "${TASK}" -eq 0 ]]; then
    DOMAIN="matched_v2"; MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"; SPLITS=(validation test)
elif [[ "${TASK}" -eq 1 ]]; then
    DOMAIN="legacy_v1"; MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json"; SPLITS=(test)
else
    DOMAIN="unseen_recipes_v1"; MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json"; SPLITS=(test)
fi

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=8 PYTHONHASHSEED=7
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$(dirname "${PYTHON}"):${PATH}"
nvidia-smi
"${PYTHON}" -c 'from pathlib import Path; import torch; expected=Path("/networkhome/WMGDS/zhou_c/miniconda3/envs/test"); actual=Path(torch.__file__).resolve(); assert str(actual).startswith(str(expected)), (actual, expected); assert torch.cuda.is_available(); assert "L40" in torch.cuda.get_device_name(0); print(torch.__version__, torch.version.cuda, actual, flush=True)'
"${PYTHON}" scripts/evaluate_sofa50_uniform_vs_cotangent_hybrid.py \
    --arm "${ARM}" --domain "${DOMAIN}" --run "${RUN}" \
    --manifest "${MANIFEST}" --output-dir "${OUTPUT}" --splits "${SPLITS[@]}" \
    --device cuda

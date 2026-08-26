#!/bin/bash

#SBATCH --job-name=s50_ctbe_meshes
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_continuous_pretrained_b_e_hybrid_lambda3e-2_lr1e-4_20k_seed7
CHECKPOINT=${RUN}/checkpoint_best.pt
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/final_evaluation/refined_test_meshes

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=12
export PYTHONHASHSEED=7
unset PYTHONHOME

test -f "${CHECKPOINT}"
test -f /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
test -f /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json
test -f /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/export_sofa50_continuous_b_e_test_meshes.py \
    --run-dir "${RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT}" \
    --dataset matched_v2 /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --dataset legacy_v1 /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json \
    --dataset unseen_recipes_v1 /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json \
    --device cuda

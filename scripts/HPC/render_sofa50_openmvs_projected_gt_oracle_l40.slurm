#!/bin/bash

#SBATCH --job-name=s50_openmvs_oracle_vis
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
ROOT=${ROOT:-${RUN}/diagnostics/openmvs48_projected_gt_oracle_v2}
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export OMP_NUM_THREADS=16
"${PYTHON}" scripts/render_sofa50_openmvs_projected_gt_oracle.py \
    --manifest "${RUN}/evaluation/openmvs48_prepared/manifest.json" \
    --diagnostic-root "${ROOT}" \
    --view 0 --image-size 480 --backend opengl

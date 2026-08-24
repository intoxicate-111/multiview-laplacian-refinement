#!/bin/bash

#SBATCH --job-name=s50_sparse_visuals
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
REPORT=${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/final
OUTPUT=${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/comparison_visuals

export PYTHONPATH=${REPO}/src
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl
cd "${REPO}"
"${PYTHON}" scripts/render_sofa50_recovery_aware_comparisons.py \
    --manifest "${MANIFEST}" \
    --report-dir "${REPORT}" \
    --output-dir "${OUTPUT}" \
    --views 0 7 --image-size 320 --expected-count 50

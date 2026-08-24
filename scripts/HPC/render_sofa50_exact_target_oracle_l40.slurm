#!/bin/bash

#SBATCH --job-name=s50_exact_render
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
V1_RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
V2_RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7
ROOT=${V2_RUN}/diagnostics/exact_target_oracle_matched_v1_v2
cd "${REPO}"
export PYTHONPATH=${REPO}/src
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl

"${PYTHON}" scripts/render_sofa50_exact_target_oracle.py \
    --v1-manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json \
    --v2-manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --v1-diagnostic-root "${ROOT}/v1_legacy_smoothing" \
    --v2-diagnostic-root "${ROOT}/v2_strong_smoothing" \
    --v1-prediction-source "${V1_RUN}/evaluation/in_domain" \
    --v2-prediction-source "${V2_RUN}/evaluation_v1_vs_v2_strong_smooth/raw" \
    --report-root "${ROOT}/final" \
    --backend opengl --view 0 --image-size 480

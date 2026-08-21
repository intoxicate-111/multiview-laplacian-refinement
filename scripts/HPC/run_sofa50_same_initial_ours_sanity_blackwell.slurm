#!/bin/bash
#SBATCH --job-name=s50_same_ours_bw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MANIFEST=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json
RUN=${PROJECT}/runs/learned_laplacian/sofa50_synthetic_current_28view_hf_resolution_1920_20k_seed7/HF_native1920_4gpu
OUTPUT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820/sanity/ours
SAMPLE=5ac05fe8-b550-4786-8e8c-c5a43c30d112__v01

test -f "${MANIFEST}"
test -f "${RUN}/checkpoint_latest.pt"
test ! -e "${OUTPUT}"
mkdir -p "${PROJECT}/slurm_logs"
cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=${PROJECT}/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" scripts/run_sofa50_same_initial_ours.py \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN}" \
    --output-dir "${OUTPUT}" \
    --sample-id "${SAMPLE}" \
    --device cuda

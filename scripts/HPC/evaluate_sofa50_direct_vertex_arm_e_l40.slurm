#!/bin/bash

#SBATCH --job-name=s50_direct_vertex_e_eval
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUN="${REPO}/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7"
OUTPUT="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
export PYTHONPATH="${REPO}/src" OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_direct_vertex_arm_e.py \
  --manifest "${MANIFEST}" --run "${RUN}" --output-dir "${OUTPUT}" --device cuda

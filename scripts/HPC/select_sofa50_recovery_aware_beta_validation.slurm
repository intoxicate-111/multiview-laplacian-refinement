#!/bin/bash

#SBATCH --job-name=s50_beta_val
#SBATCH --partition=short
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# The short partition has a one-hour hard limit.  This stage only integrates
# archived validation predictions and evaluates 50 meshes per array task.
#SBATCH --time=00:59:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
RUNS="${REPO}/runs/learned_laplacian"
OUTPUT="${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/beta_validation"
export PYTHONPATH="${REPO}/src" OMP_NUM_THREADS=8
mkdir -p "${OUTPUT}/candidates" "${REPO}/slurm_logs"
cd "${REPO}"
"${PYTHON}" scripts/select_sofa50_recovery_aware_beta.py \
    --manifest "${MANIFEST}" --runs-root "${RUNS}" --output-dir "${OUTPUT}" \
    --candidate-index "${SLURM_ARRAY_TASK_ID}"

#!/bin/bash
#SBATCH --job-name=s50_be_matched
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
AD="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/final"
AE="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
OUTPUT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1/matched"
mkdir -p "${OUTPUT}/shards" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}"
"${PYTHON}" scripts/diagnose_sofa50_representation_b_vs_e.py \
  --manifest "${MANIFEST}" --ad-report-dir "${AD}" --ae-report-dir "${AE}" \
  --output-dir "${OUTPUT}" --shard-count 5 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --chebyshev-order 128

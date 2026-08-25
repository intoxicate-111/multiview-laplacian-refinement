#!/bin/bash
#SBATCH --job-name=s50_be_match_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
AD="${REPO}/runs/diagnostics/sofa50_recovery_aware_lambda_extension/final"
AE="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
OUTPUT="${REPO}/runs/diagnostics/sofa50_representation_b_vs_e_frozen_v1/matched"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO}"
"${PYTHON}" scripts/diagnose_sofa50_representation_b_vs_e.py \
  --manifest "${MANIFEST}" --ad-report-dir "${AD}" --ae-report-dir "${AE}" \
  --output-dir "${OUTPUT}" --shard-count 5 --chebyshev-order 128 --merge-only

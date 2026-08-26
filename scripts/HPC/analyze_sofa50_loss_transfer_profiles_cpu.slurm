#!/usr/bin/env bash
#SBATCH --job-name=s50_loss_xfer
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/loss_mechanism_audit_v1
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
"${PYTHON}" scripts/analyze_sofa50_loss_transfer_profiles.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --output-dir "${OUTPUT}" \
    --device cpu \
    --indices 0,12,24,36,49 \
    --probes 3 \
    --chebyshev-order 128

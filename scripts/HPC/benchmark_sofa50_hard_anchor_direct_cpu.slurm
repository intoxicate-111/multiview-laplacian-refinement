#!/bin/bash
#SBATCH --job-name=sofa50_i_direct
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
OUTPUT="$REPO/runs/diagnostics/sofa50_hard_anchor_lambda0/direct_solver_benchmark/benchmark.json"

cd "$REPO"
mkdir -p "$REPO/slurm_logs" "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$REPO/src:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/benchmark_sofa50_hard_anchor_direct.py \
  --manifest "$MANIFEST" \
  --output "$OUTPUT"

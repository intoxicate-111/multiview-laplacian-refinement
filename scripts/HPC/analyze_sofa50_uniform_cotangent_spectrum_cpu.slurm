#!/usr/bin/env bash
#SBATCH --job-name=s50-u-cot-spectrum
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${REPO}"
mkdir -p logs
"${PYTHON}" scripts/analyze_sofa50_uniform_cotangent_spectrum_correspondence.py \
  --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
  --modes-per-band 8 \
  --eigensolver-tolerance 1e-6 \
  --eigensolver-maximum-iterations 20000 \
  --output-dir "${REPO}/reports/sofa50_multitopology_rawlap500_v2/uniform_cotangent_spectrum_correspondence_v1"

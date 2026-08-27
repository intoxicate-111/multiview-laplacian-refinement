#!/usr/bin/env bash
#SBATCH --job-name=sofa-curv-benefit
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
mkdir -p logs
"${PYTHON}" scripts/analyze_sofa50_curvature_conditioned_benefit.py \
  --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
  --arm-b-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final" \
  --arm-e-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final" \
  --output-dir "${REPO}/reports/sofa50_multitopology_rawlap500_v2/curvature_conditioned_benefit_v1"

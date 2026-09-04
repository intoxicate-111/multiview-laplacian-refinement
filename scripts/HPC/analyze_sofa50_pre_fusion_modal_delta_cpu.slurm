#!/bin/bash
#SBATCH --job-name=sofa50_prefusion_delta
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/sofa50_prefusion_delta_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/sofa50_prefusion_delta_%j.err

set -euo pipefail

cd /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
mkdir -p slurm_logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=src:scripts

/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python \
  scripts/analyze_sofa50_pre_fusion_modal_delta.py \
  --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
  --arm-b-report /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final \
  --arm-e-report /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final \
  --output-dir /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/pre_fusion_modal_delta_v1 \
  --bins 36 \
  --chebyshev-order 384 \
  --bootstrap-draws 10000 \
  --seed 7

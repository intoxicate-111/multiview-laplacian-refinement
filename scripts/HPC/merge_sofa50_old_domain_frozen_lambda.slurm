#!/bin/bash

#SBATCH --job-name=s50_old1920_lammerge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/validation_selection/frozen_lambda"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=2
cd "${REPO}"
"${PYTHON}" scripts/sweep_sofa50_old_domain_frozen_lambda.py \
  --mode merge --output-dir "${OUTPUT}" --shard-count 5

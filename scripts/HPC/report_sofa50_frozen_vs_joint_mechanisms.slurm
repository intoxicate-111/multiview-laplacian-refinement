#!/bin/bash

#SBATCH --job-name=s50_mech_report
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=4 PYTHONHASHSEED=7
"${PYTHON}" scripts/analyze_sofa50_frozen_vs_joint_mechanisms.py \
    --phase merge --output-dir "${OUTPUT}" --shard-count 5
"${PYTHON}" scripts/report_sofa50_frozen_vs_joint_mechanisms.py \
    --input-dir "${OUTPUT}" --output-dir "${OUTPUT}"

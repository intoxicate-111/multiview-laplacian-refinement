#!/usr/bin/env bash
#SBATCH --job-name=s50_s1_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid/selected_analysis
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
"${PYTHON}" scripts/analyze_sofa50_s1_selected.py --phase merge --output-dir "${OUTPUT}" --shard-count 5

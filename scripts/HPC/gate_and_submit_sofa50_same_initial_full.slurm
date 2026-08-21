#!/bin/bash
#SBATCH --job-name=s50_same_gate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
cd "${PROJECT}"
bash scripts/HPC/submit_sofa50_same_initial_full.sh

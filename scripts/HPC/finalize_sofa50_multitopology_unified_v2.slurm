#!/bin/bash

#SBATCH --job-name=s50_mt_final_v2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
DATASET=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1
UNSEEN=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1
RUN=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7
cd "${REPO}"
"${PYTHON}" scripts/generate_sofa50_multitopology_final_report.py \
    --repo-root "${REPO}" --dataset-root "${DATASET}" --unseen-root "${UNSEEN}" \
    --run-root "${RUN}" --evaluation-root "${RUN}/evaluation_unified_v2" \
    --require-unified-geometry \
    --output-dir "${REPO}/reports/sofa50_multitopology_rawlap500_v1/final_unified_v2"

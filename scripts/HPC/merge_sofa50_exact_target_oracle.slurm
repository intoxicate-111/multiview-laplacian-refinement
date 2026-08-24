#!/bin/bash

#SBATCH --job-name=s50_exact_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v2_strong_smooth_hf_20k_seed7/diagnostics/exact_target_oracle_matched_v1_v2
V1=${ROOT}/v1_legacy_smoothing
V2=${ROOT}/v2_strong_smoothing
FINAL=${ROOT}/final
cd "${REPO}"
export PYTHONPATH=${REPO}/src

"${PYTHON}" scripts/diagnose_sofa50_exact_target_oracle.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json \
    --dataset-arm v1_legacy_smoothing --output-dir "${V1}" --shard-count 4 --merge-only
"${PYTHON}" scripts/diagnose_sofa50_exact_target_oracle.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --dataset-arm v2_strong_smoothing --output-dir "${V2}" --shard-count 4 --merge-only
"${PYTHON}" scripts/generate_sofa50_exact_target_oracle_report.py \
    --v1-root "${V1}" --v2-root "${V2}" --output-dir "${FINAL}"

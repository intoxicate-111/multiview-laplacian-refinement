#!/bin/bash

#SBATCH --job-name=s50_mt500_v2_audit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2"

cd "${REPO}"
export PYTHONPATH="${REPO}/src"
"${PYTHON}" scripts/prepare_sofa50_multitopology_rawlap500.py \
    --source-manifest "${SOURCE}" \
    --output-root "${OUTPUT}" \
    --shard-count 2 \
    --smoothing-profile strong_smooth_v2 \
    --merge-only

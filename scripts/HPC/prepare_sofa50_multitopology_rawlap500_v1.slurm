#!/bin/bash

#SBATCH --job-name=s50_mt500_prep
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

"${PYTHON}" scripts/prepare_sofa50_multitopology_rawlap500.py \
    --source-manifest "${SOURCE}" \
    --output-root "${OUTPUT}" \
    --downstream-root "${REPO}" \
    --upstream-root "/networkhome/WMGDS/zhou_c/data_prepare" \
    --shard-count 2 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --seed 7 \
    --smoothing-profile legacy_v1 \
    --visibility-backend opengl \
    --resume \
    "${LIMIT_ARGS[@]}"

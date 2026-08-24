#!/bin/bash

#SBATCH --job-name=s50_mt_u25
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1"

cd "${REPO}"
export PYTHONPATH="${REPO}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"

"${PYTHON}" scripts/prepare_sofa50_multitopology_rawlap500.py \
    --source-manifest "${SOURCE}" \
    --output-root "${OUTPUT}" \
    --downstream-root "${REPO}" \
    --upstream-root "/networkhome/WMGDS/zhou_c/data_prepare" \
    --dataset-mode unseen \
    --smoothing-profile legacy_v1 \
    --visibility-backend opengl \
    --shard-count 1 \
    --shard-index 0 \
    --seed 7 \
    --resume

"${PYTHON}" scripts/prepare_sofa50_multitopology_rawlap500.py \
    --source-manifest "${SOURCE}" \
    --output-root "${OUTPUT}" \
    --dataset-mode unseen \
    --smoothing-profile legacy_v1 \
    --shard-count 1 \
    --merge-only

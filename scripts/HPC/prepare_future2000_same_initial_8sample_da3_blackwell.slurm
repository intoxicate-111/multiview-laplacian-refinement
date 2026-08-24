#!/bin/bash
#SBATCH --job-name=f2k8_da3_bw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-04
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/f2k8_da3_blackwell_20260822/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG=${PROJECT}/configs/baselines/future2000_same_initial_8sample_blackwell.json
SELECTION=${PROJECT}/configs/baselines/future2000_same_initial_8sample_selection.json
DA3=/networkhome/WMGDS/zhou_c/external_baselines/depth-anything-3
ROOT=${PROJECT}/runs/benchmarks/future2000_same_initial_8sample_20260822
OUTPUT=${ROOT}/exmesh_da3_priors
INDEX=${SLURM_ARRAY_TASK_ID}

SAMPLE_ID="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["sample_ids"][int(sys.argv[2])])' "${SELECTION}" "${INDEX}")"
OBJECT_ID=${SAMPLE_ID%%__v*}
mkdir -p "${PROJECT}/slurm_logs" "${OUTPUT}"
cd "${PROJECT}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HF_HOME=/networkhome/WMGDS/zhou_c/.cache/huggingface
export PYTHONPATH=${PROJECT}/src:${DA3}/src
"${PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --da3-root "${DA3}" \
    --output-dir "${OUTPUT}" \
    --shard-index "${INDEX}" \
    --shard-count 8 \
    --expected-test-samples 1000 \
    --expected-test-objects 200 \
    --object-id "${OBJECT_ID}" \
    --device cuda

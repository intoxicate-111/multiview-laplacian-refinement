#!/bin/bash
#SBATCH --job-name=f2k_mt20k_smoke
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/gt_query_manifest.json"
OBSERVATIONS="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/runs/diagnostics/Future2000MultiTopologyRawLap20000_v1_smoke_${SLURM_JOB_ID}"
UPSTREAM="/networkhome/WMGDS/zhou_c/data_prepare"
NVDIFFRAST="/networkhome/WMGDS/zhou_c/dmesh2/external/nvdiffrast"

cd "${REPO}"
mkdir -p "${REPO}/slurm_logs" "${OUTPUT}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${UPSTREAM}/src:${NVDIFFRAST}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

nvidia-smi
"${PYTHON}" scripts/prepare_future2000_multitopology_rawlap20k.py \
    --source-manifest "${SOURCE}" \
    --observation-manifest "${OBSERVATIONS}" \
    --output-root "${OUTPUT}" \
    --downstream-root "${REPO}" \
    --upstream-root "${UPSTREAM}" \
    --shard-count 1 \
    --shard-index 0 \
    --limit 1 \
    --seed 7 \
    --max-vertices 500000 \
    --visibility-backend cuda \
    --resume

"${PYTHON}" -c 'import json,sys; p=sys.argv[1]; a=json.load(open(p)); assert len(a["samples"]) == 10; assert all(x["contract_pass"] for x in a["samples"]); print({"smoke_samples": 10, "contract_audit": True})' \
    "${OUTPUT}/shards/audit_shard_00.json"
du -sh "${OUTPUT}"

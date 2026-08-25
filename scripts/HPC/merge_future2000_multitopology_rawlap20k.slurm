#!/bin/bash
#SBATCH --job-name=f2k_mt20k_audit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/future2000_compact/multiview_960/gt_query_manifest.json"
OBSERVATIONS="/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json"
OUTPUT="/networkhome/WMGDS/zhou_c/Future2000MultiTopologyRawLap20000_v1"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" scripts/prepare_future2000_multitopology_rawlap20k.py \
    --source-manifest "${SOURCE}" \
    --observation-manifest "${OBSERVATIONS}" \
    --output-root "${OUTPUT}" \
    --shard-count 16 \
    --merge-only

"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["contract_audit"] is True; assert r["sample_count"] == 20000; print({"sample_count": 20000, "contract_audit": True})' \
    "${OUTPUT}/full_audit.json"
du -sh "${OUTPUT}"

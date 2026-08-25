#!/bin/bash

#SBATCH --job-name=s50_cot_audit
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement-cotangent"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/uniform_vs_cotangent_single_loss/operator_audit"

mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export OMP_NUM_THREADS=32 PYTHONHASHSEED=7

"${PYTHON}" scripts/audit_sofa50_cotangent_operator.py \
    --manifest "${MANIFEST}" --output-dir "${OUTPUT}" \
    --representative-per-split 3 --lambda 3e-2 \
    --relative-area-epsilon 1e-12
"${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"]; assert p["meshes"] == 500' "${OUTPUT}/audit.json"

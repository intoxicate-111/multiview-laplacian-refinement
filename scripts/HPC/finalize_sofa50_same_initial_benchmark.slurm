#!/bin/bash
#SBATCH --job-name=s50_same_finalize
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:59:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820
REPORT=${ROOT}/full_report

cd "${PROJECT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=${PROJECT}/src
"${PYTHON}" scripts/aggregate_sofa50_same_initial_benchmark.py \
    --manifest "${ROOT}/benchmark_manifest.json" \
    --config "${PROJECT}/configs/baselines/sofa50_same_initial_benchmark.json" \
    --results-root "${ROOT}/full" \
    --output-dir "${REPORT}" \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01

if "${PYTHON}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["contract_audit"] else 2)' "${REPORT}/summary.json"; then
    "${PYTHON}" scripts/render_sofa50_same_initial_benchmark.py \
        --manifest "${ROOT}/benchmark_manifest.json" \
        --results-root "${ROOT}/full" \
        --per-sample-csv "${REPORT}/per_sample.csv" \
        --output-dir "${REPORT}/visualizations" \
        --count 5 \
        --image-size 360 \
        --view 0
else
    echo "Contract audit is false; retaining aggregate failure report and skipping incomplete comparison panels." >&2
fi

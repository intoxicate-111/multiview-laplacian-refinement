#!/bin/bash
#SBATCH --job-name=s50_ext56_stress
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
EXTERNAL=/networkhome/WMGDS/zhou_c/external_baselines
OLD_ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820
ROOT=${PROJECT}/runs/benchmarks/sofa50_external56_stress_20260826
INPUT=${ROOT}/input
MANIFEST=${INPUT}/benchmark_manifest.json
CONFIG=${PROJECT}/configs/baselines/sofa50_same_initial_56v_external_stress.json
RESULTS=${ROOT}/external
PRIORS=${ROOT}/exmesh_da3_priors
REPORT=${ROOT}/report
SOURCE56=/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_56_manifest.json
PROJECT_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
DA3_PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/future_da3/bin/python
DA3_ROOT=${EXTERNAL}/depth-anything-3

mkdir -p "${PROJECT}/slurm_logs" "${ROOT}/logs" "${RESULTS}" "${PRIORS}" "${REPORT}"
cd "${PROJECT}"
export PYTHONPATH=${PROJECT}/src
export PYTHONHASHSEED=7
export HF_HOME=/networkhome/WMGDS/zhou_c/.cache/huggingface
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

for path in \
    "${OLD_ROOT}/benchmark_manifest.json" \
    "${SOURCE56}" \
    "${CONFIG}" \
    "${PROJECT_PYTHON}" \
    "${DA3_PYTHON}"; do
    test -e "${path}"
done

echo "stage=prepare_native1920_56v start=$(date --iso-8601=seconds)"
"${PROJECT_PYTHON}" scripts/prepare_sofa50_same_initial_56v_native1920.py \
    --old-benchmark-manifest "${OLD_ROOT}/benchmark_manifest.json" \
    --source-56-manifest "${SOURCE56}" \
    --output-root "${INPUT}" \
    --render-workers 5
"${PROJECT_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] and p["sample_count"] == 25 and p["external_view_count"] == 56' "${INPUT}/contract_audit.json"
echo "stage=prepare_native1920_56v complete=$(date --iso-8601=seconds)"

echo "stage=da3_56v start=$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=0 "${DA3_PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --da3-root "${DA3_ROOT}" \
    --output-dir "${PRIORS}" --shard-index 0 --shard-count 2 \
    --expected-test-samples 25 --expected-test-objects 5 --expected-view-count 56 \
    --device cuda >"${ROOT}/logs/da3_worker0.log" 2>&1 &
DA3_0=$!
CUDA_VISIBLE_DEVICES=1 "${DA3_PYTHON}" scripts/prepare_future2000_exmesh_da3_priors.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --da3-root "${DA3_ROOT}" \
    --output-dir "${PRIORS}" --shard-index 1 --shard-count 2 \
    --expected-test-samples 25 --expected-test-objects 5 --expected-view-count 56 \
    --device cuda >"${ROOT}/logs/da3_worker1.log" 2>&1 &
DA3_1=$!
wait "${DA3_0}"
wait "${DA3_1}"
"${PROJECT_PYTHON}" -c 'import json,sys,pathlib; r=pathlib.Path(sys.argv[1]); rows=[json.load(open(p)) for p in sorted(r.glob("*/metadata.json"))]; assert len(rows)==5 and all(x["status"]=="completed" and x["view_count"]==56 for x in rows)' "${PRIORS}"
echo "stage=da3_56v complete=$(date --iso-8601=seconds)"

STAGE_ROOT=${SLURM_TMPDIR:-/tmp}/sofa50-ext56-${SLURM_JOB_ID}
mkdir -p "${STAGE_ROOT}"

gate_method() {
    local method=$1
    "${PROJECT_PYTHON}" -c 'import json,sys,pathlib; root=pathlib.Path(sys.argv[1])/sys.argv[2]/"samples"; rows=[json.load(open(p)) for p in sorted(root.glob("*/status.json"))]; normalized=[x.get("row",x) for x in rows]; assert len(normalized)==25 and all(x.get("status")=="completed" and int(x.get("view_count"))==56 for x in normalized)' "${RESULTS}" "${method}"
}

echo "stage=nds_56v start=$(date --iso-8601=seconds)"
for worker in 0 1; do
    cp -a "${EXTERNAL}/neural-deferred-shading" "${STAGE_ROOT}/nds_${worker}"
    patch --batch --forward -d "${STAGE_ROOT}/nds_${worker}" -p1 < "${PROJECT}/scripts/HPC/patches/nds_allow_nonmanifold_edges.patch"
done
CUDA_VISIBLE_DEVICES=0 PATH="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin:${PATH}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method nds \
    --external-root "${STAGE_ROOT}/nds_0" \
    --external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin/python \
    --output-dir "${RESULTS}" --shard-index 0 --shard-count 2 \
    --expected-test-samples 25 --surface-samples 3000 --metric-seed 7 \
    --fscore-threshold 0.01 --retry-failed >"${ROOT}/logs/nds_worker0.log" 2>&1 &
NDS_0=$!
CUDA_VISIBLE_DEVICES=1 PATH="/networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin:${PATH}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method nds \
    --external-root "${STAGE_ROOT}/nds_1" \
    --external-python /networkhome/WMGDS/zhou_c/miniconda3/envs/future_nds/bin/python \
    --output-dir "${RESULTS}" --shard-index 1 --shard-count 2 \
    --expected-test-samples 25 --surface-samples 3000 --metric-seed 7 \
    --fscore-threshold 0.01 --retry-failed >"${ROOT}/logs/nds_worker1.log" 2>&1 &
NDS_1=$!
wait "${NDS_0}"
wait "${NDS_1}"
gate_method nds
echo "stage=nds_56v complete=$(date --iso-8601=seconds)"

echo "stage=nvdiffrec_56v start=$(date --iso-8601=seconds)"
NVD_ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/nvdiffrec_official
CUDA_VISIBLE_DEVICES=0 CUDA_HOME="${NVD_ENV}" PATH="${NVD_ENV}/bin:${PATH}" TCNN_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
LD_LIBRARY_PATH="${NVD_ENV}/lib:${NVD_ENV}/lib64:${LD_LIBRARY_PATH:-}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method nvdiffrec \
    --external-root "${EXTERNAL}/nvdiffrec" --external-python "${NVD_ENV}/bin/python" \
    --output-dir "${RESULTS}" --shard-index 0 --shard-count 2 \
    --expected-test-samples 25 --surface-samples 3000 --metric-seed 7 \
    --fscore-threshold 0.01 --retry-failed >"${ROOT}/logs/nvdiffrec_worker0.log" 2>&1 &
NVD_0=$!
CUDA_VISIBLE_DEVICES=1 CUDA_HOME="${NVD_ENV}" PATH="${NVD_ENV}/bin:${PATH}" TCNN_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
LD_LIBRARY_PATH="${NVD_ENV}/lib:${NVD_ENV}/lib64:${LD_LIBRARY_PATH:-}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method nvdiffrec \
    --external-root "${EXTERNAL}/nvdiffrec" --external-python "${NVD_ENV}/bin/python" \
    --output-dir "${RESULTS}" --shard-index 1 --shard-count 2 \
    --expected-test-samples 25 --surface-samples 3000 --metric-seed 7 \
    --fscore-threshold 0.01 --retry-failed >"${ROOT}/logs/nvdiffrec_worker1.log" 2>&1 &
NVD_1=$!
wait "${NVD_0}"
wait "${NVD_1}"
gate_method nvdiffrec
echo "stage=nvdiffrec_56v complete=$(date --iso-8601=seconds)"

echo "stage=exmesh_56v start=$(date --iso-8601=seconds)"
EXM_ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official
CUDA_VISIBLE_DEVICES=0 CUDA_HOME="${EXM_ENV}" PATH="${EXM_ENV}/bin:${PATH}" \
LD_LIBRARY_PATH="${EXM_ENV}/lib:${EXM_ENV}/lib64:${LD_LIBRARY_PATH:-}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method exmesh \
    --external-root "${EXTERNAL}/ExMesh" --external-python "${EXM_ENV}/bin/python" \
    --exmesh-depth-root "${PRIORS}" --output-dir "${RESULTS}" \
    --shard-index 0 --shard-count 2 --expected-test-samples 25 \
    --surface-samples 3000 --metric-seed 7 --fscore-threshold 0.01 \
    --retry-failed >"${ROOT}/logs/exmesh_worker0.log" 2>&1 &
EXM_0=$!
CUDA_VISIBLE_DEVICES=1 CUDA_HOME="${EXM_ENV}" PATH="${EXM_ENV}/bin:${PATH}" \
LD_LIBRARY_PATH="${EXM_ENV}/lib:${EXM_ENV}/lib64:${LD_LIBRARY_PATH:-}" \
"${PROJECT_PYTHON}" scripts/evaluate_future2000_external_baseline.py \
    --manifest "${MANIFEST}" --config "${CONFIG}" --method exmesh \
    --external-root "${EXTERNAL}/ExMesh" --external-python "${EXM_ENV}/bin/python" \
    --exmesh-depth-root "${PRIORS}" --output-dir "${RESULTS}" \
    --shard-index 1 --shard-count 2 --expected-test-samples 25 \
    --surface-samples 3000 --metric-seed 7 --fscore-threshold 0.01 \
    --retry-failed >"${ROOT}/logs/exmesh_worker1.log" 2>&1 &
EXM_1=$!
wait "${EXM_0}"
wait "${EXM_1}"
gate_method exmesh
echo "stage=exmesh_56v complete=$(date --iso-8601=seconds)"

echo "stage=aggregate start=$(date --iso-8601=seconds)"
"${PROJECT_PYTHON}" scripts/aggregate_sofa50_56v_external_stress.py \
    --manifest "${MANIFEST}" \
    --external-results-root "${RESULTS}" \
    --old-results-root "${OLD_ROOT}/full" \
    --output-dir "${REPORT}" \
    --surface-samples 3000 --metric-seed 7 --fscore-threshold 0.01
"${PROJECT_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["contract_audit"] and p["sample_count"] == 25' "${REPORT}/summary.json"
echo "stage=aggregate complete=$(date --iso-8601=seconds)"
echo "pipeline=complete report=${REPORT}/FINAL_REPORT.md"

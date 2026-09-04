#!/bin/bash

#SBATCH --job-name=s50_anchor_report
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
ARM_AB_REPORT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final
ARM_E_REPORT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final
BP_RUN=${REPO}/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_bp_positional_anchor_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/anchor_conditioning_ablation_v1
TRAIN_STDOUT=${REPO}/slurm_logs/s50_anchor_bp_17886.out
TRAIN_STDERR=${REPO}/slurm_logs/s50_anchor_bp_17886.err
ENV_TMP=${SLURM_TMPDIR:-/tmp}/s50_anchor_report_environment_${SLURM_JOB_ID}.txt

for path in \
    "${PYTHON}" "${MANIFEST}" \
    "${ARM_AB_REPORT}/shards/A_lap_only.json" \
    "${ARM_AB_REPORT}/shards/B_lap_plus_refine.json" \
    "${ARM_E_REPORT}/shards/E_direct_vertex_residual.json" \
    "${BP_RUN}/checkpoint_best.pt" \
    "${BP_RUN}/metrics.json" \
    "${BP_RUN}/pretraining_gradient_audit.json" \
    "${TRAIN_STDOUT}" "${TRAIN_STDERR}"; do
    test -f "${path}"
done
if [[ -d "${OUTPUT}" && -n "$(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty report namespace: ${OUTPUT}" >&2
    exit 2
fi
test ! -e "${OUTPUT}" || test -d "${OUTPUT}"

mkdir -p "${REPO}/slurm_logs"
export PYTHONPATH=${REPO}/src:${REPO}/scripts
export PYTHONHASHSEED=7
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
cd "${REPO}"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURM_JOB_NODELIST}"
    echo "evaluation_device=cpu"
    echo "allocated_gpus=0"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK}"
    echo "bootstrap_replicates=10000 seed=7"
    echo "training_job=17886 training_exit_code=0 training_state=COMPLETED"
    git rev-parse HEAD
    "${PYTHON}" --version
    uname -a
    sha256sum \
        "${MANIFEST}" \
        "${BP_RUN}/checkpoint_best.pt" \
        "${BP_RUN}/launch_config.json" \
        "${BP_RUN}/pretraining_gradient_audit.json"
} > "${ENV_TMP}"

"${PYTHON}" scripts/evaluate_sofa50_anchor_conditioning_ablation.py \
    --manifest "${MANIFEST}" \
    --arm-ab-report "${ARM_AB_REPORT}" \
    --arm-e-report "${ARM_E_REPORT}" \
    --bp-run "${BP_RUN}" \
    --bp-checkpoint "${BP_RUN}/checkpoint_best.pt" \
    --output-dir "${OUTPUT}" \
    --device cpu \
    --bootstrap-replicates 10000 \
    --seed 7

test -f "${OUTPUT}/REPORT.md"
test -f "${OUTPUT}/summary.json"
test -f "${OUTPUT}/per_mesh_metrics.csv"
test -f "${OUTPUT}/anchor_interaction_results.json"
cp "${ENV_TMP}" "${OUTPUT}/evaluation_environment_${SLURM_JOB_ID}.txt"
cp "${TRAIN_STDOUT}" "${OUTPUT}/training_job_17886.out"
cp "${TRAIN_STDERR}" "${OUTPUT}/training_job_17886.err"
sha256sum "${OUTPUT}"/* > "${OUTPUT}/ARTIFACT_SHA256SUMS.txt"
echo "Anchor-conditioning report completed: ${OUTPUT}"

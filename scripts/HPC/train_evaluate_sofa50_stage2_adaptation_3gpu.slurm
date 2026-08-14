#!/bin/bash

#SBATCH --job-name=train_B_s2_3arm
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
EXPERIMENT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7"
DATASET="${EXPERIMENT}/dataset"
H2_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7"
B_CHECKPOINT="${H2_ROOT}/B_direct_raw_laplacian/checkpoint_latest.pt"
BASELINE_CSV="${H2_ROOT}/analysis/topk_oracle_replacement_per_sample.csv"
CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_synthetic_current_stage2_continuation_20k.json"
TRAIN_SCRIPT="${REPO}/scripts/train_multi_mesh_laplacian.py"
EVAL_SCRIPT="${REPO}/scripts/evaluate_sofa50_synthetic_current_stage2_adaptation.py"
ARMS=(continue_original continue_B_result continue_mix_50_50)
MANIFESTS=(continue_original_manifest.json continue_B_result_manifest.json continue_mix_50_50_manifest.json)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
MANIFEST="${DATASET}/manifests/${MANIFESTS[$SLURM_ARRAY_TASK_ID]}"
STAGE2_MANIFEST="${DATASET}/manifests/stage2_manifest.json"
TRAIN_OUTPUT="${EXPERIMENT}/training/${ARM}"
EVAL_OUTPUT="${EXPERIMENT}/evaluation/${ARM}"

for path in "${B_CHECKPOINT}" "${BASELINE_CSV}" "${CONFIG}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${MANIFEST}" "${STAGE2_MANIFEST}" "${DATASET}/contract_audit.json"; do
    test -f "${path}"
done
if [[ -f "${TRAIN_OUTPUT}/metrics.json" || -f "${EVAL_OUTPUT}/arm_summary.json" ]]; then
    echo "Refusing to overwrite completed arm ${ARM}" >&2
    exit 2
fi
mkdir -p "${REPO}/slurm_logs" "${TRAIN_OUTPUT}" "${EVAL_OUTPUT}"
cd "${REPO}"
nvidia-smi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYOPENGL_PLATFORM="egl"
PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${TRAIN_SCRIPT}" \
        --manifest "${MANIFEST}" \
        --config "${CONFIG}" \
        --output-dir "${TRAIN_OUTPUT}" \
        --device cuda \
        --resume-checkpoint "${B_CHECKPOINT}" \
        --reset-resume-tracking

PYTHONPATH="${REPO}/src" "${CONDA}" run --no-capture-output -n test \
    python "${EVAL_SCRIPT}" \
        --experiment-dir "${EXPERIMENT}" \
        --stage2-manifest "${STAGE2_MANIFEST}" \
        --baseline-csv "${BASELINE_CSV}" \
        --arm-run-dir "${TRAIN_OUTPUT}" \
        --output-dir "${EVAL_OUTPUT}" \
        --arm "${ARM}" \
        --device cuda

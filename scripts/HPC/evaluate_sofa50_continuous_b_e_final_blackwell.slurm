#!/bin/bash

#SBATCH --job-name=s50_ct_be_final
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python
MATCHED_MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json
LEGACY_MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v1/manifest.json
UNSEEN_MANIFEST=/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLapUnseen25_v1/manifest.json
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_continuous_pretrained_b_e_hybrid_lambda3e-2_lr1e-4_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/final_evaluation
PREFLIGHT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/preflight
STEP0=${RUN}/checkpoints/checkpoint_step_000000.pt
BEST=${RUN}/checkpoint_best.pt
STEPS=(000000 000100 000200 000500 001000 002000 003000 005000 007500 010000 015000 020000)
MECHANISM_STEPS=(000000 001000 005000 010000 020000)

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=12
export PYTHONHASHSEED=7
unset PYTHONHOME
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"

test -f "${RUN}/metrics.json"
test -f "${RUN}/training_history.json"
test -f "${STEP0}"
test -f "${BEST}"
for path in "${MATCHED_MANIFEST}" "${LEGACY_MANIFEST}" "${UNSEEN_MANIFEST}"; do
    test -f "${path}"
done
"${PYTHON}" -c 'import json, pathlib, torch; p=pathlib.Path("'"${RUN}"'/metrics.json"); m=json.loads(p.read_text()); assert m["optimizer_steps"] == 20000; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'

# Full matched validation trajectory.  This is the only selection domain.
for STEP in "${STEPS[@]}"; do
    CHECKPOINT=${RUN}/checkpoints/checkpoint_step_${STEP}.pt
    TARGET=${OUTPUT}/validation_step${STEP}.json
    test -f "${CHECKPOINT}"
    if [[ ! -s "${TARGET}" ]]; then
        "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
            --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
            --checkpoint "${CHECKPOINT}" --label "step${STEP}" \
            --split validation --output "${TARGET}" --device cuda
    fi
done
if [[ ! -s "${OUTPUT}/validation_best.json" ]]; then
    "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
        --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --label best --split validation \
        --output "${OUTPUT}/validation_best.json" --device cuda
fi

# Only after the validation-selected checkpoint is fixed, evaluate test and OOD.
if [[ ! -s "${OUTPUT}/matched_test_best.json" ]]; then
    "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
        --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --label best --split test \
        --output "${OUTPUT}/matched_test_best.json" --device cuda
fi
if [[ ! -s "${OUTPUT}/legacy_v1_test_best.json" ]]; then
    "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
        --manifest "${LEGACY_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --label best --split test \
        --output "${OUTPUT}/legacy_v1_test_best.json" --device cuda
fi
if [[ ! -s "${OUTPUT}/unseen_recipes_v1_test_best.json" ]]; then
    "${PYTHON}" scripts/evaluate_sofa50_continuous_checkpoint_validation.py \
        --manifest "${UNSEEN_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --label best --split test \
        --output "${OUTPUT}/unseen_recipes_v1_test_best.json" --device cuda
fi

# Fixed-subset branch/gradient/drift/redundancy trajectory.
for STEP in "${MECHANISM_STEPS[@]}"; do
    CHECKPOINT=${RUN}/checkpoints/checkpoint_step_${STEP}.pt
    TARGET=${OUTPUT}/mechanism_validation_step${STEP}.json
    if [[ ! -s "${TARGET}" ]]; then
        "${PYTHON}" scripts/analyze_sofa50_continuous_mechanism_checkpoint.py \
            --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
            --checkpoint "${CHECKPOINT}" --step0-checkpoint "${STEP0}" \
            --label "step${STEP}" --split validation --output "${TARGET}" --device cuda
    fi
done
if [[ ! -s "${OUTPUT}/mechanism_validation_best.json" ]]; then
    "${PYTHON}" scripts/analyze_sofa50_continuous_mechanism_checkpoint.py \
        --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --step0-checkpoint "${STEP0}" \
        --label best --split validation \
        --output "${OUTPUT}/mechanism_validation_best.json" --device cuda
fi

# All 50 matched test meshes for paired, spectral, and component diagnostics.
if [[ ! -s "${OUTPUT}/mechanism_test_step000000.json" ]]; then
    "${PYTHON}" scripts/analyze_sofa50_continuous_mechanism_checkpoint.py \
        --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
        --checkpoint "${STEP0}" --step0-checkpoint "${STEP0}" \
        --label step000000 --split test --indices all \
        --output "${OUTPUT}/mechanism_test_step000000.json" --device cuda
fi
if [[ ! -s "${OUTPUT}/mechanism_test_best.json" ]]; then
    "${PYTHON}" scripts/analyze_sofa50_continuous_mechanism_checkpoint.py \
        --manifest "${MATCHED_MANIFEST}" --run "${RUN}" \
        --checkpoint "${BEST}" --step0-checkpoint "${STEP0}" \
        --label best --split test --indices all \
        --output "${OUTPUT}/mechanism_test_best.json" --device cuda
fi

"${PYTHON}" scripts/report_sofa50_continuous_b_e_final.py \
    --evaluation-root "${OUTPUT}" --preflight-root "${PREFLIGHT}" --run "${RUN}"

sha256sum "${BEST}" "${OUTPUT}/final_evaluation.json" "${OUTPUT}/FINAL_REPORT.md" \
    > "${OUTPUT}/SHA256SUMS.txt"

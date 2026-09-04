#!/bin/bash
#SBATCH --job-name=s50_naive_merge
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
: "${DOMAIN:?Set DOMAIN=matched_v2 or old_native1920}"
: "${PHASE:?Set PHASE=merge-validation or merge-test}"
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" OMP_NUM_THREADS=2 PYTHONHASHSEED=7
cd "${REPO}"

if [[ "${DOMAIN}" == "matched_v2" ]]; then
  "${PYTHON}" scripts/evaluate_sofa50_naive_vertex_fusion.py \
    --domain matched_v2 --phase "${PHASE}" \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --arm-b-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final" \
    --arm-e-report "${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final" \
    --hybrid-summary "${REPO}/reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1/matched_summary.json" \
    --output-dir "${REPO}/reports/sofa50_multitopology_rawlap500_v2/naive_vertex_fusion_v1" --shard-count 10
elif [[ "${DOMAIN}" == "old_native1920" ]]; then
  ROOT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1"
  SPECIALISTS="${ROOT}/validation_selection/specialists"
  "${PYTHON}" scripts/evaluate_sofa50_naive_vertex_fusion.py \
    --domain old_native1920 --phase "${PHASE}" \
    --manifest /networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json \
    --hybrid-summary "${ROOT}/validation_selection/frozen_lambda/lambda_selection.json" \
    --specialist-summary "${SPECIALISTS}/validation_specialist_summary.json" \
    --specialist-per-sample "${SPECIALISTS}/validation_specialist_per_sample.csv" \
    --specialist-predictions "${SPECIALISTS}/validation_specialist_predictions.npz" \
    --frozen-test-summary "${ROOT}/frozen_final_test/frozen_test_summary.json" \
    --frozen-test-mesh-root "${ROOT}/frozen_final_test/refined_meshes" \
    --output-dir "${ROOT}/naive_vertex_fusion_v1" --shard-count 5
else
  echo "Unsupported DOMAIN=${DOMAIN}" >&2
  exit 2
fi

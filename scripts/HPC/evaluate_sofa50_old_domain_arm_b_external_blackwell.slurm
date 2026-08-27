#!/bin/bash

#SBATCH --job-name=s50_oldB_extcmp
#SBATCH --partition=test
#SBATCH --nodelist=gpu-04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json"
BENCHMARK="${REPO}/runs/benchmarks/sofa50_same_initial_20260820/benchmark_manifest.json"
ARCHIVE="${REPO}/runs/benchmarks/sofa50_same_initial_20260820/full"
AUTHORIZATION="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/arm_b_external_comparison_authorization.json"
RUN_B="${REPO}/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1"
OUTPUT="${REPO}/reports/sofa50_old_domain_native1920_b_e_v1/arm_b_external_comparison_v1"

for path in "${PYTHON}" "${MANIFEST}" "${BENCHMARK}" "${AUTHORIZATION}" \
  "${RUN_B}/checkpoint_best.pt"; do
  test -f "${path}"
done
test ! -e "${OUTPUT}"
mkdir -p "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts" PYTHONHASHSEED=7 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO}"
"${PYTHON}" scripts/evaluate_sofa50_old_domain_arm_b_external_comparison.py \
  --manifest "${MANIFEST}" --benchmark-manifest "${BENCHMARK}" \
  --authorization "${AUTHORIZATION}" --arm-b-run "${RUN_B}" \
  --archive-root "${ARCHIVE}" --output-dir "${OUTPUT}" --device cuda

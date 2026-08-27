#!/bin/bash

#SBATCH --job-name=s50_op_spectrum
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
B_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final"
E_REPORT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final"
OUTPUT="${REPO}/reports/sofa50_multitopology_rawlap500_v2/recovery_operator_spectrum_v1"

for path in "${PYTHON}" "${MANIFEST}" "${B_REPORT}" "${E_REPORT}"; do
    test -e "${path}"
done
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export PYTHONHASHSEED=7 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

"${PYTHON}" scripts/analyze_sofa50_recovery_operator_spectrum.py \
    --manifest "${MANIFEST}" \
    --arm-b-report "${B_REPORT}" \
    --arm-e-report "${E_REPORT}" \
    --output-dir "${OUTPUT}" \
    --chebyshev-order 128

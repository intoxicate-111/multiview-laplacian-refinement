#!/bin/bash

#SBATCH --job-name=s50_sparse_audit
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
RUNS=${REPO}/runs/learned_laplacian
OUTPUT=${REPO}/runs/diagnostics/sofa50_recovery_aware_ablation/implementation_contract_audit.json

export PYTHONPATH=${REPO}/src
cd ${REPO}
${PYTHON} scripts/audit_sofa50_recovery_aware_contract.py \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --run-a ${RUNS}/sofa50_v2_sparse_recovery_arm_a_lap_only_20k_seed7 \
    --run-b ${RUNS}/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7 \
    --arm-a-actual-config ${RUNS}/sofa50_v2_sparse_recovery_arm_a_lap_only_20k_seed7/resume_config_bw8.json \
    --output ${OUTPUT} \
    --device cuda \
    --sample-indices 0,1,2

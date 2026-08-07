#!/usr/bin/env bash
set -euo pipefail

mkdir -p slurm_logs

TRAIN_SCRIPT="scripts/slurm_jobs/sofa50_resolution_arm_50k.slurm"
ANALYSIS_SCRIPT="scripts/slurm_jobs/sofa50_resolution_analyze.slurm"

JOB_F0=$(sbatch --parsable --job-name=sofa50_F0_50k "${TRAIN_SCRIPT}" F0)
JOB_F1=$(sbatch --parsable --job-name=sofa50_F1_50k "${TRAIN_SCRIPT}" F1)
JOB_F2=$(sbatch --parsable --job-name=sofa50_F2_50k "${TRAIN_SCRIPT}" F2)

echo "Submitted:"
echo "  F0: ${JOB_F0}"
echo "  F1: ${JOB_F1}"
echo "  F2: ${JOB_F2}"

JOB_ANALYSIS=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB_F0}:${JOB_F1}:${JOB_F2} \
    "${ANALYSIS_SCRIPT}")

echo
echo "Analysis job: ${JOB_ANALYSIS}"
echo "Dependency: after F0/F1/F2 all succeed"
echo
echo "Use:"
echo "  squeue -u \$USER"
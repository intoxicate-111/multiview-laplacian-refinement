#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/multiview-laplacian-refinement}"
MANIFEST="${MANIFEST:-$HOME/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/learned_laplacian/sofa50_c1_f2_oracle_residual_expert_2000step}"
CONDA_ENV="${CONDA_ENV:-test}"
STEPS="${STEPS:-2000}"

cd "$PROJECT_ROOT"
mkdir -p slurm_logs

# Put the two .slurm files under $PROJECT_ROOT/hpc/ or override these paths.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_ROOT/hpc/sofa50_c1_f2_expert_2000.slurm}"
ANALYZE_SCRIPT="${ANALYZE_SCRIPT:-$PROJECT_ROOT/hpc/sofa50_c1_f2_expert_2000_analyze.slurm}"

test -f "$TRAIN_SCRIPT"
test -f "$ANALYZE_SCRIPT"
test -f "$MANIFEST"

EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,MANIFEST=$MANIFEST,OUTPUT_ROOT=$OUTPUT_ROOT,CONDA_ENV=$CONDA_ENV,STEPS=$STEPS"

TRAIN_JOB="$(sbatch --parsable --export="$EXPORTS" "$TRAIN_SCRIPT")"
echo "Submitted paired training array: $TRAIN_JOB"

ANALYZE_JOB="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" --export="$EXPORTS" "$ANALYZE_SCRIPT")"
echo "Submitted dependent analysis:    $ANALYZE_JOB"
echo "Output root: $OUTPUT_ROOT"
echo
echo "Monitor with:"
echo "  squeue -j $TRAIN_JOB,$ANALYZE_JOB"

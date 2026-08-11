#!/bin/bash

#SBATCH --job-name=eval_cur28_jit
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
SOFA="/networkhome/WMGDS/zhou_c/sofa_mesh"
CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"
RUN_ROOT="${REPO}/runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7"
OUTPUT="${RUN_ROOT}/analysis"

cd "${REPO}"
nvidia-smi

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_local_query_jitter_ablation.py \
        --manifest "${SOFA}/sofa50_synthetic_current_28view_v1/manifest.json" \
        --a-run "${RUN_ROOT}/A_no_jitter" \
        --b-run "${RUN_ROOT}/B_local_jitter" \
        --output-dir "${OUTPUT}/prediction" \
        --device cuda

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/evaluate_sofa50_openmvs48_local_query_jitter_ablation.py \
        --repo-root "${REPO}" \
        --source-manifest "${SOFA}/sofa50_refinement/multiview_nested_14_28_56_cpu_v3/gt_query_views_28_manifest.json" \
        --coarse-models-root "${SOFA}/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models" \
        --a-run "${RUN_ROOT}/A_no_jitter" \
        --b-run "${RUN_ROOT}/B_local_jitter" \
        --output-dir "${OUTPUT}/openmvs48" \
        --split test \
        --device cuda \
        --visibility-backend opengl \
        --visibility-size 480 \
        --require-all

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
    python scripts/analyze_sofa50_local_query_jitter_ablation.py \
        --prediction-summary "${OUTPUT}/prediction/prediction_summary.json" \
        --openmvs-summary "${OUTPUT}/openmvs48/openmvs_summary.json" \
        --output "${OUTPUT}/REPORT.md"

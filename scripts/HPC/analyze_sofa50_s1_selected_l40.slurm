#!/usr/bin/env bash
#SBATCH --job-name=s50_s1_eval
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02,gpu-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%A_%a.err

set -euo pipefail
REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
RUN=${REPO}/runs/learned_laplacian/sofa50_v2_s1_split_geometry_hybrid_lambda3e-2_20k_seed7
OUTPUT=${REPO}/reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid/selected_analysis
if (( SLURM_ARRAY_TASK_ID < 5 )); then
    SPLIT=validation
    SHARD=${SLURM_ARRAY_TASK_ID}
else
    SPLIT=test
    SHARD=$((SLURM_ARRAY_TASK_ID - 5))
fi
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
"${PYTHON}" scripts/analyze_sofa50_s1_selected.py \
    --phase shard \
    --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json \
    --run "${RUN}" \
    --output-dir "${OUTPUT}" \
    --split "${SPLIT}" \
    --shard-count 5 \
    --shard-index "${SHARD}" \
    --chebyshev-order 128 \
    --device cuda

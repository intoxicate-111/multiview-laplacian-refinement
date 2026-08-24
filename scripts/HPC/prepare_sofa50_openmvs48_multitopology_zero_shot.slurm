#!/bin/bash

#SBATCH --job-name=s50_mt_openmvs_prep
#SBATCH --partition=xlong
#SBATCH --nodelist=gpu-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
SOURCE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
COARSE="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/openmvs_texture_test_v6_48view/reconstruction/models"
OUTPUT="${REPO}/runs/learned_laplacian/sofa50_multitopology_rawlap500_v1_hf_20k_seed7/evaluation/openmvs48_prepared"

cd "${REPO}"
mkdir -p "${OUTPUT}" "${REPO}/slurm_logs"
export PYTHONPATH="${REPO}/src:${REPO}/scripts"
export OMP_NUM_THREADS=12
"${PYTHON}" scripts/prepare_sofa50_openmvs48_multitopology_zero_shot.py \
    --repo-root "${REPO}" \
    --source-manifest "${SOURCE}" \
    --coarse-models-root "${COARSE}" \
    --output-dir "${OUTPUT}" \
    --visibility-backend opengl \
    --visibility-size 480

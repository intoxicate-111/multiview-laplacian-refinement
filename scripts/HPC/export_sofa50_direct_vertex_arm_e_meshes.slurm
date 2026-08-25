#!/bin/bash

#SBATCH --job-name=s50_direct_vertex_e_meshes
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail
REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python"
MANIFEST="/networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json"
REPORT="${REPO}/runs/diagnostics/sofa50_direct_vertex_arm_e_extension/final"
OUTPUT="${REPORT}/all_test_meshes"

test ! -e "${OUTPUT}"
export PYTHONPATH="${REPO}/src"
cd "${REPO}"
"${PYTHON}" scripts/export_sofa50_direct_vertex_arm_e_meshes.py \
  --manifest "${MANIFEST}" --arm-e-report "${REPORT}" --output-dir "${OUTPUT}"

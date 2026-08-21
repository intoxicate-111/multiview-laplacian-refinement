#!/bin/bash
#SBATCH --job-name=exmesh_data
#SBATCH --partition=xlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
EXMESH="/networkhome/WMGDS/zhou_c/external_baselines/ExMesh"
DOWNLOAD="${EXMESH}/downloads_official"
TOOLS="${DOWNLOAD}/python_tools"
PYTHON="/networkhome/WMGDS/zhou_c/miniconda3/bin/python"

mkdir -p "${PROJECT}/slurm_logs" "${DOWNLOAD}" "${TOOLS}"
"${PYTHON}" -m pip install --no-cache-dir --target "${TOOLS}" gdown==5.2.0

# Source 1: the exact 2DGS-preprocessed DTU observations linked by ExMesh.
mkdir -p "${DOWNLOAD}/2dgs"
PYTHONPATH="${TOOLS}" "${PYTHON}" -m gdown --folder --remaining-ok \
    "https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9" \
    -O "${DOWNLOAD}/2dgs"

# Source 2: the exact precomputed DA3 priors linked by ExMesh.
PYTHONPATH="${TOOLS}" "${PYTHON}" -m gdown --fuzzy \
    "https://drive.google.com/file/d/1KupSX6O1Ocq3N0n-V1_dNKCenA_eDRCm/view?usp=sharing" \
    -O "${DOWNLOAD}/exmesh_dtu_da3"

# Source 3: official DTU evaluation masks/calibration and reference points.
wget --continue --output-document="${DOWNLOAD}/SampleSet.zip" \
    "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
wget --continue --output-document="${DOWNLOAD}/Points.zip" \
    "https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"

find "${DOWNLOAD}" -maxdepth 4 -type f -printf '%s %p\n' | sort -n \
    > "${DOWNLOAD}/download_manifest.txt"
sha256sum \
    "${DOWNLOAD}/2dgs/dtu.tar.gz" \
    "${DOWNLOAD}/2dgs/dtu_results.zip" \
    "${DOWNLOAD}/exmesh_dtu_da3" \
    "${DOWNLOAD}/SampleSet.zip" \
    "${DOWNLOAD}/Points.zip" \
    > "${DOWNLOAD}/sha256sums.txt"
echo "Download complete. Extraction is intentionally a separate audited step."

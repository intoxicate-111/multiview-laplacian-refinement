#!/bin/bash
#SBATCH --job-name=exmesh_hf_bw_env
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

SOURCE_PYTHON=/networkhome/WMGDS/zhou_c/video_sr_hpc/envs/uav_bw/bin/python
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/mlr_blackwell
EVAL_ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_blackwell
OFFICIAL_EVAL_ENV=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement

test -x "${SOURCE_PYTHON}"
mkdir -p "${PROJECT}/slurm_logs"
if [[ ! -x "${ENV}/bin/python" ]]; then
    "${SOURCE_PYTHON}" -m venv --system-site-packages "${ENV}"
fi
"${ENV}/bin/python" -m pip install --disable-pip-version-check 'moderngl==5.12.0'
if [[ ! -x "${EVAL_ENV}/bin/python" ]]; then
    "${CONDA}" create -y -p "${EVAL_ENV}" --clone "${OFFICIAL_EVAL_ENV}"
fi
test "$("${EVAL_ENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.11"
"${EVAL_ENV}/bin/python" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.13.0'
PYTHONPATH="${PROJECT}/src" "${ENV}/bin/python" - <<'PY'
import moderngl
import numpy
import torch
from PIL import Image
from mlr.learned_laplacian.multi_trainer import _build_model

assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 7)
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "moderngl": moderngl.__version__,
    "numpy": numpy.__version__,
    "pillow": Image.__version__,
})
PY
"${EVAL_ENV}/bin/python" - <<'PY'
import cv2
import imageio
import numpy
import open3d
import scipy
import skimage
import sklearn
import torch
import tqdm
import trimesh

print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "imageio": imageio.__version__,
    "open3d": open3d.__version__,
    "scipy": scipy.__version__,
    "skimage": skimage.__version__,
    "sklearn": sklearn.__version__,
    "tqdm": tqdm.__version__,
    "trimesh": trimesh.__version__,
})
PY

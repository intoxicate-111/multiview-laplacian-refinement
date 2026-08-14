#!/bin/bash
#SBATCH --job-name=f2k_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
CONDA=/networkhome/WMGDS/zhou_c/miniconda3/bin/conda
MANIFEST=/networkhome/WMGDS/zhou_c/future2000_gt_adaptive_synthetic_current_28view_v2/manifest.json
CONFIG="${SMOKE_CONFIG:-${REPO}/configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json}"
ROOT="${SMOKE_ROOT:-${REPO}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7/analysis/smoke}"
INPUTS="${ROOT}/inputs"
RUN="${ROOT}/run"

mkdir -p "${REPO}/slurm_logs" "${INPUTS}" "${RUN}"
cd "${REPO}"

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
  python scripts/prepare_future2000_training_smoke.py \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --output-dir "${INPUTS}"

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
  python scripts/train_multi_mesh_laplacian.py \
    --manifest "${INPUTS}/manifest.json" \
    --config "${INPUTS}/config_phase1.json" \
    --output-dir "${RUN}" \
    --device cuda

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test python -c \
  'import json, pathlib, torch; root=pathlib.Path("'"${RUN}"'"); checkpoint=torch.load(root/"checkpoint_latest.pt",map_location="cpu",weights_only=False); metrics=json.loads((root/"metrics.json").read_text()); assert checkpoint["optimizer_steps"]==5; assert metrics["optimizer_steps"]==5; assert metrics["final_validation_loss"] is not None; (root/"phase1_verification.json").write_text(json.dumps({"checkpoint_steps":checkpoint["optimizer_steps"],"validation_loss":metrics["final_validation_loss"]},indent=2)+"\n")'

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test \
  python scripts/train_multi_mesh_laplacian.py \
    --manifest "${INPUTS}/manifest.json" \
    --config "${INPUTS}/config_phase2.json" \
    --output-dir "${RUN}" \
    --device cuda \
    --resume-checkpoint "${RUN}/checkpoint_latest.pt"

PYTHONPATH=src "${CONDA}" run --no-capture-output -n test python -c \
  'import json, math, pathlib, torch; root=pathlib.Path("'"${RUN}"'"); checkpoint=torch.load(root/"checkpoint_latest.pt",map_location="cpu",weights_only=False); metrics=json.loads((root/"metrics.json").read_text()); history=json.loads((root/"training_history.json").read_text()); result={"passed":checkpoint["optimizer_steps"]==10 and metrics["optimizer_steps"]==10 and metrics["continuation_optimizer_steps"]==5 and math.isfinite(metrics["final_train_loss"]) and math.isfinite(metrics["final_validation_loss"]) and metrics["mean_train_views_per_sample"]==28.0 and metrics["peak_gpu_memory_mb"]>0,"checkpoint_steps":checkpoint["optimizer_steps"],"continuation_optimizer_steps":metrics["continuation_optimizer_steps"],"train_loss":metrics["final_train_loss"],"validation_loss":metrics["final_validation_loss"],"views_per_sample":metrics["mean_train_views_per_sample"],"peak_gpu_memory_mb":metrics["peak_gpu_memory_mb"],"history_rows":len(history)}; (root/"smoke_test.json").write_text(json.dumps(result,indent=2)+"\n"); assert result["passed"]'

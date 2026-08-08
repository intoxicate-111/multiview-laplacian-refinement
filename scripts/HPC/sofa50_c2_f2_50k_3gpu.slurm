#!/bin/bash

#SBATCH --job-name=sofa_c2f2_50k
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:3
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

# ============================================================
# Paths
# ============================================================

REPO="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"

MANIFEST="/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"

BASE_CONFIG="${REPO}/configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json"

OUTPUT_ROOT="${REPO}/runs/learned_laplacian/sofa50_c2_f2_50000step_3seed"

CONDA="/networkhome/WMGDS/zhou_c/miniconda3/bin/conda"

STEPS=50000

# Three independent replicas
SEEDS=(7 17 27)

cd "${REPO}"

mkdir -p "${REPO}/slurm_logs"
mkdir -p "${OUTPUT_ROOT}"

echo "============================================================"
echo "Sofa50 C2 + F2 / 50k / 3 GPU"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Seeds:     ${SEEDS[*]}"
echo "============================================================"

nvidia-smi

# ============================================================
# Build one config per seed
# ============================================================

for SEED in "${SEEDS[@]}"; do

    RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    CONFIG="${RUN_DIR}/launch_config.json"

    mkdir -p "${RUN_DIR}"

    python - "${BASE_CONFIG}" "${CONFIG}" "${SEED}" "${STEPS}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])
steps = int(sys.argv[4])

cfg = json.loads(src.read_text())

# ============================================================
# Seed
# ============================================================

cfg["seed"] = seed

# ============================================================
# C2 capacity
# ============================================================

cfg.setdefault("image_encoder", {})["feature_dim"] = 64

cfg.setdefault("model", {})["hidden_dim"] = 256
cfg["model"]["num_graph_layers"] = 3

# ============================================================
# F2 image feature resolution
# ============================================================

cfg["image_encoder"]["first_stride"] = 1
cfg["image_encoder"]["second_stride"] = 1

# ============================================================
# Residual expert OFF
# ============================================================

cfg["model"].pop("oracle_residual_expert", None)

# ============================================================
# Match clean F-resolution evaluation contract
# ============================================================

cfg.setdefault("query_training", {})["apply_to_validation"] = False

cfg.setdefault("training", {})["vertex_sampling"] = {
    "mode": "full"
}

# ============================================================
# 50k optimizer steps
#
# Optimizer steps are the hard stopping condition.
# Large epoch ceiling prevents premature epoch termination.
# ============================================================

multi = cfg.setdefault("multi_object_training", {})
multi["epochs"] = 6000
multi["max_optimizer_steps"] = steps

# Do not use epoch-based periodic checkpoints here.
multi["checkpoint_every_epochs"] = 0
multi["checkpoint_epochs"] = []

dst.write_text(json.dumps(cfg, indent=2) + "\n")

print("")
print("Generated:", dst)
print("seed              =", cfg["seed"])
print("feature_dim       =", cfg["image_encoder"]["feature_dim"])
print("hidden_dim        =", cfg["model"]["hidden_dim"])
print("graph_layers      =", cfg["model"]["num_graph_layers"])
print("first_stride      =", cfg["image_encoder"]["first_stride"])
print("second_stride     =", cfg["image_encoder"]["second_stride"])
print("residual expert   =", cfg["model"].get("oracle_residual_expert", "OFF"))
print("confidence        =", cfg.get("confidence", {}).get("enabled"))
print("val augmentation  =", cfg["query_training"]["apply_to_validation"])
print("vertex sampling   =", cfg["training"]["vertex_sampling"])
print("max steps         =", cfg["multi_object_training"]["max_optimizer_steps"])
print("")
PY

done

# ============================================================
# Launch 3 independent C2+F2 runs
#
# Each srun step gets:
#   1 x L40
#   8 CPUs
#
# Slurm assigns a different GPU to each exclusive step.
# ============================================================

PIDS=()

for SEED in "${SEEDS[@]}"; do

    RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
    CONFIG="${RUN_DIR}/launch_config.json"

    echo "Launching seed ${SEED}"

    srun \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --gres=gpu:L40:1 \
        bash -lc "
            cd '${REPO}'

            echo '============================================================'
            echo 'Starting seed ${SEED}'
            echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}\"
            echo '============================================================'

            PYTHONPATH=src '${CONDA}' run \
                --no-capture-output \
                -n test \
                python scripts/train_multi_mesh_laplacian.py \
                    --manifest '${MANIFEST}' \
                    --config '${CONFIG}' \
                    --output-dir '${RUN_DIR}' \
                    --device cuda \
                    --max-optimizer-steps '${STEPS}'
        " \
        > "${REPO}/slurm_logs/c2f2_seed${SEED}_${SLURM_JOB_ID}.out" \
        2> "${REPO}/slurm_logs/c2f2_seed${SEED}_${SLURM_JOB_ID}.err" &

    PIDS+=($!)

done

# ============================================================
# Wait for all three runs
# ============================================================

FAIL=0

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

echo
echo "============================================================"

if [ "${FAIL}" -ne 0 ]; then
    echo "At least one C2+F2 run failed."
    exit 1
fi

echo "All C2+F2 runs completed successfully."
echo
echo "Results:"
for SEED in "${SEEDS[@]}"; do
    echo "  ${OUTPUT_ROOT}/seed_${SEED}"
done

echo
echo "Finished: $(date)"
echo "============================================================"
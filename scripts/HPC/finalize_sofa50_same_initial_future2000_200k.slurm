#!/bin/bash
#SBATCH --job-name=s50_f2k200k_final
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:59:00
#SBATCH --output=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.out
#SBATCH --error=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python
SOURCE_ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820
ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_future2000_200k_20260822
REPORT=${ROOT}/full_report
RUN=${PROJECT}/runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_hf_200k_seed7_bw7

cd "${PROJECT}"
mkdir -p "${ROOT}/full" "${REPORT}"
for method in nds nvdiffrec exmesh; do
    test -d "${SOURCE_ROOT}/full/${method}/samples"
    if [[ ! -e "${ROOT}/full/${method}" ]]; then
        ln -s "${SOURCE_ROOT}/full/${method}" "${ROOT}/full/${method}"
    fi
done

export ROOT RUN SOURCE_ROOT
"${PYTHON}" -c 'import hashlib,json,os,pathlib; run=pathlib.Path(os.environ["RUN"]); ckpt=run/"checkpoint_latest.pt"; sha=hashlib.sha256(ckpt.read_bytes()).hexdigest(); payload={"experiment":"sofa50_same_initial_future2000_200k_cross_dataset_generalization","ours_checkpoint":str(ckpt),"ours_checkpoint_sha256":sha,"ours_training_dataset":"Future2000 GT-adaptive expanded current, 8000/1000/1000 variants","ours_training_resolution":960,"ours_training_optimizer_steps":200000,"inference_dataset":"same canonical 25 Sofa50 test samples","inference_observations":"same 28 native-1920 RGB images and cameras","inference_view_chunk_size":4,"external_outputs":"frozen archived outputs from the completed 2026-08-20 same-initial run","external_source_root":os.environ["SOURCE_ROOT"],"no_external_reoptimization":True,"comparison_scope":"cross-dataset/resolution generalization diagnostic; same benchmark inputs and unified evaluator"}; out=pathlib.Path(os.environ["ROOT"])/"EXPERIMENT_CONTRACT.json"; out.write_text(json.dumps(payload,indent=2)+"\n")'

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH=${PROJECT}/src
"${PYTHON}" scripts/aggregate_sofa50_same_initial_benchmark.py \
    --manifest "${SOURCE_ROOT}/benchmark_manifest.json" \
    --config "${PROJECT}/configs/baselines/sofa50_same_initial_benchmark.json" \
    --results-root "${ROOT}/full" \
    --output-dir "${REPORT}" \
    --surface-samples 3000 \
    --metric-seed 7 \
    --fscore-threshold 0.01

"${PYTHON}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["contract_audit"] else 2)' "${REPORT}/summary.json"
"${PYTHON}" scripts/render_sofa50_same_initial_benchmark.py \
    --manifest "${SOURCE_ROOT}/benchmark_manifest.json" \
    --results-root "${ROOT}/full" \
    --per-sample-csv "${REPORT}/per_sample.csv" \
    --output-dir "${REPORT}/visualizations" \
    --count 5 \
    --image-size 360 \
    --view 0

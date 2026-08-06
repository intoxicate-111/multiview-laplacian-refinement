#!/usr/bin/env bash
set -euo pipefail

repository_root="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement"
conda_binary="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/bin/conda"
manifest_path="/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json"
config_path="${repository_root}/configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json"
output_path="${repository_root}/runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full"

cd "${repository_root}"
export PYTHONPATH=src
exec "${conda_binary}" run --no-capture-output -n test \
  python scripts/train_multi_mesh_laplacian.py \
  --manifest "${manifest_path}" \
  --config "${config_path}" \
  --output-dir "${output_path}" \
  --device cuda

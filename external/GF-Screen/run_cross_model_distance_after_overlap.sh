#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
overlap_result="analysis/cross_model_component_fusion_overlap_only.json"
output="analysis/cross_model_component_fusion_screen.json"

while [[ ! -s "${overlap_result}" ]]; do
  sleep 15
done

"${python_bin}" -u screen_cross_model_component_fusion.py \
  --gf predictions/gf_a055_bbox_pu_best_ep280/public_raw \
  --support ../../predictions/flare25_finetune_by_epoch/epoch_0050/predictions \
  --labels ../../validation/Validation-Public-Labels \
  --distances-mm 0,2,5,10 --rescue-min-mm3 inf,1000,3000,10000 \
  --output "${output}"

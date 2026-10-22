#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT TAG" >&2
  exit 2
fi

checkpoint="$1"
tag="$2"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
root="local_selection/${tag}"
anchor_metrics="public_sweeps/robust_selector/epoch0105_mm500/metrics.csv"
evaluator="../../handoff_bundle/code/evaluate.py"
min_component_mm3="${MIN_COMPONENT_MM3:-500}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${anchor_metrics}" ]]; then
  echo "missing epoch-105 anchor metrics: ${anchor_metrics}" >&2
  exit 1
fi

mkdir -p "${root}/gpu0" "${root}/gpu1" "${root}/raw"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${images}" --save_prediction_path "${root}/gpu0" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 --focus_batch_size 2 \
  --workers 2 --num_shards 2 --shard_id 0 > "${root}/gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${images}" --save_prediction_path "${root}/gpu1" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 --focus_batch_size 2 \
  --workers 2 --num_shards 2 --shard_id 1 > "${root}/gpu1.log" 2>&1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
cp "${root}/gpu0"/*.nii.gz "${root}/raw/"
cp "${root}/gpu1"/*.nii.gz "${root}/raw/"

candidate="${root}/mm${min_component_mm3}"
mkdir -p "${candidate}"
"${python_bin}" -u fuse_and_filter_masks.py --pred-a "${root}/raw" \
  --fusion a --min-mm3 "${min_component_mm3}" --output "${candidate}/predictions" \
  > "${candidate}/transform.log" 2>&1
"${python_bin}" -u "${evaluator}" --pred "${candidate}/predictions" --gt "${labels}" \
  --csv "${candidate}/metrics.csv" > "${candidate}/score.log" 2>&1
dsc=$(awk '/^DSC  mean/ {print $3}' "${candidate}/score.log")
nsd=$(awk '/^NSD  mean/ {print $3}' "${candidate}/score.log")
echo -e "min_mm3\tDSC\tNSD" > "${root}/summary.tsv"
echo -e "${min_component_mm3}\t${dsc}\t${nsd}" >> "${root}/summary.tsv"

"${python_bin}" select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" \
  --candidate "${candidate}/metrics.csv" \
  --anchor-name epoch105 --candidate-name "${tag}_mm${min_component_mm3}" \
  --output "${root}/robust_decision.json"
echo "Fixed component filter: ${min_component_mm3} mm3"

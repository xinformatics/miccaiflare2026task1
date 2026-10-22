#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="predictions/gf_a055_bbox_pu_best_ep280"
public_raw="${root}/public_raw"
public_filtered="${root}/public_mm1000"
hidden_raw="${root}/hidden_raw"
hidden_filtered="${root}/hidden_mm1000"
submission="../../submissions/gf_a055_bbox_pu_best_ep280_mm1000.zip"
evaluator="../../handoff_bundle/code/evaluate.py"
packager="../../nnunet/package_codabench.py"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "${public_raw}" "${hidden_raw}" "${root}" "../../submissions"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${public_images}" \
  --save_prediction_path "${public_raw}" \
  --trained_pth "${checkpoint}" \
  --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 \
  --focus_batch_size 2 --workers 2 \
  > "${root}/public_inference.log" 2>&1 &
public_pid=$!

CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${hidden_images}" \
  --save_prediction_path "${hidden_raw}" \
  --trained_pth "${checkpoint}" \
  --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 \
  --focus_batch_size 2 --workers 2 \
  > "${root}/hidden_inference.log" 2>&1 &
hidden_pid=$!

wait "${public_pid}"
wait "${hidden_pid}"

public_count=$(find "${public_raw}" -maxdepth 1 -name '*.nii.gz' | wc -l)
hidden_count=$(find "${hidden_raw}" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${public_count}" -eq 50 ]]
[[ "${hidden_count}" -eq 100 ]]

mkdir -p "${public_filtered}" "${hidden_filtered}"
"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${public_raw}" --fusion a --min-mm3 1000 \
  --output "${public_filtered}" > "${root}/public_filter.log" 2>&1 &
public_filter_pid=$!
"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${hidden_raw}" --fusion a --min-mm3 1000 \
  --output "${hidden_filtered}" > "${root}/hidden_filter.log" 2>&1 &
hidden_filter_pid=$!
wait "${public_filter_pid}"
wait "${hidden_filter_pid}"

"${python_bin}" -u "${evaluator}" \
  --pred "${public_filtered}" --gt "${public_labels}" \
  --csv "${root}/public_metrics.csv" | tee "${root}/public_score.log"

"${python_bin}" -u "${packager}" \
  --images "${hidden_images}" --pred-dir "${hidden_filtered}" \
  --output "${submission}" --expected-count 100 \
  | tee "${root}/package.log"

unzip -t "${submission}" > "${root}/zip_test.log"
sha256sum "${submission}" | tee "${submission}.sha256"
unzip -Z1 "${submission}" | awk '
  /\.nii\.gz$/ {n++; if ($0 ~ /\//) nested++}
  END {printf "nifti=%d nested=%d\n", n, nested+0}
'

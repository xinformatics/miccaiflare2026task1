#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CHECKPOINT TAG GPU [MIN_MM3]" >&2
  exit 2
fi

checkpoint="$1"
tag="$2"
gpu="$3"
min_mm3="${4:-1000}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
image_dir="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
output_root="hidden_predictions/${tag}"
raw_dir="${output_root}/raw"
final_dir="${output_root}/filtered_mm${min_mm3}"
submission="../../submissions/${tag}_gate05_mm${min_mm3}.zip"
packager="../../nnunet/package_codabench.py"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 1
fi
mkdir -p "${raw_dir}" "${final_dir}" ../../submissions

echo "[${tag}] predicting 100 hidden cases on GPU ${gpu}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${image_dir}" \
  --save_prediction_path "${raw_dir}" \
  --trained_pth "${checkpoint}" \
  --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 \
  --sw_batch_size 2 --workers 2 \
  > "${output_root}/inference.log" 2>&1

count=$(find "${raw_dir}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
if [[ "${count}" -ne 100 ]]; then
  echo "Expected 100 predictions, found ${count}" >&2
  exit 1
fi

"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${raw_dir}" --fusion a --min-mm3 "${min_mm3}" --output "${final_dir}" \
  > "${output_root}/filter.log" 2>&1

"${python_bin}" -u "${packager}" \
  --images "${image_dir}" --pred-dir "${final_dir}" \
  --output "${submission}" --expected-count 100
echo "[${tag}] ready: ${submission}"

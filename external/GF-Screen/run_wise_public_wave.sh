#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
root="${OUTPUT_ROOT:-public_sweeps/wise_epoch105_to_215}"

run_candidate() {
  local gpu="$1"
  local tag="$2"
  local checkpoint="$3"
  local output="${root}/${tag}/raw"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 > "${root}/${tag}/inference.log" 2>&1
  count=$(find "${output}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "${tag}: expected 50 predictions, found ${count}" >&2
    return 1
  fi
}

run_candidate 0 wise_all_a010 checkpoint_interpolations/epoch105_to_215/wise_all_alpha0p100.pt &
pid0=$!
run_candidate 1 wise_focus_a010 checkpoint_interpolations/epoch105_to_215/wise_focus_alpha0p100.pt &
pid1=$!
wait "${pid0}"
wait "${pid1}"
echo "WiSE alpha-0.1 public inference complete: ${root}"

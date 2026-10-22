#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
root="${OUTPUT_ROOT:-public_sweeps/epoch0105_low_gate}"
lesion_thresholds="${LESION_THRESHOLDS:-0.40,0.45,0.50,0.55,0.60}"

run_gate() {
  local gpu="$1"
  local gate="$2"
  local tag="${gate/./p}"
  local output="${root}/gate_${tag}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${gate}" --lesion_thresholds "${lesion_thresholds}" \
    --focus_batch_size 2 --workers 2 > "${output}/inference.log" 2>&1
  for threshold_dir in "${output}"/lesion_*; do
    count=$(find "${threshold_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l)
    if [[ "${count}" -ne 50 ]]; then
      echo "${threshold_dir}: expected 50 predictions, found ${count}" >&2
      return 1
    fi
  done
}

# The first wave brackets the likely useful range. Refine to 0.1/0.3 only if
# one of these beats the fixed epoch-105 operating point robustly.
run_gate 0 0.2 &
pid0=$!
run_gate 1 0.4 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
echo "Low-gate inference complete: ${root}"

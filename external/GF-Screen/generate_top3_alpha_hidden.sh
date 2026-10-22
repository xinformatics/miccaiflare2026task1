#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
checkpoint_root="checkpoint_interpolations/epoch105_to_275_fine"
root="hidden_optimized/wise_focus_epoch105_to_275_alpha_sweep"
packager="../../nnunet/package_codabench.py"
submission_root="../../submissions"

predict_full() {
  local gpu="$1"
  local tag="$2"
  local raw="${root}/a${tag}/raw"
  mkdir -p "${raw}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${raw}" \
    --trained_pth "${checkpoint_root}/wise_focus_alpha0p${tag}.pt" \
    --a_min -175 --a_max 250 --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 > "${root}/a${tag}/inference.log" 2>&1
  count=$(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq 100 ]]
}

predict_shard() {
  local gpu="$1"
  local tag="$2"
  local shard="$3"
  local output="${root}/a${tag}/gpu${gpu}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint_root}/wise_focus_alpha0p${tag}.pt" \
    --a_min -175 --a_max 250 --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${root}/a${tag}/gpu${gpu}.log" 2>&1
}

mkdir -p "${root}" "${submission_root}"

# First wave uses one complete candidate per GPU.
predict_full 0 250 & pid0=$!
predict_full 1 225 & pid1=$!
wait "${pid0}"
wait "${pid1}"

# Final candidate is split across both GPUs.
predict_shard 0 200 0 & pid0=$!
predict_shard 1 200 1 & pid1=$!
wait "${pid0}"
wait "${pid1}"
mkdir -p "${root}/a200/raw"
cp "${root}/a200/gpu0"/*.nii.gz "${root}/a200/raw/"
cp "${root}/a200/gpu1"/*.nii.gz "${root}/a200/raw/"
count=$(find "${root}/a200/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 100 ]]

filter_candidate() {
  local tag="$1"
  local output="${root}/a${tag}/final_predictions_mm1000"
  mkdir -p "${output}"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${root}/a${tag}/raw" --fusion a --min-mm3 1000 --output "${output}" \
    > "${root}/a${tag}/filter_mm1000.log" 2>&1
}

pids=()
for tag in 200 225 250; do
  filter_candidate "${tag}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

package_candidate() {
  local tag="$1"
  local zip_path="${submission_root}/gf_wise_focus_epoch105_to_275_a0${tag}_mm1000.zip"
  "${python_bin}" -u "${packager}" --images "${images}" \
    --pred-dir "${root}/a${tag}/final_predictions_mm1000" \
    --output "${zip_path}" --expected-count 100 \
    > "${root}/a${tag}/package.log" 2>&1
  unzip -t "${zip_path}" > "${root}/a${tag}/zip_test.log"
  sha256sum "${zip_path}" > "${zip_path}.sha256"
}

pids=()
for tag in 200 225 250; do
  package_candidate "${tag}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

for tag in 200 225 250; do
  zip_path="${submission_root}/gf_wise_focus_epoch105_to_275_a0${tag}_mm1000.zip"
  ls -lh "${zip_path}"
  cat "${zip_path}.sha256"
done

#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
run_dir="${RUN_DIR:-runs/gf_glance_only_ep280_hn300_lr3e5_5ep_20260820}"
root="${OUTPUT_ROOT:-public_sweeps/gf_glance_only_ep280_hn300_lr3e5_5ep_20260820}"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
evaluator="evaluate_public_parallel.py"
packager="../../nnunet/package_codabench.py"
epochs=(281 282 283 284 285)

mkdir -p "${root}" ../../submissions

evaluate_checkpoint() {
  local gpu="$1"
  local epoch="$2"
  local checkpoint
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  local candidate
  candidate=$(printf '%s/epoch_%04d' "${root}" "${epoch}")
  local raw="${candidate}/raw"
  local filtered="${candidate}/mm1000"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    return 1
  fi
  mkdir -p "${raw}" "${filtered}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${raw}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 > "${candidate}/inference.log" 2>&1
  count=$(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "Epoch ${epoch}: expected 50 public predictions, found ${count}" >&2
    return 1
  fi
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${raw}" --fusion a --min-mm3 1000 --output "${filtered}" \
    > "${candidate}/filter.log" 2>&1
  "${python_bin}" -u "${evaluator}" \
    --pred "${filtered}" --gt "${public_labels}" \
    --csv "${candidate}/metrics.csv" --workers 4 > "${candidate}/score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${candidate}/score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${candidate}/score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\n' "${epoch}" "${dsc}" "${nsd}" "${combined}" \
    > "${candidate}/result.tsv"
}

# Two independent checkpoints are evaluated concurrently, one per GPU.
evaluate_checkpoint 0 281 &
pid0=$!
evaluate_checkpoint 1 282 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

evaluate_checkpoint 0 283 &
pid0=$!
evaluate_checkpoint 1 284 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

evaluate_checkpoint 0 285

printf 'epoch\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
for epoch in "${epochs[@]}"; do
  candidate=$(printf '%s/epoch_%04d' "${root}" "${epoch}")
  cat "${candidate}/result.tsv" >> "${root}/summary.tsv"
done
{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k4,4gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_epoch best_dsc best_nsd best_combined <<< "${best_line}"
best_checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${best_epoch}")
hidden_root=$(printf '%s/hidden_epoch_%04d' "${root}" "${best_epoch}")
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" "${hidden_root}/mm1000"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard0" \
  --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 \
  --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
  > "${hidden_root}/shard0.log" 2>&1 &
hidden_pid0=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard1" \
  --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_threshold 0.5 \
  --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 1 \
  > "${hidden_root}/shard1.log" 2>&1 &
hidden_pid1=$!
wait "${hidden_pid0}"
wait "${hidden_pid1}"

cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
hidden_count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${hidden_count}" -ne 100 ]]; then
  echo "Expected 100 hidden predictions, found ${hidden_count}" >&2
  exit 1
fi
"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${hidden_root}/raw" --fusion a --min-mm3 1000 \
  --output "${hidden_root}/mm1000" > "${hidden_root}/filter.log" 2>&1

submission=$(printf '../../submissions/gf_glance_only_ep%04d_g05_l05_mm1000.zip' "${best_epoch}")
"${python_bin}" -u "${packager}" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/mm1000" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" | tee "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_checkpoint.tsv"
echo "Submission ready: ${submission}"

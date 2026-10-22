#!/usr/bin/env bash
set -euo pipefail

# Measure the Focus-only ceiling by bypassing the Glance proposal gate.  This
# is deliberately queued after all current training/audits so it cannot steal
# GPU memory.  It is a bounded diagnostic, not a new default inference mode.
wait_pid="${WAIT_PID:-829739}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/full_coverage_focus_20260820}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
thresholds="0.55,0.60,0.65,0.70,0.75"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
mkdir -p "${root}/shard0" "${root}/shard1" "${root}/merged" ../../submissions

# Score completed consistency/component-box runs before spending more compute.
# This stage is part of this watcher, so downstream serialization is preserved.
bash ./audit_completed_bbox_variants_before_fullcoverage.sh \
  > "${root}/completed_bbox_audit.log" 2>&1

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.0 --lesion_thresholds "${thresholds}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "${pid0}"
wait "${pid1}"

printf 'glance_threshold\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
IFS=',' read -ra threshold_array <<< "${thresholds}"
for threshold in "${threshold_array[@]}"; do
  threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  merged="${root}/merged/${threshold_dir}"
  mkdir -p "${merged}"
  cp "${root}/shard0/${threshold_dir}"/*.nii.gz "${merged}/"
  cp "${root}/shard1/${threshold_dir}"/*.nii.gz "${merged}/"
  count=$(find "${merged}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "${threshold_dir}: expected 50 public predictions, found ${count}" >&2
    exit 1
  fi
  for min_mm3 in 0 500 1000 1500 3000; do
    score_stem="${root}/scored/${threshold_dir}_mm${min_mm3}"
    candidate="${merged}"
    mkdir -p "$(dirname "${score_stem}")"
    if [[ "${min_mm3}" -gt 0 ]]; then
      candidate="${score_stem}"
      mkdir -p "${candidate}"
      "${python_bin}" -u fuse_and_filter_masks.py \
        --pred-a "${merged}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${candidate}" > "${score_stem}.filter.log" 2>&1
    fi
    "${python_bin}" -u evaluate_public_parallel.py \
      --pred "${candidate}" --gt "${public_labels}" \
      --csv "${score_stem}.metrics.csv" --workers 4 > "${score_stem}.score.log" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
    printf '0.0\t%s\t%s\t%s\t%s\t%s\n' \
      "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
  done
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_gate best_focus best_min best_dsc best_nsd best_combined <<< "${best_line}"
focus_dir="lesion_$(printf '%.2f' "${best_focus}")"
selected_metrics="${root}/scored/${focus_dir}_mm${best_min}.metrics.csv"
candidate_tag="fullcoverage_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_g05_l05_mm1000 --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Full-coverage candidate did not clear the paired robustness gate."
  exit 0
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" "${hidden_root}/filtered"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.0 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
wait "${hidden_pid0}"
wait "${hidden_pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${count}" -ne 100 ]]; then
  echo "Expected 100 hidden predictions, found ${count}" >&2
  exit 1
fi
"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min}" \
  --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
submission="../../submissions/gf_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/filtered" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Full-coverage submission ready: ${submission}"

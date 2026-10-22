#!/usr/bin/env bash
set -euo pipefail

# Validation-only audit on known no-lesion CTs. These scans must never enter a
# training JSON. Run at the tail of the current GPU queue to establish the
# immutable champion's clean-case and false-positive-volume baseline.
wait_pid="${WAIT_PID:-878244}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
images="../../validation/HealthyImages-noLesion"
root="${OUTPUT_ROOT:-public_sweeps/gf_champion_healthy_audit_20260820}"
thresholds="0.50,0.60"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
if [[ $(find "${images}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 172 ]]; then
  echo "Expected exactly 172 held-out healthy scans" >&2
  exit 1
fi
mkdir -p "${root}/shard0" "${root}/shard1" "${root}/scored"

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds "${thresholds}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "${pid0}"
wait "${pid1}"

printf 'lesion_threshold\tmin_mm3\tempty_predictions\tfalse_positive_cases\thealthy_DSC\thealthy_NSD\tmean_false_positive_mm3\tmedian_false_positive_mm3\n' \
  > "${root}/summary.tsv"
for threshold in 0.50 0.60; do
  threshold_dir="lesion_${threshold}"
  combined="${root}/combined_${threshold_dir}"
  mkdir -p "${combined}"
  cp "${root}/shard0/${threshold_dir}"/*.nii.gz "${combined}/"
  cp "${root}/shard1/${threshold_dir}"/*.nii.gz "${combined}/"
  if [[ $(find "${combined}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 172 ]]; then
    echo "Incomplete healthy predictions for threshold ${threshold}" >&2
    exit 1
  fi
  for min_mm3 in 0 500 1000 3000; do
    candidate="${combined}"
    tag="l${threshold}_mm${min_mm3}"
    if [[ "${min_mm3}" -gt 0 ]]; then
      candidate="${root}/scored/${tag}"
      mkdir -p "${candidate}"
      "${python_bin}" -u fuse_and_filter_masks.py \
        --pred-a "${combined}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${candidate}" > "${root}/scored/${tag}.filter.log" 2>&1
    fi
    metrics="${root}/scored/${tag}.metrics.csv"
    "${python_bin}" -u evaluate_healthy_predictions.py \
      --pred "${candidate}" --output "${metrics}" --expected-count 172 \
      > "${root}/scored/${tag}.score.log"
    summary="${root}/scored/${tag}.metrics.summary.json"
    values=$("${python_bin}" -c \
      "import json,sys; d=json.load(open(sys.argv[1])); print(*(d[k] for k in ('empty_predictions','false_positive_cases','healthy_DSC','healthy_NSD','mean_false_positive_mm3','median_false_positive_mm3')),sep='\\t')" \
      "${summary}")
    printf '%s\t%s\t%s\n' "${threshold}" "${min_mm3}" "${values}" >> "${root}/summary.tsv"
  done
done
{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k5,5gr -k7,7g
} > "${root}/summary_by_clean_rate.tsv"
cat "${root}/summary_by_clean_rate.tsv"

#!/usr/bin/env bash
set -euo pipefail

# Public-label-informed diagnostic only. Ground truth selects Focus crops, which
# decomposes the Glance localization gap from Focus segmentation quality. These
# outputs are forbidden for hidden inference, submission, or model training.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt}"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
root="${OUTPUT_ROOT:-public_sweeps/focus_oracle_public_20260820}"
thresholds="${LESION_THRESHOLDS:-0.30,0.40,0.50,0.60,0.70}"

[[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
mkdir -p "${root}/shard0" "${root}/shard1" "${root}/merged" "${root}/scored"

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --proposal_mask_path "${public_labels}" --proposal_only --proposal_min_voxels 1 \
    --glance_threshold 0.50 --lesion_thresholds "${thresholds}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing \
    > "${root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "${pid0}"
wait "${pid1}"

printf 'lesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
IFS=',' read -ra threshold_array <<< "${thresholds}"
for threshold in "${threshold_array[@]}"; do
  threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  merged="${root}/merged/${threshold_dir}"
  mkdir -p "${merged}"
  cp "${root}/shard0/${threshold_dir}"/*.nii.gz "${merged}/"
  cp "${root}/shard1/${threshold_dir}"/*.nii.gz "${merged}/"
  count=$(find "${merged}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq 50 ]] || {
    echo "${threshold_dir}: expected 50 predictions, found ${count}" >&2
    exit 1
  }
  for min_mm3 in 0 500 1000; do
    score_stem="${root}/scored/${threshold_dir}_mm${min_mm3}"
    candidate="${merged}"
    if [[ "${min_mm3}" -gt 0 ]]; then
      candidate="${score_stem}"
      mkdir -p "${candidate}"
      "${python_bin}" -u fuse_and_filter_masks.py \
        --pred-a "${merged}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${candidate}" > "${score_stem}.filter.log" 2>&1
    fi
    "${python_bin}" -u evaluate_public_parallel.py \
      --pred "${candidate}" --gt "${public_labels}" \
      --csv "${score_stem}.metrics.csv" --workers 4 \
      > "${score_stem}.score.log" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
      >> "${root}/summary.tsv"
  done
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k5,5gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
printf '%s\n' \
  'DIAGNOSTIC ONLY: public ground truth selected crops; never submit these outputs.' \
  > "${root}/PUBLIC_LABEL_ORACLE_DO_NOT_SUBMIT.txt"

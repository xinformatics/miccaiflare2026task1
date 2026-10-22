#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
screen="analysis/cross_model_component_fusion_screen.json"
root="public_sweeps/cross_model_component_fusion_distance"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"

while [[ ! -s "${screen}" ]]; do
  sleep 15
done
mkdir -p "${root}"
"${python_bin}" select_cross_model_configs.py \
  --input "${screen}" --output "${root}/selected_screen_configs.tsv" \
  --top 3 --max-zero 7 --require-finite-rescue \
  > "${root}/selection.log"

printf 'distance_mm\trescue_min_mm3\tscreen_DSC\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
tail -n +2 "${root}/selected_screen_configs.tsv" | \
while IFS=$'\t' read -r distance rescue screen_dsc win_rate zeros; do
  distance_tag="${distance/./p}"
  rescue_tag="${rescue/./p}"
  candidate="${root}/d${distance_tag}_r${rescue_tag}"
  count=0
  [[ -d "${candidate}" ]] && count=$(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    mkdir -p "${candidate}"
    "${python_bin}" -u materialize_cross_model_component_fusion.py \
      --gf predictions/gf_a055_bbox_pu_best_ep280/public_raw \
      --support ../../predictions/flare25_finetune_by_epoch/epoch_0050/predictions \
      --distance-mm "${distance}" --rescue-min-mm3 "${rescue}" \
      --output "${candidate}" > "${candidate}.materialize.log" 2>&1
  fi
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${candidate}" --gt ../../validation/Validation-Public-Labels \
    --csv "${candidate}.metrics.csv" --workers 4 > "${candidate}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${candidate}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${candidate}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${distance}" "${rescue}" "${screen_dsc}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r distance rescue screen_dsc dsc nsd combined <<< "${best_line}"
distance_tag="${distance/./p}"
rescue_tag="${rescue/./p}"
candidate="${root}/d${distance_tag}_r${rescue_tag}"
"${python_bin}" select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${candidate}.metrics.csv" \
  --anchor-name gf_champion --candidate-name "cross_d${distance_tag}_r${rescue_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
cat "${root}/robust_selection.json"

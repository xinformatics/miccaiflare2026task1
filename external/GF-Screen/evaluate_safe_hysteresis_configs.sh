#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
screen="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/hysteresis_screen.json"
mask_root="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/merged"
root="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/hysteresis_official"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"

while [[ ! -s "${screen}" ]]; do
  sleep 15
done
mkdir -p "${root}"
"${python_bin}" select_hysteresis_configs.py \
  --input "${screen}" --output "${root}/selected_screen_configs.tsv" \
  --top 3 --max-zero 4 > "${root}/selection.log"

printf 'low\thigh\tmin_mm3\tscreen_DSC\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
tail -n +2 "${root}/selected_screen_configs.tsv" | \
while IFS=$'\t' read -r low high minimum screen_dsc zeros; do
  low_tag=$(printf '%.2f' "${low}")
  high_tag=$(printf '%.2f' "${high}")
  name="low${low_tag/./p}_high${high_tag/./p}_mm${minimum/./p}"
  candidate="${root}/${name}"
  count=0
  [[ -d "${candidate}" ]] && count=$(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    mkdir -p "${candidate}"
    "${python_bin}" -u hysteresis_filter_masks.py \
      --low "${mask_root}/lesion_${low_tag}" \
      --high "${mask_root}/lesion_${high_tag}" \
      --min-mm3 "${minimum}" --output "${candidate}" \
      > "${candidate}.materialize.log" 2>&1
  fi
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${candidate}" --gt ../../validation/Validation-Public-Labels \
    --csv "${candidate}.metrics.csv" --workers 4 > "${candidate}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${candidate}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${candidate}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${low}" "${high}" "${minimum}" "${screen_dsc}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k7,7gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r low high minimum screen_dsc dsc nsd combined <<< "${best_line}"
low_tag=$(printf '%.2f' "${low}")
high_tag=$(printf '%.2f' "${high}")
name="low${low_tag/./p}_high${high_tag/./p}_mm${minimum/./p}"
"${python_bin}" select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${root}/${name}.metrics.csv" \
  --anchor-name gf_champion --candidate-name "hysteresis_${name}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
cat "${root}/robust_selection.json"

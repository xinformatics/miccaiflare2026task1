#!/usr/bin/env bash
set -euo pipefail

# Resume the independent early checkpoint-283 score after an interactive
# launcher exits. This never writes into the active GPU inference directories.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}"
python_bin=/home/m348911/miniconda3/envs/radoncAI/bin/python
root=public_sweeps/gf_bbox_pseudo_positive_dual_20260820/early_checkpoint_0283
anchor=public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv
low=gf_bbox_ppw005_t098_ep280_lr1e5_5ep_20260820
high=gf_bbox_ppw015_t098_ep280_lr1e5_5ep_20260820

score_filtered_060() {
  local arm=$1
  local stem="${root}/${arm}_lesion_0.60_mm1000"
  if [[ -s "${stem}.metrics.csv" ]] && grep -q '^wrote ' "${stem}.score.log"; then
    return
  fi
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${stem}" --gt ../../validation/Validation-Public-Labels \
    --csv "${stem}.metrics.csv" --workers 4 > "${stem}.score.log" 2>&1
}

score_filtered_060 "${low}" & low_pid=$!
score_filtered_060 "${high}" & high_pid=$!
status=0
wait "${low_pid}" || status=$?
wait "${high_pid}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

printf 'variant\tthreshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
for arm in "${low}" "${high}"; do
  for threshold in 0.50 0.60; do
    for minimum in 0 1000; do
      stem="${root}/${arm}_lesion_${threshold}_mm${minimum}"
      dsc=$(awk '/DSC  mean/ {print $3}' "${stem}.score.log")
      nsd=$(awk '/NSD  mean/ {print $3}' "${stem}.score.log")
      [[ -n "${dsc}" && -n "${nsd}" ]] || {
        echo "Incomplete score: ${stem}.score.log" >&2
        exit 1
      }
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${threshold}" "${minimum}" "${dsc}" "${nsd}" "${combined}" \
        >> "${root}/summary.tsv"
    done
  done
done
{
  head -n 1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"

best=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r arm threshold minimum _dsc _nsd _combined <<< "${best}"
metrics="${root}/${arm}_lesion_${threshold}_mm${minimum}.metrics.csv"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${metrics}" \
  --anchor-name champion_bbox \
  --candidate-name "${arm}_cp283_l${threshold}_mm${minimum}" \
  --output "${root}/robust_selection_best.json" \
  > "${root}/robust_selection_best.log"
cat "${root}/summary_by_combined.tsv"

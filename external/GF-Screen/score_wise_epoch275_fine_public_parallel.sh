#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
evaluator="../../handoff_bundle/code/evaluate.py"
root="public_sweeps/wise_epoch105_to_275_fine"

score_candidate() {
  local tag="$1"
  local candidate_root="${root}/wise_focus_a${tag}"
  local filtered="${candidate_root}/mm1000"
  mkdir -p "${filtered}"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${candidate_root}/raw" --fusion a --min-mm3 1000 \
    --output "${filtered}" > "${candidate_root}/filter_mm1000.log" 2>&1
  "${python_bin}" -u "${evaluator}" --pred "${filtered}" --gt "${labels}" \
    --csv "${candidate_root}/metrics_mm1000.csv" > "${candidate_root}/score_mm1000.log" 2>&1
}

pids=()
for tag in 125 175 200 225 250; do
  score_candidate "${tag}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

printf 'alpha\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
for tag in 125 175 200 225 250; do
  score_log="${root}/wise_focus_a${tag}/score_mm1000.log"
  dsc=$(awk '/^DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/^NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '0.%s\t%s\t%s\t%s\n' "${tag}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done
{ head -n 1 "${root}/summary.tsv"; tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k4,4nr; } \
  > "${root}/ranked.tsv"
cat "${root}/ranked.tsv"

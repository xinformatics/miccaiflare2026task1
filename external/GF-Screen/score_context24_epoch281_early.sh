#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
parent_pid="${CONTEXT_PARENT_PID:-828563}"
root="public_sweeps/gf_context24_dual_20260820"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
output="${root}/early_epoch281"
workers="${EVAL_WORKERS:-8}"
variants=(
  gf_bbox_ep280_context24_a03b07_lr1e5_5ep_20260820
  gf_bbox_ep280_context24_a05b05_lr1e5_5ep_20260820
)

complete() {
  local variant source count
  for variant in "${variants[@]}"; do
    source="${root}/${variant}/checkpoint_epoch_0281/lesion_0.50"
    count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)
    [[ "${count}" -eq 50 ]] || return 1
  done
}

for _ in $(seq 1 360); do
  complete && break
  if ! kill -0 "${parent_pid}" 2>/dev/null; then
    echo "Context parent exited before both epoch-281 outputs reached 50 cases" >&2
    exit 1
  fi
  sleep 20
done
complete || { echo "Timed out waiting for epoch-281 outputs" >&2; exit 1; }

mkdir -p "${output}/scored"
printf 'variant\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${output}/summary.tsv"
for variant in "${variants[@]}"; do
  # Early signal only: score the immutable deployment operating point. The
  # parent sweep already evaluates the complete threshold/filter grid.
  threshold=0.50
  minimum=1000
  source="${root}/${variant}/checkpoint_epoch_0281/lesion_${threshold}"
  stem="${output}/scored/${variant}_lesion_${threshold}_mm${minimum}"
  candidate="${stem}"
  mkdir -p "${candidate}"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${source}" --fusion a --min-mm3 "${minimum}" \
    --output "${candidate}" > "${stem}.filter.log" 2>&1
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${candidate}" --gt "${labels}" \
    --csv "${stem}.metrics.csv" --workers "${workers}" > "${stem}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${stem}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${stem}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${threshold}" "${minimum}" "${dsc}" "${nsd}" "${combined}" \
    >> "${output}/summary.tsv"
done

{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${output}/summary_by_combined.tsv"
cat "${output}/summary_by_combined.tsv"

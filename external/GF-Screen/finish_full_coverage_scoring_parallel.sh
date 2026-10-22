#!/usr/bin/env bash
set -euo pipefail

# Resume after the expensive two-GPU all-crop inference. Each Focus threshold
# is independent, so score five threshold families concurrently while keeping
# the five component filters within each family serialized. This uses CPUs and
# releases GPUs for controlled training.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
root="${OUTPUT_ROOT:-public_sweeps/full_coverage_focus_20260820}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
oracle_summary="public_sweeps/focus_oracle_public_20260820/summary_by_combined.tsv"
thresholds=(0.55 0.60 0.65 0.70 0.75)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

mkdir -p "${root}/scored" "${root}/parallel_rows"
for threshold in "${thresholds[@]}"; do
  source="${root}/merged/lesion_$(printf '%.2f' "${threshold}")"
  mkdir -p "${source}"
  cp "${root}/shard0/lesion_$(printf '%.2f' "${threshold}")"/*.nii.gz "${source}/"
  cp "${root}/shard1/lesion_$(printf '%.2f' "${threshold}")"/*.nii.gz "${source}/"
  count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq 50 ]] || {
    echo "${source}: expected 50 completed masks, found ${count}" >&2
    exit 1
  }
done

score_threshold() {
  local threshold="$1"
  local threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  local source="${root}/merged/${threshold_dir}"
  local rows="${root}/parallel_rows/${threshold_dir}.tsv"
  : > "${rows}"
  for min_mm3 in 0 500 1000 1500 3000; do
    local score_stem="${root}/scored/${threshold_dir}_mm${min_mm3}"
    local candidate="${source}"
    if [[ "${min_mm3}" -gt 0 ]]; then
      candidate="${score_stem}"
      mkdir -p "${candidate}"
      "${python_bin}" -u fuse_and_filter_masks.py --pred-a "${source}" \
        --fusion a --min-mm3 "${min_mm3}" --output "${candidate}" --workers 4 \
        > "${score_stem}.filter.log" 2>&1
    fi
    "${python_bin}" -u evaluate_public_parallel.py --pred "${candidate}" \
      --gt "${public_labels}" --csv "${score_stem}.metrics.csv" --workers 4 \
      > "${score_stem}.score.log" 2>&1
    local dsc nsd combined
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
    printf '0.0\t%s\t%s\t%s\t%s\t%s\n' \
      "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" >> "${rows}"
  done
}

pids=()
for threshold in "${thresholds[@]}"; do
  score_threshold "${threshold}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

printf 'glance_threshold\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
for threshold in "${thresholds[@]}"; do
  cat "${root}/parallel_rows/lesion_$(printf '%.2f' "${threshold}").tsv" \
    >> "${root}/summary.tsv"
done
{ head -1 "${root}/summary.tsv"; tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr; } \
  > "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_gate best_focus best_min best_dsc best_nsd best_combined <<< "${best_line}"
selected_metrics="${root}/scored/lesion_$(printf '%.2f' "${best_focus}")_mm${best_min}.metrics.csv"
candidate_tag="fullcoverage_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py --anchor "${anchor_metrics}" \
  --candidate "${selected_metrics}" --anchor-name champion_g05_l05_mm1000 \
  --candidate-name "${candidate_tag}" --output "${root}/robust_selection.json" \
  > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" == "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/promotion_pending.tsv"
  rm -f "${root}/not_promoted.tsv"
else
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  rm -f "${root}/promotion_pending.tsv"
fi

"${python_bin}" -u analyze_focus_bottleneck.py --oracle "${oracle_summary}" \
  --full-coverage "${root}/summary_by_combined.tsv" --anchor "${anchor_metrics}" \
  --output analysis/focus_bottleneck_20260820.json \
  > "${root}/focus_bottleneck.log" 2>&1
date --iso-8601=seconds > "${root}/PARALLEL_SCORING_COMPLETE"
cat "${root}/summary_by_combined.tsv"
cat "${root}/robust_selection.json"

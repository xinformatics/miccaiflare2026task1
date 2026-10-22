#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
source_root="public_sweeps/gf_component_box_uniform_dual_20260820"
checkpoint_tag="${CHECKPOINT_TAG:-0281}"
[[ "${checkpoint_tag}" =~ ^[0-9]{4}$ ]] || {
  echo "CHECKPOINT_TAG must contain exactly four digits" >&2
  exit 2
}
output="${source_root}/early_checkpoint_${checkpoint_tag}"
labels="../../validation/Validation-Public-Labels"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
arms=(
  gf_component_box_uniform_p000_ep280_lr3e5_5ep_20260820
  gf_component_box_uniform_p050_ep280_lr3e5_5ep_20260820
)

mkdir -p "${output}/scored"
grid_args=()
printf 'variant\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${output}/summary.tsv"
for arm in "${arms[@]}"; do
  for threshold in 0.50 0.60; do
    source="${source_root}/${arm}/checkpoint_epoch_${checkpoint_tag}/lesion_${threshold}"
    while [[ $(find "${source}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -lt 50 ]]; do
      sleep 15
    done
    tag="${arm}_checkpoint_epoch_${checkpoint_tag}_lesion_${threshold}_mm1000"
    filtered="${output}/scored/${tag}"
    mkdir -p "${filtered}"
    if [[ $(find "${filtered}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
      "${python_bin}" -u fuse_and_filter_masks.py --pred-a "${source}" \
        --fusion a --min-mm3 1000 --workers 4 --output "${filtered}" \
        > "${filtered}.filter.log" 2>&1
    fi
    grid_args+=(--candidate "${tag}=${filtered}")
  done
done

"${python_bin}" -u evaluate_public_grid.py --gt "${labels}" \
  --output-dir "${output}/scored" --workers 8 "${grid_args[@]}" \
  > "${output}/grid.score.log" 2>&1

for arm in "${arms[@]}"; do
  for threshold in 0.50 0.60; do
    tag="${arm}_checkpoint_epoch_${checkpoint_tag}_lesion_${threshold}_mm1000"
    metrics="${output}/scored/${tag}.metrics.csv"
    dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
    nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
    printf '%s\t%s\t1000\t%s\t%s\t%s\n' \
      "${arm}" "${threshold}" "${dsc}" "${nsd}" "${combined}" >> "${output}/summary.tsv"
  done
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr; } \
  > "${output}/summary_by_combined.tsv"
best=$(sed -n '2p' "${output}/summary_by_combined.tsv")
IFS=$'\t' read -r arm threshold _ _ _ _ <<< "${best}"
tag="${arm}_checkpoint_epoch_${checkpoint_tag}_lesion_${threshold}_mm1000"
"${python_bin}" -u select_checkpoint_robust.py --anchor "${anchor}" \
  --candidate "${output}/scored/${tag}.metrics.csv" --anchor-name champion_bbox \
  --candidate-name "${tag}" --output "${output}/robust_selection.json" \
  > "${output}/robust_selection.log"
cat "${output}/summary_by_combined.tsv"

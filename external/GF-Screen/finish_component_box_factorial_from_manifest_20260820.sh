#!/usr/bin/env bash
set -euo pipefail

# Recovery entrypoint after the long-lived evaluator read a script while that
# file was being patched and hit a transient parse error. All six checkpoint
# prediction sets and all 36 manifest rows are complete; resume only exact
# scoring/selection, never inference or training.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
root="public_sweeps/gf_component_box_uniform_dual_20260820"
manifest="${root}/candidate_manifest.tsv"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"

[[ $(wc -l < "${manifest}") -eq 37 ]] || {
  echo "expected header plus 36 manifest rows" >&2
  exit 1
}
grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ score_tag predictions; do
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "${score_tag}: expected 50 masks" >&2
    exit 1
  }
  grid_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${manifest}")

"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 12 "${grid_args[@]}" \
  > "${root}/candidate_grid.score.log" 2>&1

printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
while IFS=$'\t' read -r variant tag checkpoint threshold min_mm3 score_tag _; do
  metrics="${root}/scored/${score_tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${tag}" "${checkpoint}" "${threshold}" "${min_mm3}" \
    "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")
{ head -1 "${root}/summary.tsv"; tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k8,8gr; } \
  > "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r variant tag checkpoint threshold min_mm3 _ _ _ <<< "${best_line}"
score_tag="${variant}_${tag}_lesion_$(printf '%.2f' "${threshold}")_mm${min_mm3}"
predictions="${root}/${variant}/${tag}/lesion_$(printf '%.2f' "${threshold}")"
[[ "${min_mm3}" -gt 0 ]] && predictions="${root}/scored/${score_tag}"
"${python_bin}" -u analyze_lesion_errors.py --labels "${public_labels}" \
  --predictions "${predictions}" --output "${root}/component_audit_best" \
  > "${root}/component_audit_best.log" 2>&1
"${python_bin}" -u select_checkpoint_robust.py --anchor "${anchor_metrics}" \
  --candidate "${root}/scored/${score_tag}.metrics.csv" \
  --anchor-name champion_bbox --candidate-name "${score_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" == "PROMOTE" ]]; then
  echo "Unexpected promotion; hidden materialization requires the standard wrapper." >&2
  exit 3
fi
printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
cat "${root}/summary_by_combined.tsv"
echo "Recovered factorial scoring complete: ${decision}"

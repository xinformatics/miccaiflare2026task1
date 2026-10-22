#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
root="${workspace}/external/BiomedParse"
gf="${workspace}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
combined="${root}/public_sweeps/combined_prompt_confirmation_20260820"
output="${root}/public_sweeps/champion_topk_rescue_20260820"
base="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_mm1000"
labels="${workspace}/validation/Validation-Public-Labels"
anchor="${gf}/public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
wait_pid="${WAIT_PID:-1111260}"

[[ "${wait_pid}" =~ ^[0-9]+$ ]] || { echo "WAIT_PID must be numeric" >&2; exit 1; }
while kill -0 "${wait_pid}" 2>/dev/null; do sleep 10; done
mkdir -p "${output}/scored"
screen="${output}/screen"
if [[ -s "${screen}/screen.json" && ! -s "${screen}/shortlist.tsv" ]]; then
  "${python_bin}" -u "${root}/biomedparse_fusion.py" resume-screen \
    --screen-json "${screen}/screen.json" --output "${screen}" --workers 3 \
    > "${output}/screen.resume.log" 2>&1
elif [[ ! -s "${screen}/screen.json" ]]; then
  "${python_bin}" -u "${root}/biomedparse_fusion.py" screen \
    --base "${base}" \
    --support "prompt_or_e050=${combined}/supports/union_e050" \
    --kind confirm --include-top-k-rescue \
    --labels "${labels}" --output "${screen}" --workers 2 \
    > "${output}/screen.log" 2>&1
fi

candidate_args=()
while IFS=$'\t\r' read -r _ _ _ _ _ _ tag _ _ _ _ _ _ _ predictions; do
  candidate_args+=(--candidate "${tag}=${predictions}")
done < <(tail -n +2 "${screen}/shortlist.tsv")
"${python_bin}" -u "${gf}/evaluate_public_grid.py" --gt "${labels}" \
  --output-dir "${output}/scored" --workers 8 "${candidate_args[@]}" \
  > "${output}/exact_grid.score.log" 2>&1

printf 'tag\tscreen_DSC\tDSC\tNSD\tcombined\tmetrics\tpredictions\tconfig\n' \
  > "${output}/summary.tsv"
while IFS=$'\t\r' read -r _ _ _ _ _ _ tag screen_dsc _ _ _ _ _ config predictions; do
  metrics="${output}/scored/${tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.4f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.4f", sum/n}' "${metrics}")
  combined_score=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${screen_dsc}" "${dsc}" "${nsd}" "${combined_score}" \
    "${metrics}" "${predictions}" "${config}" >> "${output}/summary.tsv"
done < <(tail -n +2 "${screen}/shortlist.tsv")
{ head -1 "${output}/summary.tsv"; \
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr; } \
  > "${output}/summary_by_combined.tsv"

best=$(sed -n '2p' "${output}/summary_by_combined.tsv")
IFS=$'\t' read -r tag _ _ _ _ metrics _ _ <<< "${best}"
"${python_bin}" "${gf}/select_checkpoint_robust.py" \
  --anchor "${anchor}" --candidate "${metrics}" \
  --anchor-name gf_bbox_champion --candidate-name "${tag}" \
  --output "${output}/robust_selection.json" > "${output}/robust_selection.log"
cat "${output}/summary_by_combined.tsv"
cat "${output}/robust_selection.json"

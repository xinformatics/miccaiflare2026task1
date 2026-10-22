#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
fusion="${root}/public_sweeps/automatic_text_proposals_20260820/direct_fusion"
tag="whole_body_lesion_e050_confirm_distance_mm_0_rescue_min_mm3_3000"
predictions="${fusion}/predictions/${tag}"
metrics="${fusion}/${tag}.metrics.csv"
summary="${fusion}/rescue3000_exact_summary.tsv"
wait_score_pid="${WAIT_SCORE_PID:-1106469}"
wait_materialize_pid="${WAIT_MATERIALIZE_PID:-1108327}"

for pid in "${wait_score_pid}" "${wait_materialize_pid}"; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || { echo "invalid wait PID: ${pid}" >&2; exit 1; }
  while kill -0 "${pid}" 2>/dev/null; do sleep 10; done
done

count=$(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 50 ]] || {
  echo "${tag}: expected 50 predictions, found ${count}" >&2
  exit 1
}

"${python_bin}" -u "${gf}/evaluate_public_grid.py" \
  --gt "${project}/validation/Validation-Public-Labels" \
  --candidate "${tag}=${predictions}" --output-dir "${fusion}" --workers 8 \
  > "${fusion}/rescue3000_exact.score.log" 2>&1

dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.4f", sum/n}' "${metrics}")
nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.4f", sum/n}' "${metrics}")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
printf 'tag\tDSC\tNSD\tcombined\tmetrics\tpredictions\n' > "${summary}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${predictions}" \
  >> "${summary}"
cat "${summary}"

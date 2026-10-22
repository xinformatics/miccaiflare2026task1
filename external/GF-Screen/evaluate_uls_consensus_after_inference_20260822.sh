#!/usr/bin/env bash
set -euo pipefail

# Public-only, label-free ULS+ agreement sweep. It waits for the shared official
# ULS+ inference and cannot infer hidden cases, create a ZIP, or upload.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
uls_output="${root}/public_sweeps/uls_plus_rich20_refiner_20260822"
work="${uls_output}/work"
upstream_marker="${work}/INFERENCE_COMPLETE.txt"
labels="${project}/validation/Validation-Public-Labels"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/uls_plus_consensus_20260822"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "ULS+ consensus audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${upstream_marker}" ]]; do
  printf '%s waiting for full ULS+ public inference\n' "$(date -Ins)"
  sleep 30
done

printf 'variant\tmutual\tmin_pair_mm3\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${output}/summary.tsv"
for mutual in 0.70 0.80 0.90; do
  for min_mm3 in 500 1000; do
    mutual_tag=${mutual/./p}
    tag="m${mutual_tag}_mm${min_mm3}_top1"
    candidate="${output}/${tag}"
    "${python_bin}" -u "${root}/materialize_uls_consensus.py" \
      --work "${work}" --output "${candidate}" --base "${base}" \
      --min-mutual-dice "${mutual}" --min-pair-mm3 "${min_mm3}" \
      --top-pairs-per-case 1 > "${candidate}.materialize.log" 2>&1
    [[ $(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
    metrics="${candidate}.metrics.csv"
    score_log="${candidate}.score.log"
    robust="${candidate}.robust.json"
    "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
      --pred "${candidate}" --gt "${labels}" --csv "${metrics}" --workers 4 \
      > "${score_log}" 2>&1
    "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
      --anchor "${anchor_metrics}" --candidate "${metrics}" \
      --anchor-name champion_0p7006 --candidate-name "uls_consensus_${tag}" \
      --output "${robust}" > "${candidate}.robust.log"
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
    decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tag}" "${mutual}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
      "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
  done
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k4,4gr -k5,5gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

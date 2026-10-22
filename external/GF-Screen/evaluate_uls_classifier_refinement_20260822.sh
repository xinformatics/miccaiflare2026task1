#!/usr/bin/env bash
set -euo pipefail

# Public-only combination of genuinely out-of-fold component decisions with
# label-free ULS+ refinement. No hidden inference, fitting, ZIP, or upload.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
uls_work="${root}/public_sweeps/uls_plus_rich20_refiner_20260822/work"
uls_marker="${uls_work}/INFERENCE_COMPLETE.txt"
classifier="${root}/analysis/component_classifier_cv_20260822"
classifier_marker="${classifier}/COMPLETE.txt"
labels="${project}/validation/Validation-Public-Labels"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/uls_plus_classifier_oof_20260822"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "ULS+ classifier audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${uls_marker}" || ! -s "${classifier_marker}" ]]; do
  printf '%s waiting for ULS+ inference and grouped classifier\n' "$(date -Ins)"
  sleep 30
done

printf 'variant\tmode\tgate\tagreement\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${output}/summary.tsv"
for mode in refined seed_union; do
  for gate in safe relaxed; do
    if [[ "${gate}" == safe ]]; then
      coverage=0.50; overlap=0.10; min_ratio=0.25; max_ratio=4
    else
      coverage=0.25; overlap=0.05; min_ratio=0.10; max_ratio=10
    fi
    for agreement in 0 0.80; do
      agreement_tag=${agreement/./p}
      tag="${mode}_${gate}_a${agreement_tag}"
      candidate="${output}/${tag}"
      "${python_bin}" -u "${root}/materialize_uls_classifier_refinement.py" \
        --work "${uls_work}" --component-csv "${classifier}/components_oof.csv" \
        --base "${base}" --output "${candidate}" --mode "${mode}" \
        --min-seed-coverage "${coverage}" --min-refined-overlap "${overlap}" \
        --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
        --agreement-mutual "${agreement}" --agreement-min-mm3 500 \
        > "${candidate}.materialize.log" 2>&1
      [[ $(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
      metrics="${candidate}.metrics.csv"
      score_log="${candidate}.score.log"
      robust="${candidate}.robust.json"
      "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
        --pred "${candidate}" --gt "${labels}" --csv "${metrics}" --workers 4 \
        > "${score_log}" 2>&1
      "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
        --anchor "${anchor_metrics}" --candidate "${metrics}" \
        --anchor-name champion_0p7006 --candidate-name "uls_classifier_${tag}" \
        --output "${robust}" > "${candidate}.robust.log"
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
      decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${tag}" "${mode}" "${gate}" "${agreement}" "${dsc}" "${nsd}" \
        "${combined}" "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
    done
  done
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

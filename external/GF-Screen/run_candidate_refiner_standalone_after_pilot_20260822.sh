#!/usr/bin/env bash
set -euo pipefail

# Public-only standalone candidate-refiner evaluation. Unlike the anchored
# deployment, this starts from an empty mask and therefore can reject false
# positives already present in the champion. It never reads hidden data.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream="${root}/public_sweeps/candidate_refiner_v1_20260822/PUBLIC_ONLY_COMPLETE.txt"
source="${root}/public_sweeps/candidate_refiner_v1_20260822"
disc_work="${source}/work_iso15"
conn_work="${source}/work_connected_iso15"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
labels="${project}/validation/Validation-Public-Labels"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/candidate_refiner_standalone_v1_20260822"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_standalone_after_pilot_20260822.lock"
flock -n 9 || { echo "standalone refiner evaluator already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for candidate-refiner pilot public completion\n' "$(date -Ins)"
  sleep 30
done
for required in "${disc_work}/AUDIT.json" "${conn_work}/AUDIT.json" \
  "${disc_work}/INFERENCE_COMPLETE.json" "${conn_work}/INFERENCE_COMPLETE.json" \
  "${source}/classification_thresholds.txt" "${source}/segmentation_thresholds.txt" \
  "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

printf 'variant\tclassification_threshold\tsegmentation_threshold\tgate\tstrategy\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"
classification_values=$(cat "${source}/classification_thresholds.txt")
segmentation=$(cat "${source}/segmentation_thresholds.txt")

score_variant() {
  local tag="$1" classification="$2" gate="$3" strategy="$4" prediction="$5"
  local metrics="${prediction}.metrics.csv" score_log="${prediction}.score.log"
  local robust="${prediction}.robust.json" dsc nsd combined decision
  [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${gate}" "${strategy}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
}

for classification in ${classification_values}; do
  for gate_spec in 'strict 0.05 0.05 0.25 16.0' 'expand 0.05 0.001 0.10 64.0'; do
    read -r gate seed_coverage refined_overlap min_ratio max_ratio <<< "${gate_spec}"
    ctag=${classification/./p}; stag=${segmentation/./p}
    core="c${ctag}_s${stag}_${gate}"
    disc="${output}/${core}_disc"
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
      --work "${disc_work}" --base "${base}" --empty-base --output "${disc}" \
      --classification-threshold "${classification}" \
      --segmentation-threshold "${segmentation}" --mode refined \
      --min-seed-coverage "${seed_coverage}" \
      --min-refined-overlap "${refined_overlap}" \
      --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
      > "${disc}.materialize.log" 2>&1
    for strategy in refined approved; do
      if [[ "${strategy}" == approved ]]; then conn_mode=approved_replace; short=ap; else conn_mode=refined; short=rf; fi
      tag="stand_${core}_${short}"
      conn="${output}/${tag}_conn"
      prediction="${output}/${tag}"
      "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
        --work "${conn_work}" --base "${base}" --empty-base --output "${conn}" \
        --classification-threshold "${classification}" \
        --segmentation-threshold "${segmentation}" --mode "${conn_mode}" \
        --min-seed-coverage "${seed_coverage}" \
        --min-refined-overlap "${refined_overlap}" \
        --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
        --min-base-dice 0.25 --min-base-seed-coverage 0.25 \
        > "${conn}.materialize.log" 2>&1
      "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
        --pred-a "${disc}" --pred-b "${conn}" --fusion union --min-mm3 0 \
        --output "${prediction}" --workers 4 > "${prediction}.union.log" 2>&1
      score_variant "${tag}" "${classification}" "${gate}" "${strategy}" "${prediction}"
    done
  done
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

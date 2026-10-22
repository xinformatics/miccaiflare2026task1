#!/usr/bin/env bash
set -euo pipefail

# Public-only, champion-preserving component consensus. The new unlabeled
# balanced Glance arm proposes and seeds disconnected lesions; the independent
# conservative arm and BiomedParse prompts provide confirmation. Hidden data is
# never read here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_glance_unlabeled_deployed_20260822"
output="${root}/public_sweeps/gf_unlabeled_component_consensus_20260822"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
labels="${project}/validation/Validation-Public-Labels"
whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
anchor="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
balanced="gf_glance_ul_bal_ep280_2ep_20260822"
conservative="gf_glance_ul_con_ep280_2ep_20260822"

mkdir -p "${output}/robust"
exec 9>"${output}.lock"
flock -n 9 || { echo "unlabeled component-consensus evaluation already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" ]]; do sleep 30; done

resolve_inference() {
  local arm="$1" glance="$2" result
  result=$(find "${audit}/${arm}" -type d \
    -path "*/checkpoint_epoch_0282_*/g${glance}/inference" -print -quit)
  [[ -n "${result}" && $(find "${result}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "missing complete inference for ${arm} g${glance}" >&2; return 2;
  }
  printf '%s\n' "${result}"
}

candidate=$(resolve_inference "${balanced}" 0p20)
seed=$(resolve_inference "${balanced}" 0p50)
persistence=$(resolve_inference "${conservative}" 0p35)
for directory in "${base}" "${labels}" "${whole}" "${tumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public masks: ${directory}" >&2; exit 2;
  }
done

"${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" screen \
  --base "${base}" --candidate "${candidate}" --seed "${seed}" \
  --persistence "${persistence}" --whole "${whole}" --tumor "${tumor}" \
  --gt "${labels}" --output "${output}" --workers 4 > "${output}/screen.log" 2>&1

# These are three predeclared, mechanistically distinct policies. Strict
# promotion requires paired robustness, not merely the best public mean.
printf 'config\tDSC\tNSD\tcombined\tchanged_cases\tselected_components\ttrue_selected_components\tmetrics\trobust\tdecision\tpolicy\n' > "${output}/qualified.tsv"
for config in union25_persist_2k_full dual10_persist_2k_full vote2_seed_2k_full; do
  metrics="${output}/metrics/${config}.metrics.csv"
  robust="${output}/robust/${config}.json"
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "ul_consensus_${config}" \
    --output "${robust}" > "${output}/robust/${config}.log"
  read -r dsc nsd < <("${python_bin}" -c \
    'import csv,sys,numpy as np;r=list(csv.DictReader(open(sys.argv[1])));print(np.mean([float(x["Lesion_DSC"]) for x in r]),np.mean([float(x["Lesion_NSD"]) for x in r]))' \
    "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.9f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  read -r changed selected true_selected < <(awk -F '\t' -v c="${config}" '$1==c{print $15,$16,$17}' "${output}/summary.tsv")
  tail_guard=$("${python_bin}" -c 'import json,sys;print(str(json.load(open(sys.argv[1]))["checks"]["tail_regression_guard"]).lower())' "${robust}")
  policy=$(awk -v g="${decision}" -v t="${tail_guard}" -v c="${combined}" -v n="${changed}" \
    'BEGIN{if(g=="PROMOTE")print "STRICT_PROMOTE";else if(t=="true" && n+0>0 && c>=0.5345)print "BOUNDED_EXPLORE";else print "REJECT"}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${config}" "${dsc}" "${nsd}" "${combined}" "${changed}" "${selected}" \
    "${true_selected}" "${metrics}" "${robust}" "${decision}" "${policy}" \
    >> "${output}/qualified.tsv"
done

{
  head -1 "${output}/qualified.tsv"
  tail -n +2 "${output}/qualified.tsv" | awk -F '\t' '$11!="REJECT"' | \
    sort -t $'\t' -k11,11r -k4,4gr -k2,2gr
} > "${output}/selected.tsv"
if [[ $(wc -l < "${output}/selected.tsv") -eq 1 ]]; then
  date -Ins > "${output}/NO_BOUNDED_EXPLORATION.txt"
else
  config=$(awk -F '\t' 'NR==2{print $1}' "${output}/selected.tsv")
  mkdir -p "${output}/selected_prediction"
  "${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" materialize \
    --base "${base}" --candidate "${candidate}" --seed "${seed}" \
    --persistence "${persistence}" --whole "${whole}" --tumor "${tumor}" \
    --config "${config}" --output "${output}/selected_prediction" --workers 4 \
    > "${output}/materialize.log" 2>&1
fi
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"

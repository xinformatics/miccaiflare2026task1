#!/usr/bin/env bash
set -euo pipefail

# Public-only gate for the unlabeled CBG recall continuation. The submitted
# gf_conf mask is immutable: this screen can only add independently confirmed,
# disconnected components. It never reads hidden images or creates a ZIP.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
deployed="${CBG_UL_DEPLOYED:-${root}/public_sweeps/gf_glance_unlabeled_cbg_deployed_20260823}"
output="${CBG_UL_OUTPUT:-${root}/public_sweeps/gf_conf_unlabeled_recall_additions_20260823}"
base="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf"
anchor="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/exact0.metrics.csv"
labels="${project}/validation/Validation-Public-Labels"
whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
balanced="${CBG_UL_BALANCED:-gf_glance_ul_cbg_bal_ep283_2ep_20260823}"
conservative="${CBG_UL_CONSERVATIVE:-gf_glance_ul_cbg_con_ep283_2ep_20260823}"

mkdir -p "${output}/robust"
exec 9>"${output}.lock"
flock -n 9 || { echo "unlabeled recall-addition screen already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${deployed}/PUBLIC_ONLY_COMPLETE.txt" ]]; do sleep 30; done

resolve_inference() {
  local arm="$1" glance="$2" result
  result=$(find "${deployed}/${arm}" -type d \
    -path "*/g${glance}/inference" -print | sort | tail -1)
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
[[ -s "${anchor}" ]]

"${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" screen \
  --base "${base}" --candidate "${candidate}" --seed "${seed}" \
  --persistence "${persistence}" --whole "${whole}" --tumor "${tumor}" \
  --gt "${labels}" --output "${output}" --workers 4 > "${output}/screen.log" 2>&1

printf 'config\tDSC\tNSD\tcombined\tchanged_cases\tselected_components\ttrue_selected_components\tdecision\tmetrics\trobust\n' \
  > "${output}/shortlist.tsv"
for config in union25_persist_2k_full vote2_seed_2k_full union25_persist_5k_full; do
  metrics="${output}/metrics/${config}.metrics.csv"
  robust="${output}/robust/${config}.json"
  [[ -s "${metrics}" ]]
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor}" --candidate "${metrics}" \
    --anchor-name gf_conf_cbg41_e283 --candidate-name "ul_recall_${config}" \
    --output "${robust}" > "${output}/robust/${config}.log"
  read -r dsc nsd changed selected true_selected < <(
    "${python_bin}" - "${output}/summary.tsv" "${config}" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
row=next(item for item in rows if item['name']==sys.argv[2])
print(row['DSC'],row['NSD'],row['changed_cases'],row['selected_components'],row['true_selected_components'])
PY
  )
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.9f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${config}" "${dsc}" "${nsd}" "${combined}" "${changed}" "${selected}" \
    "${true_selected}" "${decision}" "${metrics}" "${robust}" >> "${output}/shortlist.tsv"
done

{
  head -1 "${output}/shortlist.tsv"
  tail -n +2 "${output}/shortlist.tsv" | sort -t $'\t' -k8,8r -k4,4gr -k2,2gr
} > "${output}/shortlist_ranked.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/shortlist_ranked.tsv"

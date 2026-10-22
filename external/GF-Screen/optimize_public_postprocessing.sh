#!/usr/bin/env bash
set -euo pipefail

root="${1:?usage: optimize_public_postprocessing.sh SWEEP_ROOT LABEL_DIR}"
label_dir="${2:?usage: optimize_public_postprocessing.sh SWEEP_ROOT LABEL_DIR}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
evaluator="${EVALUATOR:-../../handoff_bundle/code/evaluate.py}"
base_summary="${root}/summary.tsv"
post_root="${root}/postprocess"
post_summary="${post_root}/summary.tsv"
mkdir -p "${post_root}"

best_for_prefix() {
  local prefix="$1"
  awk -F '\t' -v prefix="${prefix}" '
    NR > 1 && index($1, prefix) == 1 {
      score = $2 + $3
      if (!found || score > best) {found=1; best=score; name=$1}
    }
    END {if (found) print name}
  ' "${base_summary}"
}

best_abdomen=$(best_for_prefix abdomen_)
best_chest=$(best_for_prefix chest_)
best_single=$(awk -F '\t' '
  NR > 1 {score=$2+$3; if (!found || score > best) {found=1; best=score; name=$1}}
  END {if (found) print name}
' "${base_summary}")

if [[ -z "${best_abdomen}" || -z "${best_chest}" || -z "${best_single}" ]]; then
  echo "Could not select base configurations from ${base_summary}" >&2
  exit 1
fi

echo -e "configuration\tDSC\tNSD\tfusion\tsource_a\tsource_b\tmin_mm3" > "${post_summary}"

evaluate_candidate() {
  local name="$1"
  local fusion="$2"
  local source_a="$3"
  local source_b="$4"
  local min_mm3="$5"
  local candidate_dir="${post_root}/${name}"
  local pred_dir="${candidate_dir}/predictions"
  mkdir -p "${candidate_dir}"

  local command=("${python_bin}" -u fuse_and_filter_masks.py
    --pred-a "${root}/${source_a}/predictions"
    --fusion "${fusion}" --min-mm3 "${min_mm3}" --output "${pred_dir}")
  if [[ -n "${source_b}" ]]; then
    command+=(--pred-b "${root}/${source_b}/predictions")
  fi
  "${command[@]}" > "${candidate_dir}/transform.log" 2>&1

  "${python_bin}" -u "${evaluator}" --pred "${pred_dir}" --gt "${label_dir}" \
    --csv "${candidate_dir}/metrics.csv" > "${candidate_dir}/score.log" 2>&1
  local dsc nsd
  dsc=$(awk '/^DSC  mean/ {print $3}' "${candidate_dir}/score.log")
  nsd=$(awk '/^NSD  mean/ {print $3}' "${candidate_dir}/score.log")
  echo -e "${name}\t${dsc}\t${nsd}\t${fusion}\t${source_a}\t${source_b}\t${min_mm3}" \
    | tee -a "${post_summary}"
}

for min_mm3 in 0 10 25 50 100 200 500; do
  evaluate_candidate "single_${best_single}_mm${min_mm3}" a "${best_single}" "" "${min_mm3}"
done
for min_mm3 in 0 10 25 50 100 200 500; do
  evaluate_candidate "union_${best_abdomen}_${best_chest}_mm${min_mm3}" union \
    "${best_abdomen}" "${best_chest}" "${min_mm3}"
done
evaluate_candidate "intersection_${best_abdomen}_${best_chest}_mm0" intersection \
  "${best_abdomen}" "${best_chest}" 0

awk -F '\t' '
  NR == 1 {next}
  {score=$2+$3; if (!found || score > best) {found=1; best=score; row=$0}}
  END {if (found) print row}
' "${post_summary}" > "${post_root}/best.tsv"

echo "Best post-processing configuration:"
cat "${post_root}/best.tsv"

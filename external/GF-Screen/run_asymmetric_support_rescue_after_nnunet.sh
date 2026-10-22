#!/usr/bin/env bash
set -euo pipefail

wait_pid="${WAIT_PID:-4194303}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
project_root="../.."
root="public_sweeps/asymmetric_support_rescue_20260820"
guided_root="public_sweeps/nnunet_guided_focus_20260820"
public_gt="${project_root}/validation/Validation-Public-Labels"
public_base="predictions/gf_a055_bbox_pu_best_ep280/public_mm1000"
hidden_base="predictions/gf_a055_bbox_pu_best_ep280/hidden_mm1000"
primary_public="${project_root}/external/BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
primary_hidden="${project_root}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
anchor="${project_root}/external/BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"

while kill -0 "${wait_pid}" 2>/dev/null; do sleep 30; done
support_tag=$(awk -F'\t' 'NR==2 {print $1}' "${guided_root}/support.tsv")
secondary_public=$(awk -F'\t' 'NR==2 {print $2}' "${guided_root}/support.tsv")
secondary_hidden="${guided_root}/hidden_support_${support_tag}/combined"
for specification in \
  "${public_gt}:50" "${public_base}:50" "${primary_public}:50" \
  "${secondary_public}:50" "${hidden_base}:100" "${primary_hidden}:100" \
  "${secondary_hidden}:100"; do
  directory=${specification%:*}
  expected=${specification##*:}
  count=$(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq "${expected}" ]] || {
    echo "expected ${expected} masks in ${directory}; found ${count}" >&2
    exit 1
  }
done
mkdir -p "${root}/screen" "${root}/gates" "${project_root}/submissions"
"${python_bin}" -u asymmetric_component_rescue.py screen \
  --gt "${public_gt}" --base "${public_base}" --primary "${primary_public}" \
  --secondary "${secondary_public}" --output "${root}/screen" --workers 8 \
  > "${root}/screen.log" 2>&1

printf 'tag\tDSC\tNSD\tcombined\tmetrics\tconfig\tgate\tdecision\n' > "${root}/ranking.tsv"
while IFS=$'\t' read -r tag dsc nsd combined metrics config; do
  gate="${root}/gates/${tag}.json"
  "${python_bin}" select_checkpoint_robust.py --anchor "${anchor}" \
    --candidate "${metrics}" --anchor-name prompt_champion \
    --candidate-name "asymmetric_${support_tag}_${tag}" --output "${gate}" \
    > "${root}/gates/${tag}.log"
  decision=$(awk -F'"' '/"decision":/ {print $4}' "${gate}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${config}" \
    "${gate}" "${decision}" >> "${root}/ranking.tsv"
done < <(tail -n +2 "${root}/screen/summary.tsv")
{
  head -1 "${root}/ranking.tsv"
  tail -n +2 "${root}/ranking.tsv" | sort -t $'\t' -k8,8r -k4,4gr
} > "${root}/ranking.sorted.tsv"

best=$(sed -n '2p' "${root}/ranking.sorted.tsv")
IFS=$'\t' read -r tag dsc nsd combined metrics config gate decision <<< "${best}"
hidden_output="${root}/hidden_${support_tag}_${tag}"
"${python_bin}" -u asymmetric_component_rescue.py materialize \
  --base "${hidden_base}" --primary "${primary_hidden}" \
  --secondary "${secondary_hidden}" --config "${config}" \
  --output "${hidden_output}" --workers 8 > "${root}/materialize.log" 2>&1
submission="${project_root}/submissions/gf_asymmetric_${support_tag}_${tag}.zip"
"${python_bin}" "${project_root}/nnunet/package_codabench.py" \
  --images "${project_root}/validation/Validation-Hidden-Images" \
  --pred-dir "${hidden_output}" --output "${submission}" --expected-count 100
unzip -t "${submission}" > "${submission}.zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n' > "${root}/submission_candidate.tsv"
printf 'asymmetric_%s_%s\tasymmetric_secondary_rescue\t%s\t%s\t%s\n' \
  "${support_tag}" "${tag}" "${metrics}" "${submission}" "${gate}" \
  >> "${root}/submission_candidate.tsv"
printf '%s\n' "${best}" > "${root}/selected.tsv"
echo "asymmetric rescue candidate ready: ${submission} (${decision}, DSC=${dsc}, NSD=${nsd})"

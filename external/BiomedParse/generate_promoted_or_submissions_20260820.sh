#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
root="${workspace}/external/BiomedParse"
gf="${workspace}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
hidden_images="${workspace}/validation/Validation-Hidden-Images"
anchor_hidden="${gf}/predictions/gf_a055_bbox_pu_best_ep280/hidden_mm1000"
component16_hidden="${gf}/public_sweeps/completed_bbox_variants_20260820/hidden_component16_ep283_l0p60_mm1000/filtered"
public_root="${root}/public_sweeps/automatic_text_proposals_20260820"
combined="${root}/public_sweeps/combined_prompt_confirmation_20260820"
component16_audit="${root}/public_sweeps/component16_biomedparse_confirmation_20260820"
topk_audit="${root}/public_sweeps/champion_topk_rescue_20260820"
input_root="${public_root}/hidden_biomedparse_direct_whole_body_lesion_e030_confirm_distance_mm_0_rescue_min_mm3_inf/inputs"
output="${public_root}/hidden_prompt_or_e050_confirmation"
wait_pid="${WAIT_PID:-1098962}"

[[ "${wait_pid}" =~ ^[0-9]+$ ]] || { echo "WAIT_PID must be numeric" >&2; exit 1; }
while kill -0 "${wait_pid}" 2>/dev/null; do sleep 15; done
[[ $(find "${input_root}" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]] || {
  echo "expected 100 pre-staged hidden inputs" >&2
  exit 1
}
mkdir -p "${output}"/{whole_npz,tumor_npz,whole_nifti,tumor_nifti,prompt_or,candidates} \
  "${workspace}/submissions"

infer_prompt() {
  local gpu="$1" prompt="$2" destination="$3" log="$4"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${input_root}" --output-dir "${destination}" \
      --prompt "${prompt}" --device cuda --slice-batch-size 8 \
      --existence-threshold 0.50 --continue-prediction > "${log}" 2>&1; then
    echo "batch 8 failed; resuming at batch 4" >> "${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${input_root}" --output-dir "${destination}" \
      --prompt "${prompt}" --device cuda --slice-batch-size 4 \
      --existence-threshold 0.50 --continue-prediction >> "${log}" 2>&1
  fi
}

infer_prompt 0 "whole-body lesion" "${output}/whole_npz" "${output}/whole.inference.log" & p0=$!
infer_prompt 1 "Tumor" "${output}/tumor_npz" "${output}/tumor.inference.log" & p1=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
for variant in whole tumor; do
  [[ $(find "${output}/${variant}_npz" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]] || {
    echo "${variant}: expected 100 BiomedParse NPZ files" >&2
    exit 1
  }
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
    --images "${hidden_images}" --predictions "${output}/${variant}_npz" \
    --output "${output}/${variant}_nifti" --segmentation-key segs_e050 \
    --continue-prediction > "${output}/${variant}.convert.log" 2>&1
done
"${python_bin}" -u "${gf}/fuse_and_filter_masks.py" \
  --pred-a "${output}/whole_nifti" --pred-b "${output}/tumor_nifti" \
  --fusion union --min-mm3 0 --workers 4 --output "${output}/prompt_or" \
  > "${output}/prompt_or.log" 2>&1
[[ $(find "${output}/prompt_or" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 OR support masks" >&2
  exit 1
}

manifest="${output}/submission_manifest.tsv"
printf 'tag\tbase\tconfig\tpredictions\tsubmission\n' > "${manifest}"
materialize_candidate() {
  local tag="$1" base="$2" config="$3"
  local predictions="${output}/candidates/${tag}"
  local submission="${workspace}/submissions/gf_${tag}.zip"
  mkdir -p "${predictions}"
  "${python_bin}" -u "${root}/biomedparse_fusion.py" materialize \
    --base "${base}" --support "${output}/prompt_or" --config "${config}" \
    --output "${predictions}" > "${output}/${tag}.fusion.log" 2>&1
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "${tag}: expected 100 predictions" >&2
    return 1
  }
  "${python_bin}" -u "${workspace}/nnunet/package_codabench.py" \
    --images "${hidden_images}" --pred-dir "${predictions}" \
    --output "${submission}" --expected-count 100 > "${output}/${tag}.package.log" 2>&1
  unzip -t "${submission}" > "${output}/${tag}.zip_test.log"
  sha256sum "${submission}" > "${submission}.sha256"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${base}" "${config}" "${predictions}" "${submission}" >> "${manifest}"
}

no_rescue_config="${combined}/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
[[ -f "${no_rescue_config}" ]] || {
  echo "promoted OR no-rescue config is missing" >&2
  exit 1
}
materialize_candidate "prompt_or_e050_confirm_d0_norescue" \
  "${anchor_hidden}" "${no_rescue_config}"

decision_of() {
  awk -F'"' '/"decision":/ {print $4}' "$1"
}
if [[ -s "${topk_audit}/robust_selection.json" ]]; then
  if [[ $(decision_of "${topk_audit}/robust_selection.json") == "PROMOTE" ]]; then
    topk_line=$(sed -n '2p' "${topk_audit}/summary_by_combined.tsv")
    IFS=$'\t' read -r topk_tag _ _ _ _ _ _ topk_config <<< "${topk_line}"
    if ! cmp -s "${topk_config}" "${no_rescue_config}"; then
      materialize_candidate "prompt_or_e050_${topk_tag}" \
        "${anchor_hidden}" "${topk_config}"
    fi
  fi
fi
if [[ -s "${component16_audit}/robust_vs_champion.json" ]]; then
  if [[ $(decision_of "${component16_audit}/robust_vs_champion.json") == "PROMOTE" ]]; then
    component_line=$(sed -n '2p' "${component16_audit}/summary_by_combined.tsv")
    IFS=$'\t' read -r component_tag _ _ _ _ _ _ _ component_config <<< "${component_line}"
    materialize_candidate "component16_prompt_or_e050_${component_tag}" \
      "${component16_hidden}" "${component_config}"
  fi
fi
confidence_selection="${root}/public_sweeps/component_confidence_rescue_20260820/selection.json"
if [[ -s "${confidence_selection}" ]] \
   && [[ $(awk -F'"' '/"decision":/ {print $4}' "${confidence_selection}") == "MATERIALIZE_SECONDARY" ]]; then
  bash "${root}/generate_confidence_rescue_submission_20260820.sh"
  confidence_tag="prompt_or_e050_focus075_tumor_mean_e010_mm1000_10000"
  confidence_predictions="${root}/public_sweeps/component_confidence_rescue_20260820/hidden_focus075_tumor_mean_e010_mm1000_10000/predictions"
  confidence_config="${root}/public_sweeps/component_confidence_rescue_20260820/focus075_tumor_mean_e010_mm1000_10000.json"
  confidence_submission="${workspace}/submissions/gf_${confidence_tag}.zip"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${confidence_tag}" "${anchor_hidden}" "${confidence_config}" \
    "${confidence_predictions}" "${confidence_submission}" >> "${manifest}"
fi
cat "${manifest}"

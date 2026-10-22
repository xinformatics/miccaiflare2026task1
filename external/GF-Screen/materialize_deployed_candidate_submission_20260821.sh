#!/usr/bin/env bash
set -euo pipefail

# Materialize a public-gated GF checkpoint with the exact current champion
# deployment recipe.  This script saves a local ZIP only; it never uploads.
# Required environment: CHECKPOINT, TAG, METRICS, ROBUST_JSON.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${CHECKPOINT:?set CHECKPOINT}"
tag="${TAG:?set TAG}"
metrics="${METRICS:?set METRICS}"
robust_json="${ROBUST_JSON:?set ROBUST_JSON}"
glance_threshold="${GLANCE_THRESHOLD:-0.50}"
lesion_threshold="${LESION_THRESHOLD:-0.50}"
glance_box_localization="${GLANCE_BOX_LOCALIZATION:-0}"
glance_center_localization="${GLANCE_CENTER_LOCALIZATION:-0}"
glance_center_threshold="${GLANCE_CENTER_THRESHOLD:-}"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
output="${root}/hidden_predictions/deployed_candidates_20260821/${tag}"
submission="${project}/submissions/${tag}.zip"

[[ "${tag}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe TAG: ${tag}" >&2; exit 2; }
[[ ${#tag} -le 55 ]] || { echo "TAG must be <=55 characters" >&2; exit 2; }
for file in "${checkpoint}" "${metrics}" "${robust_json}" "${prompt_config}"; do
  [[ -s "${file}" ]] || { echo "missing required file: ${file}" >&2; exit 2; }
done
decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust_json}")
if [[ "${decision}" != "PROMOTE" && "${ALLOW_FALLBACK:-0}" != 1 ]]; then
  echo "refusing hidden materialization: robustness decision is ${decision}" >&2
  exit 3
fi
for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 hidden files: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${output}"/{shard0,shard1,raw,filtered,prompted,deployed} \
  "${project}/submissions"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "candidate materialization already active: ${tag}" >&2
  exit 1
fi

infer_shard() {
  local gpu="$1" shard="$2" batch=16
  local log="${output}/shard${shard}.log"
  local -a model_args=()
  if [[ "${glance_box_localization}" == 1 ]]; then
    model_args+=(--glance_box_localization)
  fi
  if [[ "${glance_center_localization}" == 1 ]]; then
    model_args+=(--glance_center_localization)
  fi
  if [[ -n "${glance_center_threshold}" ]]; then
    [[ "${glance_center_localization}" == 1 ]] || {
      echo "GLANCE_CENTER_THRESHOLD requires GLANCE_CENTER_LOCALIZATION=1" >&2
      return 2
    }
    model_args+=(--glance_center_threshold "${glance_center_threshold}")
  fi
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${output}/shard${shard}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold "${glance_threshold}" --lesion_threshold "${lesion_threshold}" \
      --focus_batch_size "${batch}" --workers 2 --num_shards 2 --shard_id "${shard}" \
      "${model_args[@]}" --skip_existing > "${log}" 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${output}/shard${shard}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold "${glance_threshold}" --lesion_threshold "${lesion_threshold}" \
      --focus_batch_size 8 --workers 2 --num_shards 2 --shard_id "${shard}" \
      "${model_args[@]}" --skip_existing >> "${log}" 2>&1
  fi
}
infer_shard 0 0 & p0=$!
infer_shard 1 1 & p1=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
cp "${output}/shard0"/*.nii.gz "${output}/raw/"
cp "${output}/shard1"/*.nii.gz "${output}/raw/"
[[ $(find "${output}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${output}/raw" --fusion a --min-mm3 1000 --workers 4 \
  --output "${output}/filtered" > "${output}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${output}/filtered" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${output}/prompted" \
  > "${output}/prompted.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${output}/prompted" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${output}/deployed" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${output}/deployed.log" 2>&1
[[ $(find "${output}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/deployed" \
  --output "${submission}" --expected-count 100 > "${output}/package.log" 2>&1
unzip -t "${submission}" > "${output}/zip_test.log"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
sha256sum "${submission}" | tee "${submission}.sha256"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tdeployed_candidate\t%s\t%s\t%s\n' \
  "${tag}" "$(realpath "${metrics}")" "$(realpath "${submission}")" \
  "$(realpath "${robust_json}")" > "${output}/submission_candidate.tsv"
echo "saved submission candidate: ${submission}"

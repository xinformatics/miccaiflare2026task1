#!/usr/bin/env bash
set -euo pipefail

# Public/hidden ordering is weak for partial-label training, but hidden
# submissions are scarce. Require the best Context-24 candidate to beat the
# actual prompt-confirmed deployed champion before materialization.
wait_pid="${WAIT_PID:-828563}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
root="${OUTPUT_ROOT:-public_sweeps/gf_context24_dual_20260820}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
biomed_root="../BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
summary="${root}/summary_by_combined.tsv"
if [[ ! -f "${summary}" ]]; then
  echo "Context-24 summary was not produced: ${summary}" >&2
  exit 1
fi
best_line=$(sed -n '2p' "${summary}")
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min best_dsc best_nsd best_combined \
  <<< "${best_line}"
candidate_tag="${best_variant}_${best_tag}_l${best_focus/./p}_mm${best_min}"
score_tag="${best_variant}_${best_tag}_lesion_$(printf '%.2f' "${best_focus}")_mm${best_min}"
if [[ "${best_min}" -gt 0 ]]; then
  public_base="${root}/scored/${score_tag}"
else
  public_base="${root}/${best_variant}/${best_tag}/lesion_$(printf '%.2f' "${best_focus}")"
fi
for required in "${prompt_config}" "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing prompt-confirmation asset: ${required}" >&2; exit 1; }
done
[[ $(find "${public_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 selected Context-24 public masks" >&2
  exit 1
}
[[ $(find "${public_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
[[ $(find "${hidden_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}
public_prompt="${root}/public_prompt_${candidate_tag}"
mkdir -p "${public_prompt}"
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${public_base}" --support "${public_prompt_support}" \
  --config "${prompt_config}" --output "${public_prompt}" \
  > "${root}/public_prompt_materialize.log" 2>&1
prompt_metrics="${root}/public_prompt_${candidate_tag}.metrics.csv"
"${python_bin}" -u evaluate_public_parallel.py --pred "${public_prompt}" \
  --gt "${public_labels}" --csv "${prompt_metrics}" --workers 8 \
  > "${root}/public_prompt.score.log" 2>&1
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${prompt_anchor}" --candidate "${prompt_metrics}" \
  --anchor-name prompt_confirmed_champion \
  --candidate-name "${candidate_tag}_prompt_or_e050" \
  --output "${root}/robust_selection_prompt_confirmed.json" \
  > "${root}/robust_selection_prompt_confirmed.log"
prompt_decision=$(awk -F'"' '/"decision":/ {print $4}' \
  "${root}/robust_selection_prompt_confirmed.json")
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "Context-24 candidate did not beat the prompt-confirmed champion; no hidden inference."
  exit 0
fi
submission="../../submissions/gf_${candidate_tag}_prompt_or_e050.zip"
if [[ -f "${submission}" ]]; then
  unzip -t "${submission}" >/dev/null
  printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tcontext24_short\t%s\t%s\t%s\n' \
    "${candidate_tag}" "$(realpath "${prompt_metrics}")" \
    "$(realpath "${submission}")" \
    "$(realpath "${root}/robust_selection_prompt_confirmed.json")" \
    > "${root}/submission_candidate.tsv"
  echo "Exploratory Context-24 submission already exists: ${submission}"
  exit 0
fi

hidden_root="${root}/hidden_exploratory_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
wait "${hidden_pid0}"
wait "${hidden_pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
hidden_count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${hidden_count}" -ne 100 ]]; then
  echo "Expected 100 hidden predictions, found ${hidden_count}" >&2
  exit 1
fi
"${python_bin}" -u fuse_and_filter_masks.py \
  --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min}" \
  --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${hidden_prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden masks" >&2
  exit 1
}
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tcontext24_short\t%s\t%s\t%s\n' \
  "${candidate_tag}" "$(realpath "${prompt_metrics}")" \
  "$(realpath "${submission}")" \
  "$(realpath "${root}/robust_selection_prompt_confirmed.json")" \
  > "${root}/submission_candidate.tsv"
echo "Prompt-confirmed Context-24 submission ready: ${submission}"

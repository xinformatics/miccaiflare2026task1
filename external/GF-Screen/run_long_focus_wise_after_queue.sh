#!/usr/bin/env bash
set -euo pipefail

# Stabilize the two long Focus endpoints by interpolating them back toward the
# immutable 0.6889 hidden-DSC bounding-box champion. Alpha 0 is the already
# scored champion and alpha 1 is the already scored adapted endpoint, so only
# the three interior points require inference.
wait_pid="${WAIT_PID:-4194303}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
anchor_checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
long_root="public_sweeps/long_focus_dual_20260820"
long_summary="${long_root}/summary_by_combined.tsv"
root="${OUTPUT_ROOT:-public_sweeps/long_focus_wise_dual_20260820}"
interpolation_root="checkpoint_interpolations/long_focus_wise_20260820"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
biomed_root="../BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"

while kill -0 "${wait_pid}" 2>/dev/null; do sleep 30; done
[[ -f "${anchor_checkpoint}" && -s "${long_summary}" ]] || {
  echo "champion checkpoint or long-Focus summary is missing" >&2
  exit 1
}
for required in "${prompt_config}" "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing prompt-confirmation asset: ${required}" >&2; exit 1; }
done
[[ $(find "${public_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
[[ $(find "${hidden_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}
mkdir -p "${root}" "${interpolation_root}" ../../submissions
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/miccai26-mpl}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_pair() {
  local left="$1" right="$2" status=0
  wait "${left}" || status=$?
  wait "${right}" || status=$?
  return "${status}"
}

run_priority_guided_focus() {
  # Keep the active queue blocked until the highest-priority recall-recovery
  # audit finishes. The independent legacy watcher will later see the same
  # AUDIT_COMPLETE marker and exit without repeating inference.
  SCORE_WORKERS="${SCORE_WORKERS:-4}" \
    bash ../BiomedParse/run_public_audit_before_focus_queue.sh
}

# summary_by_combined.tsv is globally sorted. Retain the best row from each of
# the two distinct long-run variants, preserving that global order.
{
  head -1 "${long_summary}"
  awk -F $'\t' 'NR > 1 && !seen[$1]++ {print; count++; if (count == 2) exit}' \
    "${long_summary}"
} > "${root}/selected_endpoints.tsv"
mapfile -t endpoints < <(tail -n +2 "${root}/selected_endpoints.tsv")
[[ "${#endpoints[@]}" -eq 2 ]] || {
  echo "expected two distinct long-Focus endpoints, found ${#endpoints[@]}" >&2
  exit 1
}

variants=()
checkpoint_tags=()
endpoint_checkpoints=()
interpolation_dirs=()
for index in 0 1; do
  IFS=$'\t' read -r variant checkpoint_tag checkpoint _rest <<< "${endpoints[$index]}"
  [[ -f "${checkpoint}" ]] || { echo "missing endpoint: ${checkpoint}" >&2; exit 1; }
  variants+=("${variant}")
  checkpoint_tags+=("${checkpoint_tag}")
  endpoint_checkpoints+=("${checkpoint}")
  interpolation_dir="${interpolation_root}/${variant}_${checkpoint_tag}"
  interpolation_dirs+=("${interpolation_dir}")
  if [[ $(find "${interpolation_dir}" -maxdepth 1 -name 'wise_focus_alpha*.pt' 2>/dev/null | wc -l) -ne 3 ]]; then
    "${python_bin}" -u interpolate_checkpoints.py \
      --anchor "${anchor_checkpoint}" --adapted "${checkpoint}" \
      --output-dir "${interpolation_dir}" --alphas 0.25 0.50 0.75 \
      --scopes focus > "${interpolation_dir}.log" 2>&1
  fi
done

evaluate_endpoint() {
  local gpu="$1" index="$2"
  local variant="${variants[$index]}" endpoint_tag="${checkpoint_tags[$index]}"
  local interpolation_dir="${interpolation_dirs[$index]}"
  local output="${root}/${variant}_${endpoint_tag}"
  mkdir -p "${output}"
  : > "${output}/candidates.tsv"
  for checkpoint in "${interpolation_dir}"/wise_focus_alpha{0p250,0p500,0p750}.pt; do
    [[ -f "${checkpoint}" ]] || { echo "missing interpolation: ${checkpoint}" >&2; return 1; }
    tag=$(basename "${checkpoint}" .pt)
    destination="${output}/${tag}"
    mkdir -p "${destination}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
        --test_data_path "${public_images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --focus_batch_size 16 --workers 2 --skip_existing \
        > "${destination}.inference.log" 2>&1; then
      echo "Focus batch 16 failed; resuming completed cases at batch 8" \
        >> "${destination}.inference.log"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
        --test_data_path "${public_images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --focus_batch_size 8 --workers 2 --skip_existing \
        >> "${destination}.inference.log" 2>&1
    fi
    printf '%s\t%s\t%s\t%s\n' \
      "${variant}" "${tag}" "${checkpoint}" "${destination}" \
      >> "${output}/candidates.tsv"
  done
}

evaluate_endpoint 0 0 & eval0=$!
evaluate_endpoint 1 1 & eval1=$!
wait_pair "${eval0}" "${eval1}"

manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\n' \
  > "${manifest}"
for index in 0 1; do
  endpoint_dir="${root}/${variants[$index]}_${checkpoint_tags[$index]}"
  while IFS=$'\t' read -r variant tag checkpoint destination; do
    for threshold in 0.50 0.60; do
      source="${destination}/lesion_$(printf '%.2f' "${threshold}")"
      [[ $(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
        echo "${source}: expected 50 masks" >&2
        exit 1
      }
      score_tag="${variant}_${checkpoint_tags[$index]}_${tag}_lesion_$(printf '%.2f' "${threshold}")_mm1000"
      filtered="${root}/scored/${score_tag}"
      mkdir -p "${filtered}"
      if [[ $(find "${filtered}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
        "${python_bin}" -u fuse_and_filter_masks.py --pred-a "${source}" \
          --fusion a --min-mm3 1000 --workers 4 --output "${filtered}" \
          > "${filtered}.filter.log" 2>&1
      fi
      printf '%s\t%s\t%s\t%s\t1000\t%s\t%s\t%s.metrics.csv\n' \
        "${variant}" "${tag}" "${checkpoint}" "${threshold}" \
        "${score_tag}" "${filtered}" "${root}/scored/${score_tag}" >> "${manifest}"
    done
  done < "${endpoint_dir}/candidates.tsv"
done

grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ tag predictions _; do
  grid_args+=(--candidate "${tag}=${predictions}")
done < <(tail -n +2 "${manifest}")
"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1

printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
while IFS=$'\t' read -r variant tag checkpoint threshold min_mm3 score_tag _ _; do
  metrics="${root}/scored/${score_tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${tag}" "${checkpoint}" "${threshold}" "${min_mm3}" \
    "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")
{ head -1 "${root}/summary.tsv"; tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k8,8gr; } \
  > "${root}/summary_by_combined.tsv"

prompt_rank_root="${root}/prompted_ranking"
"${python_bin}" -u rank_prompted_manifest.py --manifest "${manifest}" \
  --output "${prompt_rank_root}" --labels "${public_labels}" \
  --prompt-support "${public_prompt_support}" --prompt-config "${prompt_config}" \
  --prompt-anchor "${prompt_anchor}" --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator evaluate_prompted_grid.py --selector select_checkpoint_robust.py \
  --workers 8 > "${root}/prompted_ranking.log" 2>&1
best_line=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min \
  score_tag selected_predictions selected_metrics public_prompt prompt_metrics \
  prompt_robust prompt_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best_line}"
candidate_tag="longwise_${best_variant}_${best_tag}_l${best_focus/./p}_mm1000"
"${python_bin}" -u select_checkpoint_robust.py --anchor "${anchor_metrics}" \
  --candidate "${selected_metrics}" --anchor-name champion_bbox \
  --candidate-name "${candidate_tag}" --output "${root}/robust_selection.json" \
  > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Long-Focus interpolation did not clear the raw paired gate; packaging as an explicitly ranked fallback."
fi

cp "${prompt_robust}" "${root}/robust_selection_prompt_confirmed.json"
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "Long-Focus interpolation did not clear the deployed paired gate; packaging as an explicitly ranked fallback."
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}"/{shard0,shard1,raw,filtered,prompt_confirmed}
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --focus_batch_size 8 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  [[ "${shard}" -eq 0 ]] && hp0=$! || hp1=$!
done
wait_pair "${hp0}" "${hp1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
[[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden masks" >&2
  exit 1
}
"${python_bin}" -u fuse_and_filter_masks.py --pred-a "${hidden_root}/raw" \
  --fusion a --min-mm3 1000 --workers 4 --output "${hidden_root}/filtered" \
  > "${hidden_root}/filter.log" 2>&1
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${hidden_prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden masks" >&2
  exit 1
}
submission="../../submissions/gf_${candidate_tag}_prompt_or_e050.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py --images "${hidden_images}" \
  --pred-dir "${hidden_root}/prompt_confirmed" --output "${submission}" --expected-count 100 \
  > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Long-Focus interpolation submission ready: ${submission}"
run_priority_guided_focus
"${python_bin}" -u build_three_submission_shortlist.py \
  --project ../.. --output ../../submissions/final_shortlist_20260820 \
  --anchor "${prompt_anchor}"

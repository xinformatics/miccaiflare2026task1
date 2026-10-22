#!/usr/bin/env bash
set -euo pipefail

# Bounded test-time augmentation audit on the immutable champion. Average
# Focus probabilities after restoring flipped crop orientation. A measured
# public maximum above 55 s/case is ineligible under the 60 s challenge limit.
wait_pid="${WAIT_PID:-878244}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/gf_champion_focus_tta_20260820}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
thresholds="0.45,0.50,0.55,0.60"
biomed_root="../BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
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
mkdir -p "${root}" ../../submissions

infer_variant() {
  local gpu="$1"
  local tag="$2"
  local axes="$3"
  local output="${root}/${tag}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds "${thresholds}" \
    --focus_tta_axes "${axes}" --focus_batch_size 2 --workers 2 \
    > "${output}.inference.log" 2>&1
}

infer_variant 0 tta_x x &
pid_x=$!
infer_variant 1 tta_xy xy &
pid_xy=$!
wait "${pid_x}"
wait "${pid_xy}"

printf 'variant\tflip_axes\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\tmean_seconds\tmax_seconds\truntime_eligible\n' \
  > "${root}/summary.tsv"
manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\tflip_axes\tmean_seconds\tmax_seconds\n' \
  > "${manifest}"
for specification in "tta_x:x" "tta_xy:xy"; do
  tag="${specification%%:*}"
  axes="${specification#*:}"
  read -r mean_seconds max_seconds <<< "$(awk '/^Duration:/ {s+=$2; n++; if($2>m)m=$2} END {printf "%.3f %.3f",s/n,m}' "${root}/${tag}.inference.log")"
  eligible=yes
  if awk -v maximum="${max_seconds}" 'BEGIN {exit !(maximum > 55.0)}'; then
    eligible=no
  fi
  for threshold in 0.45 0.50 0.55 0.60; do
    threshold_dir="lesion_${threshold}"
    source="${root}/${tag}/${threshold_dir}"
    if [[ $(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
      echo "Incomplete predictions: ${source}" >&2
      exit 1
    fi
    for min_mm3 in 500 1000; do
      score_stem="${root}/scored/${tag}_${threshold_dir}_mm${min_mm3}"
      mkdir -p "${score_stem}"
      "${python_bin}" -u fuse_and_filter_masks.py \
        --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${score_stem}" > "${score_stem}.filter.log" 2>&1
      if [[ "${eligible}" == "yes" ]]; then
        score_tag="${tag}_${threshold_dir}_mm${min_mm3}"
        printf '%s\tchampion\t%s\t%s\t%s\t%s\t%s\t%s.metrics.csv\t%s\t%s\t%s\n' \
          "${tag}" "${checkpoint}" "${threshold}" "${min_mm3}" "${score_tag}" \
          "${score_stem}" "${score_stem}" "${axes}" "${mean_seconds}" \
          "${max_seconds}" >> "${manifest}"
      fi
    done
  done
done

if [[ $(wc -l < "${manifest}") -le 1 ]]; then
  echo "No Focus TTA variant satisfied the runtime guard" >&2
  exit 0
fi
grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ score_tag predictions _ _ _ _; do
  grid_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${manifest}")
"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1
while IFS=$'\t' read -r tag _ _ threshold min_mm3 _ _ metrics axes mean_seconds max_seconds; do
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tyes\n' \
    "${tag}" "${axes}" "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" \
    "${combined}" "${mean_seconds}" "${max_seconds}" >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | awk -F'\t' '$10 == "yes"' | sort -t $'\t' -k7,7gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
prompt_rank_root="${root}/prompted_ranking"
"${python_bin}" -u rank_prompted_manifest.py --manifest "${manifest}" \
  --output "${prompt_rank_root}" --labels "${public_labels}" \
  --prompt-support "${public_prompt_support}" --prompt-config "${prompt_config}" \
  --prompt-anchor "${prompt_anchor}" --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator evaluate_prompted_grid.py --selector select_checkpoint_robust.py \
  --workers 8 > "${root}/prompted_ranking.log" 2>&1
best_line=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r best_tag _ _ best_focus best_min _ selected_predictions \
  selected_metrics best_axes best_mean best_max public_prompt prompt_metrics \
  prompt_robust prompt_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best_line}"
candidate_tag="${best_tag}_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name gf_champion --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Focus TTA candidate did not clear the raw paired gate; retaining fallback status."
fi
cp "${prompt_robust}" "${root}/robust_selection_prompt_confirmed.json"
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "Focus TTA candidate did not clear the deployed paired gate; packaging as an explicitly ranked fallback."
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --focus_tta_axes "${best_axes}" --focus_batch_size 2 --workers 2 \
    --num_shards 2 --shard_id "${shard}" > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
wait "${hidden_pid0}"
wait "${hidden_pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
if [[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  echo "Expected 100 hidden predictions" >&2
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
submission="../../submissions/gf_champion_${candidate_tag}_prompt_or_e050.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tfocus_tta\t%s\t%s\t%s\n' \
  "champion_${candidate_tag}" "$(realpath "${prompt_metrics}")" \
  "$(realpath "${submission}")" \
  "$(realpath "${root}/robust_selection_prompt_confirmed.json")" \
  > "${root}/submission_candidate.tsv"
echo "Focus TTA submission ready: ${submission}"

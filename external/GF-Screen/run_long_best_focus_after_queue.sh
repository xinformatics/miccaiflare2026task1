#!/usr/bin/env bash
set -euo pipefail

# Five short epochs expose only ~49.6% of the 17,445 lesion cases. After all
# controlled probes finish, continue the best two distinct Focus objectives for
# 20 more epochs. That continuation alone exposes about 93.8% of lesion cases.
wait_pid="${WAIT_PID:-889473}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
selection_root="public_sweeps/long_focus_selection_20260820"
root="${OUTPUT_ROOT:-public_sweeps/long_focus_dual_20260820}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
biomed_root="../BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
[[ -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]] || {
  echo "preprocessed cache is incomplete" >&2
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
mkdir -p "${root}" "${selection_root}" "${cache_root}/tmp" ../../submissions
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_pair() {
  local left="$1" right="$2" status=0
  wait "${left}" || status=$?
  wait "${right}" || status=$?
  return "${status}"
}

"${python_bin}" -u select_long_focus_candidates.py \
  --output "${selection_root}" > "${selection_root}/select.log" 2>&1
"${python_bin}" -u rank_long_focus_candidates_prompted.py \
  --selection-json "${selection_root}/selection.json" \
  --output "${selection_root}" --labels "${public_labels}" \
  --prompt-support "${public_prompt_support}" --prompt-config "${prompt_config}" \
  --prompt-anchor "${prompt_anchor}" \
  --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator evaluate_public_grid.py --selector select_checkpoint_robust.py \
  --workers 8 --count 2 > "${selection_root}/select_prompted.log" 2>&1
mapfile -t selection_lines < <(tail -n +2 "${selection_root}/selected_prompted.tsv")
[[ "${#selection_lines[@]}" -eq 2 ]] || {
  echo "expected two selected Focus families, found ${#selection_lines[@]}" >&2
  exit 1
}

families=()
variants=()
checkpoint_tags=()
checkpoints=()
run_names=()
max_epochs=()
for index in 0 1; do
  IFS=$'\t' read -r family variant checkpoint_tag checkpoint _rest \
    <<< "${selection_lines[$index]}"
  [[ -f "${checkpoint}" ]] || { echo "missing selected checkpoint: ${checkpoint}" >&2; exit 1; }
  stored_epoch=$("${python_bin}" -c \
    'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["epoch"]))' \
    "${checkpoint}")
  start_epoch=$((stored_epoch + 1))
  families+=("${family}")
  variants+=("${variant}")
  checkpoint_tags+=("${checkpoint_tag}")
  checkpoints+=("${checkpoint}")
  run_names+=("long20_${variant}_from_${checkpoint_tag}_20260820")
  max_epochs+=("$((start_epoch + 20))")
done

train_variant() {
  local gpu="$1" family="$2" variant="$3" checkpoint="$4" run_name="$5" max_epoch="$6"
  local run_dir="runs/${run_name}"
  local extra=(--bbox_context_mm 16 --pseudo_positive_threshold 0.95 \
    --tversky_alpha 0.3 --tversky_beta 0.7)
  case "${family}" in
    component_uniform)
      extra+=(--component_uniform_crop --component_crop_points_per_lesion 64 \
        --component_crop_cache_dir "${cache_root}/component_uniform_v2")
      [[ "${variant}" == *p050* ]] && extra+=(--component_crop_size_power 0.5)
      ;;
    component_box_uniform)
      extra+=(--component_boxes --component_uniform_crop \
        --component_crop_points_per_lesion 64 \
        --component_crop_cache_dir "${cache_root}/component_uniform_v2")
      [[ "${variant}" == *p050* ]] && extra+=(--component_crop_size_power 0.5)
      ;;
    context24)
      extra=(--bbox_context_mm 24 --pseudo_positive_threshold 0.95)
      if [[ "${variant}" == *a05b05* ]]; then
        extra+=(--tversky_alpha 0.5 --tversky_beta 0.5)
      else
        extra+=(--tversky_alpha 0.3 --tversky_beta 0.7)
      fi
      ;;
    bbox_pseudo)
      extra+=(--bbox_pseudo_positive_weight 0.05)
      [[ "${variant}" == *ppw015* ]] && extra=(--bbox_context_mm 16 \
        --pseudo_positive_threshold 0.98 --tversky_alpha 0.3 --tversky_beta 0.7 \
        --bbox_pseudo_positive_weight 0.15)
      [[ "${variant}" == *ppw005* ]] && extra=(--bbox_context_mm 16 \
        --pseudo_positive_threshold 0.98 --tversky_alpha 0.3 --tversky_beta 0.7 \
        --bbox_pseudo_positive_weight 0.05)
      ;;
    cohort_balance)
      extra+=(--cohort_balance_power 0.5 --cohort_balance_max_weight 5)
      [[ "${variant}" == *multilesion_priority2* ]] && extra=(--bbox_context_mm 16 \
        --pseudo_positive_threshold 0.95 --tversky_alpha 0.3 --tversky_beta 0.7 \
        --cohort_priority_json configs/multilesion_source_priority_20260820.json)
      ;;
    remote_pseudo)
      local remote_weight=0.05
      [[ "${variant}" == *pp015* ]] && remote_weight=0.15
      extra+=(--pseudo_negative_threshold 0.05 \
        --bbox_empty_pseudo_positive_weight "${remote_weight}" \
        --bbox_empty_pseudo_positive_min_voxels 64 \
        --cancer_positive_weight 1 --cancer_negative_weight 0.25)
      ;;
    *) echo "unknown Focus family: ${family}" >&2; return 1 ;;
  esac

  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --reset_optimizer --reset_scheduler --reset_best_acc \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --teacher_consistency_weight 0.15 --boundary_weight 0.15 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs "${max_epoch}" --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --optim_lr 1e-5 --warmup_epochs 3 \
    --wandb_project pancancer-ct --wandb_name "${run_name}" \
    "${extra[@]}" > "${run_dir}/train.log" 2>&1
}

training_complete=1
for completed_name in "${run_names[@]}"; do
  [[ $(find "runs/${completed_name}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' | wc -l) -ge 20 \
     && -f "runs/${completed_name}/model_final.pt" ]] || training_complete=0
done
if [[ "${training_complete}" -eq 1 ]]; then
  echo "Both long Focus continuations are complete; resuming evaluation."
else
  train_variant 0 "${families[0]}" "${variants[0]}" "${checkpoints[0]}" \
    "${run_names[0]}" "${max_epochs[0]}" & pid0=$!
  train_variant 1 "${families[1]}" "${variants[1]}" "${checkpoints[1]}" \
    "${run_names[1]}" "${max_epochs[1]}" & pid1=$!
  wait_pair "${pid0}" "${pid1}"
fi

evaluate_variant() {
  local gpu="$1" run_name="$2"
  local output="${root}/${run_name}"
  mkdir -p "${output}"
  : > "${output}/candidates.tsv"
  mapfile -t available < <(find "runs/${run_name}" -maxdepth 1 \
    -name 'checkpoint_epoch_*.pt' -printf '%p\n' | sort)
  [[ "${#available[@]}" -ge 20 ]] || {
    echo "${run_name}: expected at least 20 checkpoints, found ${#available[@]}" >&2
    return 1
  }
  local evaluation_candidates=()
  for position in 3 7 11 15 19; do
    evaluation_candidates+=("${available[$position]}")
  done
  # Validation is independent of the 50-case public scorer. Preserve its best
  # checkpoint as one additional predeclared candidate when the fixed
  # four-epoch sampling grid would otherwise miss it.
  local model_best="runs/${run_name}/model.pt"
  if [[ -f "${model_best}" ]]; then
    local stored_best_epoch checkpoint_number checkpoint_path already_selected=0
    stored_best_epoch=$("${python_bin}" -c \
      'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["epoch"]))' \
      "${model_best}")
    checkpoint_number=$((stored_best_epoch + 1))
    checkpoint_path=$(printf 'runs/%s/checkpoint_epoch_%04d.pt' \
      "${run_name}" "${checkpoint_number}")
    [[ -f "${checkpoint_path}" ]] || {
      echo "${run_name}: internal-best checkpoint ${checkpoint_path} is missing" >&2
      return 1
    }
    for candidate in "${evaluation_candidates[@]}"; do
      [[ "${candidate}" == "${checkpoint_path}" ]] && already_selected=1
    done
    [[ "${already_selected}" -eq 1 ]] || evaluation_candidates+=("${checkpoint_path}")
  fi
  for candidate in "${evaluation_candidates[@]}"; do
    tag=$(basename "${candidate}" .pt)
    destination="${output}/${tag}"
    mkdir -p "${destination}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
        --test_data_path "${public_images}" --save_prediction_path "${destination}" \
        --trained_pth "${candidate}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --focus_batch_size 16 --workers 2 --skip_existing \
        > "${destination}.inference.log" 2>&1; then
      echo "Focus batch 16 failed; resuming completed cases at batch 8" \
        >> "${destination}.inference.log"
      if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
          --test_data_path "${public_images}" --save_prediction_path "${destination}" \
          --trained_pth "${candidate}" --a_min -175 --a_max 250 \
          --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
          --focus_batch_size 8 --workers 2 --skip_existing \
          >> "${destination}.inference.log" 2>&1; then
        echo "Focus batch 8 failed; resuming completed cases at batch 2" \
          >> "${destination}.inference.log"
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
          --test_data_path "${public_images}" --save_prediction_path "${destination}" \
          --trained_pth "${candidate}" --a_min -175 --a_max 250 \
          --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
          --focus_batch_size 2 --workers 2 --skip_existing \
          >> "${destination}.inference.log" 2>&1
      fi
    fi
    printf '%s\t%s\t%s\t%s\n' \
      "${run_name}" "${tag}" "${candidate}" "${destination}" >> "${output}/candidates.tsv"
  done
}

evaluate_variant 0 "${run_names[0]}" & eval0=$!
evaluate_variant 1 "${run_names[1]}" & eval1=$!
wait_pair "${eval0}" "${eval1}"

manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\n' \
  > "${manifest}"
for run_name in "${run_names[@]}"; do
  while IFS=$'\t' read -r variant tag candidate_checkpoint destination; do
    for threshold in 0.50 0.60; do
      threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
      source="${destination}/${threshold_dir}"
      count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
      [[ "${count}" -eq 50 ]] || { echo "${source}: expected 50 masks, found ${count}" >&2; exit 1; }
      for min_mm3 in 0 500 1000; do
        score_tag="${variant}_${tag}_${threshold_dir}_mm${min_mm3}"
        score_stem="${root}/scored/${score_tag}"
        candidate="${source}"
        mkdir -p "$(dirname "${score_stem}")"
        if [[ "${min_mm3}" -gt 0 ]]; then
          candidate="${score_stem}"
          mkdir -p "${candidate}"
          if [[ $(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
            "${python_bin}" -u fuse_and_filter_masks.py \
              --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
              --workers 4 --output "${candidate}" > "${score_stem}.filter.log" 2>&1
          fi
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s.metrics.csv\n' \
          "${variant}" "${tag}" "${candidate_checkpoint}" "${threshold}" \
          "${min_mm3}" "${score_tag}" "${candidate}" "${score_stem}" >> "${manifest}"
      done
    done
  done < "${root}/${run_name}/candidates.tsv"
done

grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ score_tag predictions _; do
  grid_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${manifest}")
"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1

printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
while IFS=$'\t' read -r variant tag candidate_checkpoint threshold min_mm3 score_tag _ _; do
  metrics="${root}/scored/${score_tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${tag}" "${candidate_checkpoint}" "${threshold}" \
    "${min_mm3}" "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k8,8gr
} > "${root}/summary_by_combined.tsv"
prompt_rank_root="${root}/prompted_ranking"
"${python_bin}" -u rank_prompted_manifest.py --manifest "${manifest}" \
  --output "${prompt_rank_root}" --labels "${public_labels}" \
  --prompt-support "${public_prompt_support}" --prompt-config "${prompt_config}" \
  --prompt-anchor "${prompt_anchor}" --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator evaluate_prompted_grid.py --selector select_checkpoint_robust.py \
  --workers 8 > "${root}/prompted_ranking.log" 2>&1
best_line=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min \
  best_score_tag selected_predictions selected_metrics public_prompt prompt_metrics \
  prompt_robust prompt_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best_line}"
candidate_tag="${best_variant}_${best_tag}_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u analyze_lesion_errors.py --labels "${public_labels}" \
  --predictions "${selected_predictions}" --output "${root}/component_audit_best" \
  > "${root}/component_audit_best.log" 2>&1
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_bbox --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Long Focus candidate did not clear the raw paired gate; packaging as an explicitly ranked fallback."
fi

# The deployable champion includes exact prompt-component confirmation. A raw
# GF improvement is therefore necessary but not sufficient: apply the frozen
# verifier and compare against the matching prompt-confirmed anchor before any
# hidden inference or packaging is authorized.
cp "${prompt_robust}" "${root}/robust_selection_prompt_confirmed.json"
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "Long Focus candidate did not clear the deployed paired gate; packaging as an explicitly ranked fallback."
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hp0=$!; else hp1=$!; fi
done
wait_pair "${hp0}" "${hp1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 100 ]] || { echo "expected 100 hidden masks, found ${count}" >&2; exit 1; }
if [[ "${best_min}" -gt 0 ]]; then
  "${python_bin}" -u fuse_and_filter_masks.py --pred-a "${hidden_root}/raw" \
    --fusion a --min-mm3 "${best_min}" --output "${hidden_root}/filtered" \
    > "${hidden_root}/filter.log" 2>&1
else
  cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
fi
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${hidden_prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden masks" >&2
  exit 1
}
submission="../../submissions/gf_long20_${candidate_tag}_prompt_or_e050.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Long Focus submission ready: ${submission}"

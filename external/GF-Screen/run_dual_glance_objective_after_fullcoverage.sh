#!/usr/bin/env bash
set -euo pipefail

# Compare the published strong group-relative Glance objective (classification
# 0.1 / RL 1.0) with the conservative classification-heavy objective
# (classification 1.0 / RL 0.1).  Both arms use the same teacher-safe targets:
# annotated positives are preserved, sizeable high-confidence frozen-Focus
# islands become positives, teacher-confident empty crops become negatives,
# and uncertain crops are ignored.  This fixes the failure mode in which an
# unannotated lesion or a real lesion missed by Focus receives a false negative
# RL reward.  Batch size 8 with two crops per case gives the paper's intended
# GRL group size N=16.  Focus is frozen exactly in both variants.  Keep the
# proxy-negative dose at the champion's 100x: these 38 training masks are tiny
# rather than truly empty, so the earlier 300x dose is a known confound.  Both
# arms use the identical audited multi-lesion source prior; loss weighting is
# the only controlled difference.
wait_pid="${WAIT_PID:-838761}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
champion_checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
long_summary="public_sweeps/long_focus_dual_20260820/summary_by_combined.tsv"
wise_summary="public_sweeps/long_focus_wise_dual_20260820/summary_by_combined.tsv"
long_selection="public_sweeps/long_focus_dual_20260820/robust_selection_prompt_confirmed.json"
wise_selection="public_sweeps/long_focus_wise_dual_20260820/robust_selection_prompt_confirmed.json"
checkpoint="${champion_checkpoint}"
checkpoint_source="champion"
wise_decision=""
long_decision=""
if [[ -s "${wise_selection}" ]]; then
  wise_decision=$(awk -F'"' '/"decision":/ {print $4}' "${wise_selection}")
fi
if [[ -s "${long_selection}" ]]; then
  long_decision=$(awk -F'"' '/"decision":/ {print $4}' "${long_selection}")
fi
if [[ "${wise_decision}" == "PROMOTE" && -s "${wise_summary}" ]]; then
  selected_wise_checkpoint=$(awk -F $'\t' 'NR == 2 {print $3}' "${wise_summary}")
  if [[ -f "${selected_wise_checkpoint}" ]]; then
    checkpoint="${selected_wise_checkpoint}"
    checkpoint_source="long_focus_wise"
  fi
elif [[ "${long_decision}" == "PROMOTE" && -s "${long_summary}" ]]; then
  selected_long_checkpoint=$(awk -F $'\t' 'NR == 2 {print $3}' "${long_summary}")
  if [[ -f "${selected_long_checkpoint}" ]]; then
    checkpoint="${selected_long_checkpoint}"
    checkpoint_source="long_focus"
  fi
fi
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/gf_glance_objective_dual_20260820}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
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
stored_epoch=$("${python_bin}" -c \
  'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["epoch"]))' \
  "${checkpoint}")
start_epoch=$((stored_epoch + 1))
max_epochs=$((start_epoch + 5))
mkdir -p "${root}" "${cache_root}/tmp" ../../submissions
printf 'checkpoint_source\tcheckpoint\tstored_epoch\tstart_epoch\tmax_epochs\n%s\t%s\t%s\t%s\t%s\n' \
  "${checkpoint_source}" "${checkpoint}" "${stored_epoch}" "${start_epoch}" "${max_epochs}" \
  > "${root}/initialization.tsv"
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wait_pair() {
  local left="$1" right="$2" status=0
  wait "${left}" || status=$?
  wait "${right}" || status=$?
  return "${status}"
}

train_variant() {
  local gpu="$1"
  local name="$2"
  local classification_weight="$3"
  local rl_weight="$4"
  local run_dir="runs/${name}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override "${start_epoch}" \
    --reset_optimizer --reset_best_acc --freeze_focus \
    --partial_label_teacher --glance_teacher_labels \
    --cancer_positive_weight 1.0 --cancer_negative_weight 0.25 \
    --glance_pseudo_positive_threshold 0.95 \
    --glance_pseudo_negative_threshold 0.50 \
    --glance_pseudo_positive_min_voxels 64 \
    --glance_teacher_batch_size 2 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs "${max_epochs}" --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --glance_lr 1e-5 --warmup_epochs 1 \
    --cohort_priority_json configs/multilesion_source_priority_20260820.json \
    --glance_classification_weight "${classification_weight}" \
    --glance_rl_weight "${rl_weight}" \
    --wandb_project pancancer-ct --wandb_name "${name}" \
    > "${run_dir}/train.log" 2>&1
}

control_name="gf_glance_teacher_safe_strongrl_postfocus_c01_rl10_n16_lr1e5_5ep_20260820"
teacher_name="gf_glance_teacher_safe_conservative_postfocus_c10_rl01_n16_lr1e5_5ep_20260820"
training_complete=1
for completed_name in "${control_name}" "${teacher_name}"; do
  [[ $(find "runs/${completed_name}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' | wc -l) -ge 5 \
     && -f "runs/${completed_name}/model_final.pt" ]] || training_complete=0
done
if [[ "${training_complete}" -eq 1 ]]; then
  echo "Both teacher-safe Glance runs are complete; resuming evaluation."
else
  train_variant 0 "${control_name}" 0.1 1.0 & train_pid0=$!
  train_variant 1 "${teacher_name}" 1.0 0.1 & train_pid1=$!
  wait_pair "${train_pid0}" "${train_pid1}"
fi

evaluate_variant() {
  local gpu="$1"
  local name="$2"
  local output="${root}/${name}"
  mkdir -p "${output}"
  : > "${output}/candidates.tsv"
  declare -A seen=()
  # The prior Glance run's internal-best checkpoint was its worst public
  # checkpoint, so retain first/middle/final temporal coverage rather than
  # trusting internal Dice. This removes four redundant 50-volume passes per
  # arm because every selected checkpoint is still tested at both gate values.
  mapfile -t available < <(find "runs/${name}" -maxdepth 1 \
    -name 'checkpoint_epoch_*.pt' -printf '%p\n' | sort)
  [[ "${#available[@]}" -ge 5 ]] || {
    echo "${name}: expected at least five checkpoints" >&2
    return 1
  }
  for position in 0 2 4; do
    candidate="${available[$position]}"
    [[ -f "${candidate}" ]] || continue
    local digest
    digest=$("${python_bin}" checkpoint_state_digest.py "${candidate}")
    [[ -n "${seen[${digest}]:-}" ]] && continue
    seen["${digest}"]=1
    local tag
    tag=$(basename "${candidate}" .pt)
    for gate in 0.35 0.50; do
      local gate_tag="g${gate/./p}"
      local destination="${output}/${tag}_${gate_tag}"
      mkdir -p "${destination}"
      if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
          --test_data_path "${public_images}" --save_prediction_path "${destination}" \
          --trained_pth "${candidate}" --a_min -175 --a_max 250 \
          --glance_threshold "${gate}" --lesion_thresholds 0.50,0.60 \
          --focus_batch_size 16 --workers 2 --skip_existing \
          > "${destination}.inference.log" 2>&1; then
        echo "Focus batch 16 failed; resuming completed cases at batch 8" \
          >> "${destination}.inference.log"
        if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
            --test_data_path "${public_images}" --save_prediction_path "${destination}" \
            --trained_pth "${candidate}" --a_min -175 --a_max 250 \
            --glance_threshold "${gate}" --lesion_thresholds 0.50,0.60 \
            --focus_batch_size 8 --workers 2 --skip_existing \
            >> "${destination}.inference.log" 2>&1; then
          echo "Focus batch 8 failed; resuming completed cases at batch 2" \
            >> "${destination}.inference.log"
          CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
            --test_data_path "${public_images}" --save_prediction_path "${destination}" \
            --trained_pth "${candidate}" --a_min -175 --a_max 250 \
            --glance_threshold "${gate}" --lesion_thresholds 0.50,0.60 \
            --focus_batch_size 2 --workers 2 --skip_existing \
            >> "${destination}.inference.log" 2>&1
        fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${tag}" "${candidate}" "${gate}" "${destination}" \
        >> "${output}/candidates.tsv"
    done
  done
}

evaluate_variant 0 "${control_name}" &
eval_pid0=$!
evaluate_variant 1 "${teacher_name}" &
eval_pid1=$!
wait_pair "${eval_pid0}" "${eval_pid1}"

printf 'variant\tcheckpoint_tag\tcheckpoint\tglance_threshold\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\tglance_threshold\n' \
  > "${manifest}"
for name in "${control_name}" "${teacher_name}"; do
  while IFS=$'\t' read -r variant tag candidate_checkpoint gate destination; do
    for threshold in 0.50 0.60; do
      threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
      source="${destination}/${threshold_dir}"
      count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
      if [[ "${count}" -ne 50 ]]; then
        echo "${source}: expected 50 predictions, found ${count}" >&2
        exit 1
      fi
      for min_mm3 in 0 1000; do
        score_tag="${variant}_${tag}_g${gate/./p}_${threshold_dir}_mm${min_mm3}"
        score_stem="${root}/scored/${score_tag}"
        candidate="${source}"
        mkdir -p "$(dirname "${score_stem}")"
        if [[ "${min_mm3}" -gt 0 ]]; then
          candidate="${score_stem}"
          mkdir -p "${candidate}"
          "${python_bin}" -u fuse_and_filter_masks.py \
            --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
            --output "${candidate}" > "${score_stem}.filter.log" 2>&1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s.metrics.csv\t%s\n' \
          "${variant}" "${tag}" "${candidate_checkpoint}" "${threshold}" \
          "${min_mm3}" "${score_tag}" "${candidate}" "${score_stem}" "${gate}" \
          >> "${manifest}"
      done
    done
  done < "${root}/${name}/candidates.tsv"
done

grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ score_tag predictions _ _; do
  grid_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${manifest}")
"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1
while IFS=$'\t' read -r variant tag candidate_checkpoint threshold min_mm3 _ _ metrics gate; do
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${tag}" "${candidate_checkpoint}" "${gate}" "${threshold}" \
    "${min_mm3}" "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k9,9gr
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
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min _ \
  selected_predictions selected_metrics best_gate public_prompt prompt_metrics \
  prompt_robust prompt_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best_line}"
candidate_tag="${best_variant}_${best_tag}_g${best_gate/./p}_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_glance_released --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Reweighted Glance candidate did not clear the raw paired gate; retaining fallback status."
fi
cp "${prompt_robust}" "${root}/robust_selection_prompt_confirmed.json"
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "Reweighted Glance candidate did not clear the deployed paired gate; packaging as an explicitly ranked fallback."
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${best_gate}" --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
wait_pair "${hidden_pid0}" "${hidden_pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${count}" -ne 100 ]]; then
  echo "Expected 100 hidden predictions, found ${count}" >&2
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
submission="../../submissions/gf_${candidate_tag}_prompt_or_e050.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tglance_objective\t%s\t%s\t%s\n' \
  "${candidate_tag}" "$(realpath "${prompt_metrics}")" \
  "$(realpath "${submission}")" \
  "$(realpath "${root}/robust_selection_prompt_confirmed.json")" \
  > "${root}/submission_candidate.tsv"
echo "Reweighted Glance submission ready: ${submission}"

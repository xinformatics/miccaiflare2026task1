#!/usr/bin/env bash
set -euo pipefail

# Two controlled Focus continuations, one per GPU. Both expand the trusted
# target box from 16 to 24 mm; only the Tversky precision/recall balance differs.
wait_pid="${WAIT_PID:-827637}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/gf_context24_dual_20260820}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
if [[ ! -f "${checkpoint}" || ! -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]]; then
  echo "Champion checkpoint or preprocessing completion marker is missing" >&2
  exit 1
fi
mkdir -p "${root}" "${cache_root}/tmp" ../../submissions
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_variant() {
  local gpu="$1"
  local name="$2"
  local alpha="$3"
  local beta="$4"
  local run_dir="runs/${name}"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_glance \
    --partial_label_teacher --bbox_partial_label --bbox_context_mm 24 \
    --pseudo_positive_threshold 0.95 --teacher_consistency_weight 0.15 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 1 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --optim_lr 1e-5 --warmup_epochs 1 \
    --boundary_weight 0.15 --tversky_alpha "${alpha}" --tversky_beta "${beta}" \
    --wandb_project pancancer-ct --wandb_name "${name}" \
    > "${run_dir}/train.log" 2>&1
}

recall_name="gf_bbox_ep280_context24_a03b07_lr1e5_5ep_20260820"
balanced_name="gf_bbox_ep280_context24_a05b05_lr1e5_5ep_20260820"
train_variant 0 "${recall_name}" 0.3 0.7 &
train_pid0=$!
train_variant 1 "${balanced_name}" 0.5 0.5 &
train_pid1=$!
wait "${train_pid0}"
wait "${train_pid1}"

evaluate_variant() {
  local gpu="$1"
  local name="$2"
  local output="${root}/${name}"
  mkdir -p "${output}"
  : > "${output}/checkpoints.tsv"
  declare -A seen=()
  # Public and internal ranking diverged sharply in the Glance audit. Score
  # every short-run epoch here as well. Hash only state_dict tensors so
  # optimizer metadata in model.pt cannot trigger duplicate inference.
  for candidate in "runs/${name}"/checkpoint_epoch_028{1..5}.pt "runs/${name}/model.pt"; do
    [[ -f "${candidate}" ]] || continue
    local digest
    digest=$("${python_bin}" checkpoint_state_digest.py "${candidate}")
    [[ -n "${seen[${digest}]:-}" ]] && continue
    seen["${digest}"]=1
    local tag
    tag=$(basename "${candidate}" .pt)
    local destination="${output}/${tag}"
    mkdir -p "${destination}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
      --test_data_path "${public_images}" --save_prediction_path "${destination}" \
      --trained_pth "${candidate}" --a_min -175 --a_max 250 \
      --glance_threshold 0.5 --lesion_thresholds 0.50,0.60 \
      --focus_batch_size 2 --workers 2 > "${destination}.inference.log" 2>&1
    printf '%s\t%s\t%s\n' "${name}" "${tag}" "${candidate}" >> "${output}/checkpoints.tsv"
  done
}

evaluate_variant 0 "${recall_name}" &
eval_pid0=$!
evaluate_variant 1 "${balanced_name}" &
eval_pid1=$!
wait "${eval_pid0}"
wait "${eval_pid1}"

printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
for name in "${recall_name}" "${balanced_name}"; do
  while IFS=$'\t' read -r variant tag candidate_checkpoint; do
    for threshold in 0.50 0.60; do
      threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
      source="${root}/${name}/${tag}/${threshold_dir}"
      count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
      if [[ "${count}" -ne 50 ]]; then
        echo "${name}/${tag}/${threshold_dir}: expected 50 predictions, found ${count}" >&2
        exit 1
      fi
      for min_mm3 in 0 1000; do
        score_stem="${root}/scored/${name}_${tag}_${threshold_dir}_mm${min_mm3}"
        candidate="${source}"
        mkdir -p "$(dirname "${score_stem}")"
        if [[ "${min_mm3}" -gt 0 ]]; then
          candidate="${score_stem}"
          mkdir -p "${candidate}"
          "${python_bin}" -u fuse_and_filter_masks.py \
            --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
            --output "${candidate}" > "${score_stem}.filter.log" 2>&1
        fi
        "${python_bin}" -u evaluate_public_parallel.py \
          --pred "${candidate}" --gt "${public_labels}" \
          --csv "${score_stem}.metrics.csv" --workers 4 > "${score_stem}.score.log" 2>&1
        dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
        nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
        combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${variant}" "${tag}" "${candidate_checkpoint}" "${threshold}" "${min_mm3}" \
          "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
      done
    done
  done < "${root}/${name}/checkpoints.tsv"
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k8,8gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min best_dsc best_nsd best_combined \
  <<< "${best_line}"
focus_dir="lesion_$(printf '%.2f' "${best_focus}")"
selected_metrics="${root}/scored/${best_variant}_${best_tag}_${focus_dir}_mm${best_min}.metrics.csv"
candidate_tag="${best_variant}_${best_tag}_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_context16 --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Context-24 candidate did not clear the paired robustness gate."
  exit 0
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" "${hidden_root}/filtered"
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
submission="../../submissions/gf_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/filtered" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Context-24 submission ready: ${submission}"

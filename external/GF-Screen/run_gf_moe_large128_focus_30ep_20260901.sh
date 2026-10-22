#!/usr/bin/env bash
set -euo pipefail

# Train the second GF Focus expert from the exact shared-Focus checkpoint used
# by the deployed winner.  The general expert and Glance are immutable.  This
# specialist sees 128x128x64 crops, prioritizes the disjoint Task-2 LCT cohort,
# and uses a recall-weighted, partial-label-safe component-box objective.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
anchor="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
run_dir="${root}/runs/gf_moe_large128_focus_30ep_20260901"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
component_cache="${cache_root}/component_uniform_lct_20260830/focus"
analysis="${root}/analysis/gf_moe_large128_focus_30ep_20260901"
lock="${root}/runs/run_gf_moe_large128_focus_30ep_20260901.lock"

mkdir -p "${run_dir}" "${analysis}" "${cache_root}/tmp"
exec 9>"${lock}"
flock -n 9 || { echo "GF large-context Focus specialist already active" >&2; exit 1; }
[[ -s "${anchor}" ]] || { echo "missing deployed GF Focus anchor: ${anchor}" >&2; exit 2; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=60
export WANDB_HTTP_TIMEOUT=30
export OMP_NUM_THREADS=4

cd "${root}"
if [[ ! -s "${run_dir}/checkpoint_epoch_0030.pt" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 "${python_bin}" -u GF_RL_train.py --distributed \
    --dist-url tcp://127.0.0.1:23661 \
    --checkpoint "${anchor}" --partial_label_teacher_checkpoint "${anchor}" \
    --start_epoch_override 0 --reset_optimizer --reset_scheduler --reset_best_acc \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --bbox_context_mm 16 --component_boxes --component_uniform_crop \
    --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
    --component_crop_cache_dir "${component_cache}" \
    --cancer_positive_weight 1 --cancer_negative_weight 0 \
    --teacher_consistency_weight 0.15 --boundary_weight 0.10 \
    --tversky_alpha 0.25 --tversky_beta 0.75 \
    --data_dir . --abdomen_json jsons/lct_task2_20260830/abdomen.json \
    --chest_json jsons/lct_task2_20260830/chest.json \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --cohort_priority_json configs/lct_task2_priority04_20260830.json \
    --roi_x 128 --roi_y 128 --roi_z 64 \
    --max_epochs 30 --val_every 1000 --skip_final_validation \
    --checkpoint_every 2 --batch_size 2 --sw_batch_size 1 \
    --validation_focus_batch_size 1 --samples_per_epoch 8192 --workers 0 \
    --optim_lr 6e-5 --warmup_epochs 3 --lrschedule warmup_cosine \
    --seed 20260901 --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct \
    --wandb_name gf_moe_large128_focus_30ep_20260901 \
    >> "${run_dir}/train.log" 2>&1
fi

for roi in 128 96; do
  screen="${analysis}/component_screen_roi${roi}.json"
  if [[ ! -s "${screen}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u evaluate_focus_component_crops.py \
      --checkpoint "anchor=${anchor}" \
      --checkpoint "ep10=${run_dir}/checkpoint_epoch_0010.pt" \
      --checkpoint "ep20=${run_dir}/checkpoint_epoch_0020.pt" \
      --checkpoint "ep30=${run_dir}/checkpoint_epoch_0030.pt" \
      --output "${screen}" --gpu 0 --batch-size 2 \
      --cases-per-group 8 --components-per-case 3 \
      --roi-x "${roi}" --roi-y "${roi}" --roi-z 64 \
      >> "${analysis}/component_screen_roi${roi}.log" 2>&1
  fi
done

printf '%s training and internal specialist screens completed\n' "$(date -Ins)" \
  > "${analysis}/TRAINING_COMPLETE.txt"

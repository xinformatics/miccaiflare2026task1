#!/usr/bin/env bash
set -euo pipefail

# Start the GPU-1 box-localization arm while the final public-only interpolation
# runs on GPU 0. The queued dual-arm controller will wait for GPU 1 and then
# skip this arm once its final checkpoint is present.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/challenge_train_labeled_all_partial_20260822.json"
empty_healthy="${root}/jsons/local/no_proxy_healthy.json"
component_cache="${root}/cache/component_uniform_v3_bg4096"
arm="gf_glance_boxloc_w015_allpartial_ep280_3ep_20260822"
run_dir="${root}/runs/${arm}"

mkdir -p "${run_dir}" "${cache_root}/tmp" "${component_cache}"
exec 9>"${root}/runs/run_glance_boxloc_high_gpu1_early_20260822.lock"
flock -n 9 || { echo "early GPU-1 box-localization arm already active" >&2; exit 1; }
[[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]] && exit 0

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${root}"
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u GF_RL_train.py \
  --checkpoint "${checkpoint}" --start_epoch_override 280 \
  --reset_optimizer --reset_best_acc --freeze_focus \
  --partial_label_teacher --glance_teacher_labels --hybrid_cohort_glance_labels \
  --complete_label_cohorts_json "${complete_manifest}" \
  --glance_rl_weight 0 --glance_classification_weight 1 \
  --glance_box_localization --glance_box_localization_weight 0.15 \
  --glance_pseudo_positive_threshold 0.95 --glance_pseudo_negative_threshold 0.05 \
  --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
  --cancer_positive_weight 4 --cancer_negative_weight 1 \
  --data_dir . --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json --healthy_json "${empty_healthy}" \
  --healthy_repeat 1 --cache_dir "${cache_root}" --logdir "${run_dir}" \
  --max_epochs 283 --val_every 3 --batch_size 8 --sw_batch_size 2 \
  --samples_per_epoch 20000 --workers 1 --glance_lr 3e-5 --warmup_epochs 1 \
  --component_uniform_crop \
  --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
  --component_background_points 4096 --component_crop_cache_dir "${component_cache}" \
  --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
  --enable_scale_intensity_aug --enable_contrast_aug \
  --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
  --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
  --harmonize_cached_chest_window --seed 20260822 \
  --wandb_project pancancer-ct --wandb_name "${arm}" \
  > "${run_dir}/train.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

# Continue the locked Focus-only LCT recipe for five more epochs. Training uses
# both GPUs at batch 4/rank (global batch 8). Epoch 170 ended the preceding
# cosine cycle at zero LR, so this launch preserves AdamW state but starts one
# fresh five-epoch warmup/cosine cycle. Checkpoints are screened afterward.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
parent="${root}/runs/gf_wu_full_17k_lr3e4_20260812/checkpoint_epoch_0160.pt"
run_dir="${root}/runs/gf_lct947_focus_e160_to170_20260830"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
component_cache="${cache_root}/component_uniform_lct_20260830/focus"
work="${root}/analysis/lct_task2_finetune_20260830"
lock="${root}/runs/run_lct_focus_ddp_171_to175_20260831.lock"

mkdir -p "${work}" "${cache_root}/tmp"
exec 9>"${lock}"
flock -n 9 || { echo "two-GPU Focus continuation already active" >&2; exit 1; }
[[ -s "${run_dir}/checkpoint_epoch_0170.pt" ]] || {
  echo "completed epoch-170 checkpoint is required" >&2; exit 2;
}

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=60
export WANDB_HTTP_TIMEOUT=30
export OMP_NUM_THREADS=4

cd "${root}"
if [[ ! -s "${run_dir}/checkpoint_epoch_0175.pt" ]]; then
  "${python_bin}" -u GF_RL_train.py --distributed \
    --dist-url tcp://127.0.0.1:23641 \
    --checkpoint "${run_dir}/checkpoint_epoch_0170.pt" --reset_scheduler \
    --partial_label_teacher_checkpoint "${parent}" \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --bbox_context_mm 16 --component_boxes --component_uniform_crop \
    --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
    --component_crop_cache_dir "${component_cache}" \
    --cancer_positive_weight 1 --cancer_negative_weight 0 \
    --teacher_consistency_weight 0.15 --boundary_weight 0.15 \
    --tversky_alpha 0.3 --tversky_beta 0.7 \
    --data_dir . --abdomen_json jsons/lct_task2_20260830/abdomen.json \
    --chest_json jsons/lct_task2_20260830/chest.json \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --cohort_priority_json configs/lct_task2_priority04_20260830.json \
    --max_epochs 175 --val_every 1000 --skip_final_validation \
    --batch_size 4 --sw_batch_size 2 --validation_focus_batch_size 1 \
    --samples_per_epoch 8192 --workers 1 --optim_lr 3e-5 \
    --warmup_epochs 2 --lrschedule warmup_cosine --seed 20260830 \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct \
    --wandb_name gf_lct947_focus_e170_to175_restart_20260831 \
    >> "${run_dir}/train.log" 2>&1
fi

for target_epoch in 171 172 173 174 175; do
  completed="${run_dir}/checkpoint_epoch_0${target_epoch}.pt"
  screen="${work}/focus_component_screen_ep${target_epoch}.json"
  [[ -s "${completed}" ]]
  if [[ ! -s "${screen}" ]]; then
    CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u evaluate_focus_component_crops.py \
      --checkpoint "ep${target_epoch}=${completed}" --output "${screen}" \
      --gpu 0 --batch-size 1 --cases-per-group 8 --components-per-case 3 \
      >> "${work}/focus_component_screen_epochs171_to175.log" 2>&1
  fi
done

printf '%s two-GPU Focus continuation and paired screens completed\n' \
  "$(date -Ins)" > "${work}/FOCUS_DDP_171_TO175_COMPLETE.txt"

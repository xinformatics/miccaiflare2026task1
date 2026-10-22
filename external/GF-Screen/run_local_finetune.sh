#!/usr/bin/env bash
set -euo pipefail

run_name="gf_screen_flare25_partial_safe_$(date +%Y%m%d_%H%M%S)"
run_dir="runs/${run_name}"
mkdir -p "${run_dir}"

# MONAI PersistentDataset stages torch archives through tempfile. /tmp and
# this NVMe are different filesystems, so shutil.move degrades to a visible
# copy and another DDP worker can observe a half-written archive. Keeping the
# temporary directory beside the cache makes the final publication atomic.
mkdir -p cache/tmp
export TMPDIR="$PWD/cache/tmp"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Two 48 GB GPUs: local loader batch 2 x two sampled crops, global effective
# crop batch 8. The conservative LR preserves the strong released solution;
# the gate moves ten times slower than the segmenter to protect healthy-case FP.
python GF_RL_train.py \
  --distributed \
  --checkpoint pretrained/model_final.pt \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir cache/local_1x1x3 \
  --logdir "${run_dir}" \
  --max_epochs 110 \
  --val_every 1 \
  --update_epochs 1 \
  --batch_size 4 \
  --sw_batch_size 3 \
  --samples_per_epoch 2730 \
  --workers 4 \
  --optim_lr 3e-5 \
  --glance_lr 0 \
  --freeze_glance \
  --freeze_focus_encoder \
  --reg_weight 1e-5 \
  --warmup_epochs 1 \
  --wandb_project pancancer-ct \
  --wandb_name "${run_name}" \
  2>&1 | tee "${run_dir}/train.log"

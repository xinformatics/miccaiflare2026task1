#!/usr/bin/env bash
set -euo pipefail

# Two complementary learned adaptations from the exact deployed GF epoch-160
# parent. GPU 0 updates Focus on component-balanced positive lesion crops; GPU
# 1 updates Glance with teacher-safe crop labels. The 947 disjoint LCT cases
# are oversampled fourfold, while the 8,275 Task-2 duplicates are excluded.
# This controller trains only: it cannot read hidden labels, create a ZIP, or
# upload. Public evaluation remains a later serialized gate.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_wu_full_17k_lr3e4_20260812/checkpoint_epoch_0160.pt"
abdomen_json="${root}/jsons/lct_task2_20260830/abdomen.json"
chest_json="${root}/jsons/lct_task2_20260830/chest.json"
audit="${root}/jsons/lct_task2_20260830/AUDIT.json"
priority="${root}/configs/lct_task2_priority04_20260830.json"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
component_cache="${cache_root}/component_uniform_lct_20260830"
focus_name="gf_lct947_focus_e160_to170_20260830"
glance_name="gf_lct947_glance_e160_to170_20260830"
focus_dir="${root}/runs/${focus_name}"
glance_dir="${root}/runs/${glance_name}"
work="${root}/analysis/lct_task2_finetune_20260830"
terminal="${work}/TERMINAL.json"
lock="${root}/runs/run_dual_lct_task2_finetune_20260830.lock"

mkdir -p "${focus_dir}" "${glance_dir}" "${work}" "${component_cache}" \
  "${cache_root}/tmp"
exec 9>"${lock}"
flock -n 9 || { echo "LCT GF fine-tune controller already active" >&2; exit 1; }

for required in "${checkpoint}" "${abdomen_json}" "${chest_json}" \
    "${audit}" "${priority}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${audit}" "${checkpoint}" "${work}/PREFLIGHT.json" <<'PY'
import hashlib,json,os,sys,torch
from pathlib import Path
audit_path,checkpoint,output=map(Path,sys.argv[1:])
audit=json.load(open(audit_path,encoding="utf-8"))
if audit.get("status")!="COMPLETE" or audit.get("staged_case_count")!=947:
    raise SystemExit("LCT staging audit is incomplete")
if audit.get("unique_patients")!=300 or audit.get("excluded_brain_cases")!=3:
    raise SystemExit("LCT patient/exclusion inventory mismatch")
stored=torch.load(checkpoint,map_location="cpu",weights_only=False)
if int(stored.get("epoch",-1))!=159 or len(stored.get("state_dict",{}))!=271:
    raise SystemExit("GF epoch-160 parent identity mismatch")
payload={
 "schema_version":1,"status":"PASS",
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
 "checkpoint_stored_epoch":int(stored["epoch"]),
 "staging_audit":{"path":str(audit_path.resolve()),"inventory_sha256":audit["inventory_sha256"]},
 "staged_cases":audit["staged_case_count"],"unique_patients":audit["unique_patients"],
 "public_labels_used_for_gradient":False,"hidden_read":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

free_nvme2_kb=$(df --output=avail "${root}" | tail -1 | tr -d '[:space:]')
free_nvme3_kb=$(df --output=avail "${cache_root}" | tail -1 | tr -d '[:space:]')
[[ "${free_nvme2_kb}" =~ ^[0-9]+$ && "${free_nvme2_kb}" -ge 83886080 ]] || {
  echo "NVMe2 has less than 80 GiB free" >&2; exit 2;
}
[[ "${free_nvme3_kb}" =~ ^[0-9]+$ && "${free_nvme3_kb}" -ge 157286400 ]] || {
  echo "NVMe3 has less than 150 GiB free for GF cache" >&2; exit 2;
}

# Populate only absent deterministic cache keys. Existing Task-1 entries are
# reused; one serialized pass prevents the two training arms racing on writes.
cd "${root}"
"${python_bin}" -u preprocess_offline_cache.py \
  --cache-dir "${cache_root}" --workers 2 --abdomen-workers 2 \
  --chest-workers 2 --healthy-workers 0 \
  --abdomen-json "${abdomen_json}" --chest-json "${chest_json}" \
  --healthy-json jsons/local/no_proxy_healthy.json \
  > "${work}/preprocess.log" 2>&1

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=60
export WANDB_HTTP_TIMEOUT=30

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 2048 && "${used1}" -le 2048 ]]; then break; fi
  printf '%s waiting for idle GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

train_focus() {
  [[ -s "${focus_dir}/checkpoint_epoch_0170.pt" ]] && return 0
  local training_checkpoint="${checkpoint}"
  local resume_args=(--start_epoch_override 160 --reset_optimizer --reset_best_acc)
  local latest
  latest=$(find "${focus_dir}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort | tail -1)
  if [[ -n "${latest}" ]]; then
    training_checkpoint="${latest}"
    resume_args=()
    echo "resuming LCT Focus arm from ${latest}"
  fi
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${training_checkpoint}" "${resume_args[@]}" \
    --partial_label_teacher_checkpoint "${checkpoint}" \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --bbox_context_mm 16 --component_boxes --component_uniform_crop \
    --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
    --component_crop_cache_dir "${component_cache}/focus" \
    --cancer_positive_weight 1 --cancer_negative_weight 0 \
    --teacher_consistency_weight 0.15 --boundary_weight 0.15 \
    --tversky_alpha 0.3 --tversky_beta 0.7 \
    --data_dir . --abdomen_json "${abdomen_json}" --chest_json "${chest_json}" \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${focus_dir}" \
    --cohort_priority_json "${priority}" \
    --max_epochs 170 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --validation_focus_batch_size 1 \
    --samples_per_epoch 8192 --workers 1 --optim_lr 3e-5 \
    --warmup_epochs 2 --lrschedule warmup_cosine --seed 20260830 \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct --wandb_name "${focus_name}" \
    >> "${focus_dir}/train.log" 2>&1
}

train_glance() {
  [[ -s "${glance_dir}/checkpoint_epoch_0170.pt" ]] && return 0
  local training_checkpoint="${checkpoint}"
  local resume_args=(--start_epoch_override 160 --reset_optimizer --reset_best_acc)
  local latest
  latest=$(find "${glance_dir}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort | tail -1)
  if [[ -n "${latest}" ]]; then
    training_checkpoint="${latest}"
    resume_args=()
    echo "resuming LCT Glance arm from ${latest}"
  fi
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${training_checkpoint}" "${resume_args[@]}" \
    --freeze_focus --partial_label_teacher --glance_teacher_labels \
    --glance_classification_weight 0.1 --glance_rl_weight 1.0 \
    --glance_pseudo_positive_threshold 0.98 \
    --glance_pseudo_negative_threshold 0.02 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 2 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --component_boxes --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 --component_background_points 4096 \
    --component_crop_cache_dir "${component_cache}/glance" \
    --data_dir . --abdomen_json "${abdomen_json}" --chest_json "${chest_json}" \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${glance_dir}" \
    --cohort_priority_json "${priority}" \
    --max_epochs 170 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --validation_focus_batch_size 1 \
    --samples_per_epoch 8192 --workers 1 --glance_lr 3e-5 \
    --warmup_epochs 2 --lrschedule warmup_cosine --seed 20260830 \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct --wandb_name "${glance_name}" \
    >> "${glance_dir}/train.log" 2>&1
}

train_focus & focus_pid=$!
train_glance & glance_pid=$!
status=0
wait "${focus_pid}" || status=$?
wait "${glance_pid}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
[[ -s "${focus_dir}/checkpoint_epoch_0170.pt" ]]
[[ -s "${glance_dir}/checkpoint_epoch_0170.pt" ]]

"${python_bin}" - "${work}/PREFLIGHT.json" \
  "${focus_dir}/checkpoint_epoch_0170.pt" \
  "${glance_dir}/checkpoint_epoch_0170.pt" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
preflight,focus,glance,output=map(Path,sys.argv[1:])
asset=lambda p:{"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"AWAIT_SERIALIZED_FIXED_PUBLIC_EVALUATION",
 "preflight":asset(preflight),"focus_checkpoint":asset(focus),"glance_checkpoint":asset(glance),
 "public_labels_used_for_gradient":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

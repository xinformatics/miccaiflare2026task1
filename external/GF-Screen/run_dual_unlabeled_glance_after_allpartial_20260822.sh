#!/usr/bin/env bash
set -euo pipefail

# Champion-anchored positive-unlabeled Glance adaptation. This controller is
# serialized behind all-partial Focus saving, produces nested high-confidence
# teacher pseudo labels, and trains two Glance-only arms. No upload is possible.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${UNLABELED_CHECKPOINT:-${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt}"
unlabeled="${project}/train_unlabel"
empty_healthy="${root}/jsons/local/no_proxy_healthy.json"
contract="${root}/configs/challenge_train_labeled_all_partial_20260822.json"
upstream="${UNLABELED_UPSTREAM:-${root}/public_sweeps/gf_focus_allpartial_saved_20260822/SAVE_ONLY_COMPLETE.txt}"
work="${UNLABELED_WORK:-${root}/public_sweeps/gf_unlabeled_teacher_20260822}"
inference="${work}/teacher_nested"
balanced_manifest="${work}/balanced/manifest.json"
conservative_manifest="${work}/conservative/manifest.json"
arm_balanced="${UNLABELED_ARM_BALANCED:-gf_glance_ul_bal_ep280_2ep_20260822}"
arm_conservative="${UNLABELED_ARM_CONSERVATIVE:-gf_glance_ul_con_ep280_2ep_20260822}"
deployed="${UNLABELED_DEPLOYED:-${root}/public_sweeps/gf_glance_unlabeled_deployed_20260822}"
start_epoch="${UNLABELED_START_EPOCH:-280}"
max_epochs="${UNLABELED_MAX_EPOCHS:-282}"
eval_epochs="${UNLABELED_EVAL_EPOCHS:-281 282}"
teacher_workers="${UNLABELED_TEACHER_WORKERS:-1}"
teacher_focus_batch_size="${UNLABELED_TEACHER_FOCUS_BATCH_SIZE:-8}"
pseudo_workers="${UNLABELED_PSEUDO_WORKERS:-4}"

for value_name in teacher_workers teacher_focus_batch_size pseudo_workers; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${value_name} must be a positive integer, got ${value}" >&2; exit 2;
  }
done

for required in "${checkpoint}" "${empty_healthy}" "${contract}" \
    "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${unlabeled}" -type f -name '*.nii.gz' | wc -l) -eq 2856 ]] || {
  echo "Train-Unlabel inventory is not exactly 2856 files" >&2; exit 2;
}
mkdir -p "${work}" "${inference}" "${root}/runs/${arm_balanced}" \
  "${root}/runs/${arm_conservative}" "${cache_root}/tmp"
exec 9>"${root}/runs/run_dual_unlabeled_glance_after_allpartial_20260822.lock"
flock -n 9 || { echo "unlabeled Glance experiment already active" >&2; exit 1; }
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for all-partial Focus save-only completion\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" "${root}/audit_train_unlabel.py" \
  --unlabeled "${unlabeled}" \
  --labeled "${project}/train_label/imagesTr_part1" \
  --labeled "${project}/train_label/imagesTr_part2" \
  --validation "${project}/validation/Validation-Public-Images" \
  --validation "${project}/validation/Validation-Hidden-Images" \
  --output "${work}/train_unlabel_audit.json" \
  > "${work}/train_unlabel_audit.log"

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
gpu0_limit_mib="${UNLABELED_GPU0_MAX_MIB:-4096}"
gpu1_limit_mib="${UNLABELED_GPU1_MAX_MIB:-4096}"
[[ "${gpu0_limit_mib}" =~ ^[0-9]+$ && "${gpu1_limit_mib}" =~ ^[0-9]+$ ]] || {
  echo "invalid Train-Unlabel GPU admission limit" >&2; exit 2;
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${gpu0_limit_mib}" \
      && "${used1}" -le "${gpu1_limit_mib}" ]]; then break; fi
  printf '%s waiting for Train-Unlabel GPUs: gpu0=%s/%s gpu1=%s/%s MiB\n' \
    "$(date -Ins)" "${used0}" "${gpu0_limit_mib}" \
    "${used1}" "${gpu1_limit_mib}"
  sleep 30
done

infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${unlabeled}" --recursive_inputs \
    --save_prediction_path "${inference}" --trained_pth "${checkpoint}" \
    --num_shards 2 --shard_id "${shard}" --workers "${teacher_workers}" \
    --focus_batch_size "${teacher_focus_batch_size}" \
    --a_min -175 --a_max 250 --glance_threshold 0.20 \
    --lesion_thresholds 0.85,0.90,0.95,0.97 --skip_existing \
    > "${work}/teacher_shard${shard}.log" 2>&1
}
infer_shard 0 0 & pid0=$!
infer_shard 1 1 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
for threshold in 0.85 0.90 0.95 0.97; do
  [[ $(find "${inference}/lesion_${threshold}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 2856 ]] || {
    echo "incomplete teacher threshold ${threshold}" >&2; exit 2;
  }
done

"${python_bin}" "${root}/build_unlabeled_hysteresis_pseudolabels.py" \
  --images "${unlabeled}" --support "${inference}/lesion_0.85" \
  --seed "${inference}/lesion_0.95" \
  --output-labels "${work}/balanced/labels" \
  --output-manifest "${balanced_manifest}" \
  --output-audit "${work}/balanced/audit.json" \
  --exclude-audit "${work}/train_unlabel_audit.json" \
  --min-seed-mm3 25 --min-component-mm3 100 --recipe bal085s095 \
  --workers "${pseudo_workers}" \
  > "${work}/balanced.log"
"${python_bin}" "${root}/build_unlabeled_hysteresis_pseudolabels.py" \
  --images "${unlabeled}" --support "${inference}/lesion_0.90" \
  --seed "${inference}/lesion_0.97" \
  --output-labels "${work}/conservative/labels" \
  --output-manifest "${conservative_manifest}" \
  --output-audit "${work}/conservative/audit.json" \
  --exclude-audit "${work}/train_unlabel_audit.json" \
  --min-seed-mm3 50 --min-component-mm3 100 --recipe con090s097 \
  --workers "${pseudo_workers}" \
  > "${work}/conservative.log"

# The pseudo labels are a reusable data product.  Callers can stop here when
# experiment policy requires a longer or materially different continuation
# instead of the legacy two-epoch Glance screen.
if [[ "${UNLABELED_GENERATE_ONLY:-0}" == 1 ]]; then
  "${python_bin}" - "${work}/train_unlabel_audit.json" \
    "${work}/balanced/audit.json" "${work}/conservative/audit.json" \
    "${work}/PSEUDO_LABELS_COMPLETE.json" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path

inputs = list(map(Path, sys.argv[1:-1]))
output = Path(sys.argv[-1])
for path in inputs:
    if not path.is_file():
        raise FileNotFoundError(path)

def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "decision": "PSEUDO_LABELS_READY_FOR_LONG_TRAINING",
    "train_unlabel_audit": identity(inputs[0]),
    "balanced_pseudo_audit": identity(inputs[1]),
    "conservative_pseudo_audit": identity(inputs[2]),
    "teacher_empty_policy": "unknown_not_negative",
    "training_started": False,
    "public_read": False,
    "hidden_read": False,
    "submission_created": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
  exit 0
fi

pseudo_repeat() {
  "${python_bin}" -c 'import json,math,sys;n=len(json.load(open(sys.argv[1]))["training"]);assert n>0;print(max(1,min(8,math.ceil(4000/n))))' "$1"
}
repeat_balanced=$(pseudo_repeat "${balanced_manifest}")
repeat_conservative=$(pseudo_repeat "${conservative_manifest}")
component_cache="${root}/cache/component_uniform_unlabeled_20260822"
mkdir -p "${component_cache}"

train_arm() {
  local gpu="$1" arm="$2" manifest="$3" repeat="$4" run_dir="${root}/runs/$2"
  local final_checkpoint
  final_checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${max_epochs}")
  [[ -s "${final_checkpoint}" && -s "${run_dir}/model_final.pt" ]] && return
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override "${start_epoch}" \
    --reset_optimizer --reset_best_acc --freeze_focus \
    --partial_label_teacher --glance_teacher_labels --hybrid_cohort_glance_labels \
    --complete_label_cohorts_json "${contract}" \
    --glance_rl_weight 0 --glance_classification_weight 1 \
    --glance_box_localization --glance_box_localization_weight 0.05 \
    --glance_pseudo_positive_threshold 0.95 --glance_pseudo_negative_threshold 0.05 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --pseudo_abdomen_json "${manifest}" --pseudo_abdomen_repeat "${repeat}" \
    --chest_json jsons/local/chest.json --healthy_json "${empty_healthy}" \
    --healthy_repeat 1 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs "${max_epochs}" --val_every 2 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 22000 --workers 1 --glance_lr 1e-5 --warmup_epochs 1 \
    --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 --component_background_points 4096 \
    --component_crop_cache_dir "${component_cache}" \
    --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --harmonize_cached_chest_window --seed 20260822 \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}
train_arm 0 "${arm_balanced}" "${balanced_manifest}" "${repeat_balanced}" & train0=$!
train_arm 1 "${arm_conservative}" "${conservative_manifest}" "${repeat_conservative}" & train1=$!
status=0
wait "${train0}" || status=$?
wait "${train1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_balanced}" ARM_B="${arm_conservative}" EVAL_EPOCHS="${eval_epochs}" \
  GLANCE_THRESHOLDS="0.20 0.35 0.50" DEPLOYED_OUTPUT="${deployed}" \
bash "${root}/evaluate_dual_glance_box_localization_public_20260822.sh"
if [[ "${UNLABELED_SKIP_LEGACY_SAVE:-0}" != 1 ]]; then
  bash "${root}/save_unlabeled_glance_candidate_20260822.sh"
fi

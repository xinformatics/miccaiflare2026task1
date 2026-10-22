#!/usr/bin/env bash
set -euo pipefail

# Single materially longer Focus experiment.  It starts only when the direct
# no-TTA trained-model route fails the fixed-public gate.  Challenge labeled
# masks remain partial; Train-Unlabel masks are positive-only teacher targets.
# The immutable epoch-283 Focus supplies teacher logits throughout resumes.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
component_cache="${root}/cache/component_uniform_pu_long_20260825"
source_checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
pseudo_root="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
pseudo_terminal="${pseudo_root}/PSEUDO_LABELS_COMPLETE.json"
pseudo_manifest="${pseudo_root}/balanced/manifest.json"
pseudo_audit="${pseudo_root}/balanced/audit.json"
pseudo_cache_contract="${pseudo_root}/PSEUDO_CACHE_DATASET_CONTRACT.json"
pseudo_cache_terminal="${pseudo_root}/PSEUDO_CACHE_COMPLETE.json"
focus_terminal="${root}/public_sweeps/focus_encoder_notta_direct_20260825/TERMINAL.json"
focus_route="${root}/public_sweeps/focus_encoder_notta_direct_20260825/ROUTE.json"
focus_calibration="${root}/public_sweeps/focus_encoder_notta_direct_20260825/PREDECLARED_CALIBRATION.json"
focus_summary="${root}/public_sweeps/focus_encoder_notta_direct_20260825/summary_by_dsc.tsv"
run_name="gf_focus_pu_long30_e283_to313_20260825"
run_dir="${root}/runs/${run_name}"
work="${root}/public_sweeps/${run_name}"
recipe="${work}/TRAINING_RECIPE.json"
dataset_contract="${work}/PSEUDO_DATASET_CONTRACT.json"
terminal="${work}/TERMINAL.json"
public_output="${root}/public_sweeps/gf_focus_pu_long30_trajectory_20260825"
lock="${root}/runs/run_long_focus_pu_after_direct_public_20260825.lock"

mkdir -p "${run_dir}" "${work}" "${cache_root}/tmp" "${component_cache}" \
  "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long positive-unlabeled Focus controller already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" \
    "${source_checkpoint}" "${recipe}" "${run_dir}/checkpoint_epoch_0313.pt" \
    "${public_output}/TERMINAL.json" "${terminal}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
status, decision, reason = sys.argv[1:4]
source, recipe, checkpoint, public, output = map(Path, sys.argv[4:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path):
    return {"path": str(path.resolve()), "sha256": sha(path)} if path.is_file() else None
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": status,
    "decision": decision,
    "reason": reason,
    "source_checkpoint": asset(source),
    "training_recipe": asset(recipe),
    "final_checkpoint": asset(checkpoint),
    "public_terminal": asset(public),
    "challenge_labeled_contract": "all_partial_primary_lesion_only",
    "train_unlabel_contract": "positive_only_teacher_targets",
    "public_used_for_training": False,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}
failure_line=unknown
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_TOUCHCOMP \
    "long Focus controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE|SKIPPED) trap - EXIT; exit 0 ;;
    FAILED)
      preserved_terminal="${work}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved_terminal}"
      printf '%s preserved failed controller terminal for exact resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing long Focus terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while [[ ! -s "${pseudo_terminal}" ]]; do
  printf '%s waiting for audited Train-Unlabel pseudo labels\n' "$(date -Ins)"
  sleep 30
done
while [[ ! -s "${pseudo_cache_terminal}" ]]; do
  printf '%s waiting for deterministic Train-Unlabel pseudo cache\n' "$(date -Ins)"
  sleep 30
done
while [[ ! -s "${focus_terminal}" ]]; do
  printf '%s waiting for direct no-TTA fixed-public decision\n' "$(date -Ins)"
  sleep 30
done

for file in "${focus_route}" "${focus_calibration}" "${focus_summary}"; do
  [[ -s "${file}" ]] || { echo "missing direct Focus public asset: ${file}" >&2; exit 2; }
done
focus_decision=$("${python_bin}" "${root}/validate_direct_focus_public_terminal.py" \
  --terminal "${focus_terminal}" --route "${focus_route}" \
  --calibration "${focus_calibration}" --summary "${focus_summary}")
if [[ "${focus_decision}" == QUALIFY_SUBMISSION_PREPARATION ]]; then
  write_terminal SKIPPED QUALIFIED_ROUTE_NEEDS_PREDICTION \
    "the higher-priority direct Focus route passed its fixed-public gate; preserve GPUs for its hidden prediction"
  trap - EXIT
  exit 0
fi
[[ "${focus_decision}" == KEEP_TOUCHCOMP ]] || {
  echo "unexpected direct Focus decision: ${focus_decision}" >&2; exit 2;
}

for file in "${source_checkpoint}" "${pseudo_manifest}" "${pseudo_audit}" \
    "${pseudo_cache_contract}" "${pseudo_cache_terminal}" \
    "${root}/configs/challenge_train_labeled_all_partial_20260822.json" \
    "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  [[ -s "${file}" ]] || { echo "missing long-training asset: ${file}" >&2; exit 2; }
done

"${python_bin}" "${root}/validate_positive_unlabeled_manifest.py" \
  --pseudo-terminal "${pseudo_terminal}" --manifest "${pseudo_manifest}" \
  --pseudo-audit "${pseudo_audit}" --output "${dataset_contract}" \
  > "${work}/pseudo_dataset_contract.log"
"${python_bin}" "${root}/preprocess_positive_unlabeled_cache.py" \
  --cache-dir "${cache_root}" --manifest "${pseudo_manifest}" \
  --dataset-contract "${pseudo_cache_contract}" \
  --output "${pseudo_cache_terminal}" --verify-only \
  > "${work}/pseudo_cache_contract.log"

read -r pseudo_cases pseudo_repeat < <(PYTHONPATH="${root}" "${python_bin}" - \
  "${pseudo_terminal}" "${pseudo_manifest}" "${pseudo_audit}" \
  "${dataset_contract}" "${recipe}" "${source_checkpoint}" "${run_dir}" \
  "${run_name}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
from pseudo_exposure import exposure_fraction, select_integer_repeat
pseudo_terminal, manifest_path, audit_path, contract_path, output, checkpoint, run_dir = map(Path, sys.argv[1:8])
run_name = sys.argv[8]
terminal = json.load(open(pseudo_terminal, encoding="utf-8"))
manifest = json.load(open(manifest_path, encoding="utf-8"))
audit = json.load(open(audit_path, encoding="utf-8"))
contract = json.load(open(contract_path, encoding="utf-8"))
if contract.get("status") != "PASS":
    raise SystemExit("positive-unlabeled dataset contract did not pass")
training = manifest.get("training")
if int(contract.get("usable_images", -1)) != 2850 or int(contract.get("excluded_images", -1)) != 6:
    raise SystemExit("positive-unlabeled dataset inventory mismatch")
if int(contract.get("positive_cases", -1)) != len(training):
    raise SystemExit("validated positive-case count mismatch")
labeled_cases = 11283 + 6162
target_fraction = 0.20
repeat = select_integer_repeat(
    len(training), labeled_cases, target_fraction, minimum=1, maximum=8
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "experiment": "single long positive-unlabeled Focus continuation",
    "source_checkpoint": str(checkpoint.resolve()),
    "source_checkpoint_sha256": sha(checkpoint),
    "frozen_teacher_checkpoint": str(checkpoint.resolve()),
    "pseudo_manifest": str(manifest_path.resolve()),
    "pseudo_manifest_sha256": sha(manifest_path),
    "pseudo_audit": str(audit_path.resolve()),
    "pseudo_audit_sha256": sha(audit_path),
    "pseudo_dataset_contract": str(contract_path.resolve()),
    "pseudo_dataset_contract_sha256": sha(contract_path),
    "pseudo_positive_cases": len(training),
    "pseudo_repeat": repeat,
    "pseudo_target_fraction": target_fraction,
    "pseudo_repeat_selection": "bounded_integer_minimum_absolute_fraction_error_tie_lower",
    "expected_pseudo_case_fraction": exposure_fraction(
        len(training), labeled_cases, repeat
    ),
    "challenge_labeled_cases": labeled_cases,
    "challenge_labeled_contract": "all masks partial; primary lesion only",
    "pseudo_contract": "foreground trusted; uncertain background ignored; frozen-teacher-safe background only",
    "start_epoch": 283,
    "max_epochs": 313,
    "new_epochs": 30,
    "samples_per_epoch_global": 8192,
    "batch_size_per_gpu": 4,
    "gpus": 2,
    "focus_lr": 3e-5,
    "scheduler": "3-epoch linear warmup plus cosine decay",
    "bbox_context_mm": 16,
    "component_uniform_crop": True,
    "component_boxes": True,
    "component_crop_size_power": 0.5,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "boundary_weight": 0.15,
    "teacher_consistency_weight": 0.15,
    "teacher_positive_threshold": 0.98,
    "teacher_negative_threshold": 0.02,
    "bbox_pseudo_positive_weight": 0.10,
    "empty_crop_pseudo_positive_weight": 0.05,
    "cancer_crop_positive_negative_ratio": "4:1",
    "healthy_proxy_cases": 0,
    "wandb_project": "pancancer-ct",
    "wandb_name": run_name,
    "wandb_run_id": hashlib.sha256(
        f"{run_dir.resolve()}::{run_name}".encode("utf-8")
    ).hexdigest()[:16],
    "wandb_resume": "allow",
    "wandb_failure_policy": "disable_tracking_and_continue_training",
    "public_used_for_training": False,
    "hidden_read": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(len(training), repeat)
PY
)
echo "positive-only pseudo cases=${pseudo_cases}, exposure repeat=${pseudo_repeat}"

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_INIT_TIMEOUT=60
export WANDB_HTTP_TIMEOUT=30

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 1024 ]]; then break; fi
  printf '%s waiting for both training GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

training_checkpoint="${source_checkpoint}"
resume_args=(--start_epoch_override 283 --reset_optimizer --reset_best_acc)
if [[ ! -s "${run_dir}/checkpoint_epoch_0313.pt" ]]; then
  latest=$(find "${run_dir}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort | tail -1)
  if [[ -n "${latest}" ]]; then
    training_checkpoint="${latest}"
    resume_args=()
    echo "resuming interrupted long Focus run from ${latest} with immutable epoch-283 teacher"
  fi
fi

if [[ ! -s "${run_dir}/checkpoint_epoch_0313.pt" ]]; then
  cd "${root}"
  CUDA_VISIBLE_DEVICES=0,1 "${python_bin}" -u GF_RL_train.py \
    --distributed --checkpoint "${training_checkpoint}" "${resume_args[@]}" \
    --partial_label_teacher_checkpoint "${source_checkpoint}" \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --bbox_context_mm 16 --pseudo_positive_threshold 0.98 \
    --pseudo_negative_threshold 0.02 --bbox_pseudo_positive_weight 0.10 \
    --bbox_empty_pseudo_positive_weight 0.05 \
    --bbox_empty_pseudo_positive_min_voxels 64 \
    --teacher_consistency_weight 0.15 --boundary_weight 0.15 \
    --tversky_alpha 0.3 --tversky_beta 0.7 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --component_boxes --component_uniform_crop \
    --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
    --component_background_points 4096 \
    --component_crop_cache_dir "${component_cache}" \
    --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --pseudo_abdomen_json "${pseudo_manifest}" \
    --pseudo_abdomen_repeat "${pseudo_repeat}" \
    --chest_json jsons/local/chest.json \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 313 --val_every 5 --batch_size 4 --sw_batch_size 2 \
    --samples_per_epoch 8192 --workers 2 --optim_lr 3e-5 \
    --warmup_epochs 3 --lrschedule warmup_cosine --seed 20260825 \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct --wandb_name "${run_name}" \
    >> "${run_dir}/train.log" 2>&1
fi
# The per-epoch checkpoint is written after every completed optimizer epoch and
# is the exact asset consumed by the public evaluator.  model_final.pt is a
# redundant validation snapshot; an interruption between those two writes must
# not strand an otherwise complete epoch-313 run.
[[ -s "${run_dir}/checkpoint_epoch_0313.pt" ]]

LONG_FOCUS_RUN_DIR="${run_dir}" \
LONG_FOCUS_TRAJECTORY_OUTPUT="${public_output}" \
bash "${root}/evaluate_long_focus_pu_trajectory_public_20260825.sh" \
  >> "${work}/public_controller.log" 2>&1
[[ -s "${public_output}/TERMINAL.json" ]]
public_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["decision"])' \
  "${public_output}/TERMINAL.json")
write_terminal COMPLETE "${public_decision}" \
  "30-epoch positive-unlabeled Focus training and predeclared fixed-public trajectory screen complete"
trap - EXIT
cat "${public_output}/summary.tsv"

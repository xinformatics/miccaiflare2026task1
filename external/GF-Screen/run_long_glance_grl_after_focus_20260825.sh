#!/usr/bin/env bash
set -euo pipefail

# One genuinely long, paper-aligned Glance continuation. It is deliberately
# downstream of the active Focus-first chain and its save-only consumers. All
# challenge masks remain partial; Train-Unlabel masks are positive-only. Focus
# is frozen, the GRL group is global N=16 under two-GPU DDP, and only the fixed
# public 50 can promote a checkpoint. This controller cannot read hidden data,
# package predictions, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
fallback_checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
pseudo_root="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
pseudo_terminal="${pseudo_root}/PSEUDO_LABELS_COMPLETE.json"
pseudo_manifest="${pseudo_root}/balanced/manifest.json"
pseudo_audit="${pseudo_root}/balanced/audit.json"
pseudo_cache_contract="${pseudo_root}/PSEUDO_CACHE_DATASET_CONTRACT.json"
pseudo_cache_terminal="${pseudo_root}/PSEUDO_CACHE_COMPLETE.json"
long_controller="${root}/public_sweeps/gf_focus_pu_long30_e283_to313_20260825/TERMINAL.json"
long_public="${root}/public_sweeps/gf_focus_pu_long30_trajectory_20260825/TERMINAL.json"
direct_saved="${root}/public_sweeps/focus_encoder_notta_saved_20260825/TERMINAL.json"
long_saved="${root}/public_sweeps/gf_focus_pu_long30_trajectory_saved_20260825/TERMINAL.json"
# The corrected 107M ResEnc campaign is the current GPU priority. Keep this
# older Glance continuation wait-only until the staged base, positive-only,
# conservative organ-target, and dynamic dual-window ResEnc branches have all
# finished their public-only selection controllers. The downstream controllers
# would otherwise become eligible together and race for both idle GPUs.
resenc_priority_public="${root}/../../nnunet/public_sweeps/resenc_s6_dualwindow_endpoints_20260827/TERMINAL.json"
resenc_priority_saved="${root}/../../nnunet/public_sweeps/resenc_s6_dualwindow_saved_20260827/TERMINAL.json"
run_name="gf_glance_grl_long30_20260825"
run_dir="${root}/runs/${run_name}"
work="${root}/public_sweeps/${run_name}"
source_contract="${work}/SOURCE.json"
recipe="${work}/TRAINING_RECIPE.json"
terminal="${work}/TERMINAL.json"
public_output="${root}/public_sweeps/gf_glance_grl_long30_trajectory_20260825"
component_cache="${root}/cache/component_uniform_glance_grl_long_20260825"
lock="${root}/runs/run_long_glance_grl_after_focus_20260825.lock"

mkdir -p "${run_dir}" "${work}" "${public_output}" "${component_cache}" \
  "${cache_root}/tmp" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long strong-GRL Glance controller already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" \
    "${source_contract}" "${recipe}" "${public_output}/TERMINAL.json" \
    "${terminal}" <<'PY'
import datetime,hashlib,json,os,re,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
source,recipe,public,output=map(Path,sys.argv[4:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
source_payload=json.load(open(source,encoding="utf-8")) if source.is_file() else {}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "source":asset(source),"training_recipe":asset(recipe),"public_terminal":asset(public),
 "challenge_labeled_contract":"all_partial_primary_lesion_only",
 "train_unlabel_contract":"positive_only_teacher_targets",
 "public_used_for_training":bool(source_payload.get("public_used_for_training",False)),
 "public_labels_used_for_gradient":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_TOUCHCOMP \
    "long Glance controller failed at line ${line}, status ${status}"
  exit "${status}"
}
failure_line=unknown
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
      printf '%s preserved failed long Glance terminal for exact resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing long Glance terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

for marker in "${pseudo_terminal}" "${pseudo_cache_terminal}" \
    "${long_controller}" "${direct_saved}" "${long_saved}" \
    "${resenc_priority_public}" "${resenc_priority_saved}"; do
  while :; do
    if [[ ! -s "${marker}" ]]; then
      printf '%s waiting for upstream terminal %s\n' "$(date -Ins)" "${marker}"
      sleep 30
      continue
    fi
    marker_status=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
      "${marker}")
    case "${marker_status}" in
      COMPLETE|SKIPPED) break ;;
      FAILED)
        printf '%s waiting for replacement of failed upstream terminal %s\n' \
          "$(date -Ins)" "${marker}"
        sleep 30
        ;;
      *) echo "unexpected upstream terminal status ${marker_status}: ${marker}" >&2; exit 2 ;;
    esac
  done
done

for file in "${fallback_checkpoint}" "${pseudo_manifest}" "${pseudo_audit}" \
    "${pseudo_cache_contract}" "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  [[ -s "${file}" ]] || { echo "missing long-Glance asset: ${file}" >&2; exit 2; }
done
"${python_bin}" "${root}/preprocess_positive_unlabeled_cache.py" \
  --cache-dir "${cache_root}" --manifest "${pseudo_manifest}" \
  --dataset-contract "${pseudo_cache_contract}" \
  --output "${pseudo_cache_terminal}" --verify-only \
  > "${work}/pseudo_cache_contract.log"

if [[ ! -s "${source_contract}" ]]; then
  "${python_bin}" - "${fallback_checkpoint}" "${long_controller}" \
    "${long_public}" "${direct_saved}" "${long_saved}" \
    "${source_contract}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
fallback,long_controller,long_public,direct_saved,long_saved,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)}
for name,path in (("direct saver",direct_saved),("long saver",long_saved)):
    payload=json.load(open(path,encoding="utf-8"))
    if payload.get("status") not in {"COMPLETE","SKIPPED"}:
        raise SystemExit(f"{name} did not finish safely: {payload.get('status')}")
controller=json.load(open(long_controller,encoding="utf-8"))
if controller.get("status") not in {"COMPLETE","SKIPPED"}:
    raise SystemExit("conditional long Focus controller is not terminal")
checkpoint=fallback
source_kind="immutable_epoch283_fallback"
public_selected=False
upstream_public=None
if controller.get("decision")=="QUALIFY_SUBMISSION_PREPARATION":
    upstream_public=json.load(open(long_public,encoding="utf-8"))
    identity=controller.get("public_terminal",{})
    if Path(identity.get("path","")).resolve()!=long_public.resolve() or identity.get("sha256")!=sha(long_public):
        raise SystemExit("long Focus controller/public identity mismatch")
    if upstream_public.get("status")!="COMPLETE" or upstream_public.get("decision")!="QUALIFY_SUBMISSION_PREPARATION":
        raise SystemExit("long Focus public state did not qualify")
    identity=upstream_public.get("checkpoint",{})
    checkpoint=Path(identity.get("path",""))
    if not checkpoint.is_file() or identity.get("sha256")!=sha(checkpoint):
        raise SystemExit("qualified long Focus checkpoint identity mismatch")
    source_kind="fixed_public_qualified_long_focus"
    public_selected=True
elif controller.get("decision") not in {"KEEP_TOUCHCOMP","QUALIFIED_ROUTE_NEEDS_PREDICTION"}:
    raise SystemExit(f"unexpected long Focus decision: {controller.get('decision')}")
import torch
stored=torch.load(checkpoint,map_location="cpu",weights_only=False)
stored_epoch=int(stored.get("epoch",-1))
match=re.fullmatch(r"checkpoint_epoch_(\d+)\.pt",checkpoint.name)
if match is None or stored_epoch+1!=int(match.group(1)):
    raise SystemExit("source checkpoint filename/stored epoch mismatch")
epoch=stored_epoch+1
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"PASS","source_kind":source_kind,"source_epoch":epoch,
 "source_checkpoint":asset(checkpoint),"long_focus_controller":asset(long_controller),
 "long_focus_public":asset(long_public) if long_public.is_file() else None,
 "direct_save_terminal":asset(direct_saved),"long_save_terminal":asset(long_saved),
 "challenge_labeled_contract":"all_partial_primary_lesion_only",
 "complete_label_cohorts_used":False,
 "public_used_for_source_selection":public_selected,
 "public_used_for_training":public_selected,
 "public_labels_used_for_gradient":False,
 "hidden_labels_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
fi

read -r source_epoch source_checkpoint < <("${python_bin}" - "${source_contract}" <<'PY'
import hashlib,json,sys
from pathlib import Path
payload=json.load(open(sys.argv[1],encoding="utf-8"))
if payload.get("status")!="PASS": raise SystemExit("source contract failed")
identity=payload.get("source_checkpoint",{});checkpoint=Path(identity.get("path",""))
if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=identity.get("sha256"):
    raise SystemExit("source checkpoint identity mismatch")
print(int(payload["source_epoch"]),checkpoint)
PY
)
max_epochs=$((source_epoch + 30))
checkpoint_final=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${max_epochs}")

read -r pseudo_cases pseudo_repeat < <(PYTHONPATH="${root}" "${python_bin}" - \
  "${pseudo_manifest}" "${pseudo_terminal}" "${source_contract}" "${recipe}" \
  "${run_name}" "${source_epoch}" "${max_epochs}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
from pseudo_exposure import exposure_fraction,select_integer_repeat
manifest_path,pseudo_terminal,source_path,output=map(Path,sys.argv[1:5])
run_name=sys.argv[5];source_epoch=int(sys.argv[6]);max_epochs=int(sys.argv[7])
manifest=json.load(open(manifest_path,encoding="utf-8"));training=manifest.get("training",[])
if not training: raise SystemExit("pseudo manifest contains no positive cases")
labeled=11283+6162;target=0.20
repeat=select_integer_repeat(len(training),labeled,target,minimum=1,maximum=8)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source=json.load(open(source_path,encoding="utf-8"))
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "experiment":"single long teacher-safe strong-GRL Glance continuation",
 "source":source,"pseudo_manifest":{"path":str(manifest_path.resolve()),"sha256":sha(manifest_path)},
 "pseudo_terminal":{"path":str(pseudo_terminal.resolve()),"sha256":sha(pseudo_terminal)},
 "pseudo_positive_cases":len(training),"pseudo_repeat":repeat,
 "pseudo_target_fraction":target,
 "pseudo_repeat_selection":"bounded_integer_minimum_absolute_fraction_error_tie_lower",
 "expected_pseudo_case_fraction":exposure_fraction(len(training),labeled,repeat),
 "challenge_labeled_cases":labeled,"challenge_labeled_contract":"all masks partial; primary lesion only",
 "start_epoch":source_epoch,"max_epochs":max_epochs,"new_epochs":30,
 "samples_per_epoch_global":8192,"batch_size_per_gpu":4,"crops_per_case":2,
 "gpus":2,"global_GRL_group_size":16,"global_group_advantages":True,
 "focus_frozen":True,"glance_lr":3e-5,"scheduler":"3-epoch linear warmup plus cosine decay",
 "glance_classification_weight":0.1,"glance_rl_weight":1.0,"policy_update_epochs":1,
 "teacher_positive_threshold":0.98,"teacher_negative_threshold":0.02,
 "teacher_positive_min_voxels":64,"cancer_crop_positive_negative_ratio":"4:1",
 "component_uniform_crop":True,"component_crop_size_power":0.5,
 "healthy_proxy_cases":0,"complete_label_cohorts_used":False,
 "predeclared_checkpoint_offsets":[10,20,30],
 "predeclared_glance_thresholds":[0.10,0.20,0.35],
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "wandb_project":"pancancer-ct","wandb_name":run_name,
 "public_used_for_training":bool(source.get("public_used_for_training",False)),
 "public_labels_used_for_gradient":False,"hidden_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(len(training),repeat)
PY
)
echo "long Glance source epoch=${source_epoch}; pseudo cases=${pseudo_cases} x ${pseudo_repeat}"

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
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for both long-Glance GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

training_checkpoint="${source_checkpoint}"
resume_args=(--start_epoch_override "${source_epoch}" --reset_optimizer --reset_best_acc)
if [[ ! -s "${checkpoint_final}" ]]; then
  latest=$(find "${run_dir}" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort | tail -1)
  if [[ -n "${latest}" ]]; then
    training_checkpoint="${latest}"
    resume_args=()
    echo "resuming interrupted long Glance run from ${latest}"
  fi
fi

if [[ ! -s "${checkpoint_final}" ]]; then
  cd "${root}"
  CUDA_VISIBLE_DEVICES=0,1 "${python_bin}" -u GF_RL_train.py \
    --distributed --checkpoint "${training_checkpoint}" "${resume_args[@]}" \
    --freeze_focus --partial_label_teacher --glance_teacher_labels \
    --global_group_advantages --update_epochs 1 \
    --glance_classification_weight 0.1 --glance_rl_weight 1.0 \
    --glance_pseudo_positive_threshold 0.98 \
    --glance_pseudo_negative_threshold 0.02 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 2 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --component_boxes --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 --component_background_points 4096 \
    --component_crop_cache_dir "${component_cache}" \
    --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --pseudo_abdomen_json "${pseudo_manifest}" --pseudo_abdomen_repeat "${pseudo_repeat}" \
    --chest_json jsons/local/chest.json \
    --healthy_json jsons/local/no_proxy_healthy.json --healthy_repeat 1 \
    --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs "${max_epochs}" --val_every 5 --batch_size 4 --sw_batch_size 2 \
    --samples_per_epoch 8192 --workers 2 --glance_lr 3e-5 \
    --warmup_epochs 3 --lrschedule warmup_cosine --seed 20260825 \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --wandb_project pancancer-ct --wandb_name "${run_name}" \
    >> "${run_dir}/train.log" 2>&1
fi
[[ -s "${checkpoint_final}" ]]

LONG_GLANCE_RUN_DIR="${run_dir}" LONG_GLANCE_SOURCE="${source_contract}" \
LONG_GLANCE_TRAJECTORY_OUTPUT="${public_output}" \
bash "${root}/evaluate_long_glance_grl_trajectory_public_20260825.sh" \
  >> "${work}/public_controller.log" 2>&1
[[ -s "${public_output}/TERMINAL.json" ]]
public_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["decision"])' \
  "${public_output}/TERMINAL.json")
write_terminal COMPLETE "${public_decision}" \
  "30-epoch teacher-safe strong-GRL Glance training and fixed-public trajectory screen complete"
trap - EXIT
cat "${public_output}/summary.tsv"

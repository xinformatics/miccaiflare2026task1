#!/usr/bin/env bash
set -euo pipefail

# Fail-closed staged controller for the prospective 12-mm shape-band branch.
# The authorization and preflight are intentionally unarmed at creation time.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${project_root}/nnunet/run_resenc_s6_partialband12.py"
trainer_source="${project_root}/nnunet/resenc_s6_partialband12.py"
target_source="${project_root}/nnunet/partial_region_targets.py"
parent_selector="${project_root}/nnunet/select_resenc_s6_partialband12_parent.py"
authorization="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_AUTHORIZATION_20260827.json"
preflight="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_TRAINER_PREFLIGHT_20260827.json"
controller_preflight="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_CONTROLLER_PREFLIGHT_20260827.json"
public_policy="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_PUBLIC_SELECTION_POLICY_20260827.json"
base_training="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827/TRAINING_TERMINAL.json"
base_selection="${project_root}/nnunet/public_sweeps/resenc_s6_partialbox_endpoints_20260827/TERMINAL.json"
plans="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json"
dataset="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/dataset.json"
cache="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/nnUNetTrainerResEncS6PanCancerPartialBand12__nnUNetPlans__3d_fullres_ResEncS6"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
work="${project_root}/nnunet/runs/resenc_s6_partialband12_100ep_20260827"
terminal="${work}/TRAINING_TERMINAL.json"
public_root="${project_root}/nnunet/public_sweeps/resenc_s6_partialband12_endpoints_20260827"
public_unit="miccai-resenc-s6-partialband12-public-20260827.service"
stage_end_epochs=(25 50 100)

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "partial-band controller already active" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${work}/.matplotlib"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-pancancer-ct}"
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=0
export nnUNet_compile=false
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
unset RESENC_STAGE_END_EPOCH RESENC_POSITIVE_STAGE_END_EPOCH || true

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"REVIEW_PARTIALBAND12_CONTROLLER",
 "public_labels_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  [[ "${status}" -eq 0 || -s "${terminal}" ]] || write_failure \
    "partial-band controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${runner}" "${trainer_source}" "${target_source}" "${parent_selector}" "${authorization}" \
  "${preflight}" "${controller_preflight}" "${public_policy}" "${base_training}" "${base_selection}" \
  "${plans}" "${dataset}"; do
  [[ -s "${file}" ]] || { echo "missing partial-band input: ${file}" >&2; exit 2; }
done
[[ -d "${cache}" ]]

read -r parent_epoch parent_sha parent < <("${python_bin}" - \
  "${project_root}" "${parent_selector}" "${controller_preflight}" "${authorization}" "${preflight}" "${public_policy}" \
  "${base_training}" "${base_selection}" "${plans}" "${dataset}" "${cache}" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
from nnunet.select_resenc_s6_partialband12_parent import select_parent
root=Path(sys.argv[1]);selector_path=Path(sys.argv[2]);controller_preflight_path=Path(sys.argv[3]);auth_path,preflight_path,policy_path,training_path,selection_path,plans_path,dataset_path,cache=map(Path,sys.argv[4:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
controller_preflight=json.load(open(controller_preflight_path,encoding="utf-8"))
safety=controller_preflight.get("safety",{})
if (controller_preflight.get("status")!="PASS_ARMED"
    or safety.get("current_authorization_satisfies_launch") is not True
    or safety.get("current_trainer_preflight_satisfies_launch") is not True):
    raise SystemExit("partial-band controller preflight is not armed")
runtime_bindings={
    "training controller":controller_preflight.get("controllers",{}).get("training",{}),
    "public controller":controller_preflight.get("controllers",{}).get("public",{}),
    "parent selector":controller_preflight.get("parent_selector",{}),
    "trainer preflight":controller_preflight.get("trainer_preflight",{}),
    "authorization":controller_preflight.get("authorization",{}),
}
for name,item in runtime_bindings.items():
    path=(root/item.get("path","")).resolve()
    if not path.is_file() or sha(path)!=item.get("sha256"):
        raise SystemExit(f"partial-band runtime binding changed: {name}")
if (selector_path.resolve()!=(root/runtime_bindings["parent selector"]["path"]).resolve()
    or auth_path.resolve()!=(root/runtime_bindings["authorization"]["path"]).resolve()
    or preflight_path.resolve()!=(root/runtime_bindings["trainer preflight"]["path"]).resolve()):
    raise SystemExit("partial-band runtime binding path mismatch")
auth=json.load(open(auth_path,encoding="utf-8"))
if (auth.get("status")!="AUTHORIZED_AFTER_BASE_SELECTION"
    or auth.get("decision")!="RUN_ONE_CONTROLLED_PARTIALBAND12_CONTINUATION"
    or auth.get("child",{}).get("epochs")!=100
    or auth.get("child",{}).get("initial_lr")!=1e-4
    or auth.get("child",{}).get("partial_label_policy")!="shape_linf_band_12mm"
    or auth.get("automatic_hidden_inference") or auth.get("automatic_upload")):
    raise SystemExit("partial-band authorization is not armed")
preflight=json.load(open(preflight_path,encoding="utf-8"))
if (preflight.get("status")!="PASS_ARMED"
    or preflight.get("model",{}).get("parameter_count")!=107467050
    or preflight.get("recipe",{}).get("checkpoint_epochs")!=[25,50,100]
    or preflight.get("recipe",{}).get("margin_voxels_zyx")!=[5,8,8]
    or preflight.get("authorization",{}).get("queue_armed") is not True
    or preflight.get("hidden_read") or preflight.get("submission_created")):
    raise SystemExit("partial-band trainer preflight is not armed")
for item in preflight.get("code",{}).values():
    path=root/item["path"]
    if sha(path)!=item["sha256"]:
        raise SystemExit(f"partial-band code hash mismatch: {path}")
policy=json.load(open(policy_path,encoding="utf-8"));gate=policy.get("selection",{}).get("qualification",{})
if (policy.get("model")!="nnUNetTrainerResEncS6PanCancerPartialBand12"
    or policy.get("checkpoint_epochs")!=[25,50,100]
    or gate.get("minimum_fixed_public_DSC")!=0.6366639257651584
    or gate.get("minimum_fixed_public_NSD")!=0.5347520263176905
    or policy.get("threshold_search_after_public_read") is not False):
    raise SystemExit("partial-band public policy changed")
training=json.load(open(training_path,encoding="utf-8"))
if (training.get("status")!="COMPLETE" or training.get("public_read")
    or training.get("hidden_read")):
    raise SystemExit("base training is not safely complete")
expected=(root/"nnUNet_results/Dataset501_PanCancer/nnUNetTrainerResEncS6PanCancerPartialBox__nnUNetPlans__3d_fullres_ResEncS6/fold_all").resolve()
parent_record=select_parent(selection_path,expected)
parent=Path(parent_record["checkpoint"]["path"])
parent_epoch=int(parent_record["epoch"])
parent_sha=parent_record["checkpoint"]["sha256"]
plans=json.load(open(plans_path,encoding="utf-8"));dataset=json.load(open(dataset_path,encoding="utf-8"))
c=plans["configurations"]["3d_fullres_ResEncS6"]
if (c.get("patch_size")!=[128,160,160] or c.get("spacing")!=[2.4,1.6,1.6]
    or dataset.get("numTraining")!=17575):
    raise SystemExit("Dataset501 plan contract changed")
if len(list(cache.glob("*.pkl")))!=17575 or len(list(cache.glob("*.b2nd")))!=35150:
    raise SystemExit("Dataset501 cache inventory changed")
print(parent_epoch,parent_sha,str(parent))
PY
)

export RESENC_BAND12_PARENT_EPOCH="${parent_epoch}"
export RESENC_BAND12_PARENT_SHA256="${parent_sha}"

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${work}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected partial-band terminal status: ${status}" >&2; exit 2 ;;
  esac
fi

if ! systemctl --user is-active --quiet "${public_unit}"; then
  echo "partial-band training requires its public endpoint service" >&2
  exit 2
fi

checkpoint_epoch() {
  local checkpoint="$1"
  "${python_bin}" - "${checkpoint}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPartialBand12"
    or not isinstance(x.get("network_weights"),dict)
    or not isinstance(x.get("optimizer_state"),dict)):
    raise SystemExit("invalid partial-band optimizer checkpoint")
print(int(x.get("current_epoch",-1)))
PY
}

for stage_end in "${stage_end_epochs[@]}"; do
  stage4=$(printf '%04d' "${stage_end}")
  landmark="${fold}/checkpoint_epoch_${stage4}.pth"
  [[ "${stage_end}" -eq 100 ]] && landmark="${final}"
  endpoint_terminal="${public_root}/epoch_${stage4}/ENDPOINT_TERMINAL.json"
  current=0
  resume_checkpoint=""
  for candidate in "${final}" "${latest}"; do
    [[ -s "${candidate}" ]] || continue
    candidate_epoch=$(checkpoint_epoch "${candidate}")
    if (( candidate_epoch > current )); then
      current="${candidate_epoch}";resume_checkpoint="${candidate}"
    fi
  done
  if [[ -z "${resume_checkpoint}" && -e "${fold}" ]]; then
    [[ -z "$(find "${fold}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
      echo "nonempty partial-band fold lacks a resume checkpoint" >&2;exit 2;
    }
  fi
  if (( current < stage_end )); then
    export RESENC_BAND12_STAGE_END_EPOCH="${stage_end}"
    export PANCANCER_AUGMENTATION_SEED_OFFSET="$((parent_epoch * 1000000 + current * 1000000))"
    launch_args=(-pretrained_weights "${parent}")
    [[ -n "${resume_checkpoint}" ]] && launch_args=(--c)
    printf '%s starting partial-band stage %s from child epoch %s, parent epoch %s\n' \
      "$(date -Ins)" "${stage_end}" "${current}" "${parent_epoch}" | tee -a "${work}/stage_controller.log"
    "${python_bin}" -u "${runner}" Dataset501_PanCancer 3d_fullres_ResEncS6 all \
      -tr nnUNetTrainerResEncS6PanCancerPartialBand12 -p nnUNetPlans \
      "${launch_args[@]}" -num_gpus 2 >>"${work}/train.log" 2>&1
  fi
  [[ -s "${landmark}" ]]
  observed=$(checkpoint_epoch "${landmark}")
  [[ "${observed}" -eq "${stage_end}" ]]
  landmark_sha=$(sha256sum "${landmark}" | awk '{print $1}')
  while :; do
    if [[ -s "${endpoint_terminal}" ]] && "${python_bin}" - \
      "${endpoint_terminal}" "${landmark_sha}" "${stage_end}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"));sha=sys.argv[2];epoch=int(sys.argv[3])
if (p.get("status")!="COMPLETE" or p.get("epoch")!=epoch
    or p.get("checkpoint",{}).get("sha256")!=sha or p.get("case_count")!=50
    or p.get("dataset_role")!="public" or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
PY
    then break; fi
    sleep 30
  done
done

unset RESENC_BAND12_STAGE_END_EPOCH PANCANCER_AUGMENTATION_SEED_OFFSET
[[ -s "${final}" ]]
"${python_bin}" - "${final}" "${parent}" "${parent_epoch}" "${parent_sha}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,torch
from pathlib import Path
final,parent=map(Path,sys.argv[1:3]);parent_epoch=int(sys.argv[3]);parent_sha=sys.argv[4];output=Path(sys.argv[5])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();x=torch.load(final,map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPartialBand12"
    or int(x.get("current_epoch",-1))!=100 or not isinstance(x.get("optimizer_state"),dict)
    or sha(parent)!=parent_sha):
    raise SystemExit("invalid partial-band final checkpoint")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"ResEnc S6 partial-band 12-mm",
 "parent":{"path":str(parent.resolve()),"sha256":parent_sha,"epoch":parent_epoch},
 "recipe":{"epochs":100,"initial_lr":1e-4,"global_batch_size":16,
   "partial_label_policy":"shape_linf_band_12mm","endpoint_epochs":[25,50,100]},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "public_selection_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT

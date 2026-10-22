#!/usr/bin/env bash
set -euo pipefail

# Continue the positive-only 107M ResEnc S6 with dynamically derived soft-
# tissue and lung windows. This controller is fail-closed and starts only
# after the conservative organ audit releases both GPUs.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${project_root}/nnunet/run_resenc_s6_dualwindow.py"
trainer_source="${project_root}/nnunet/resenc_s6_dualwindow.py"
authorization="${project_root}/nnunet/RESENC_S6_DUALWINDOW_AUTHORIZATION_20260827.json"
preflight="${project_root}/nnunet/RESENC_S6_DUALWINDOW_PREFLIGHT_20260827.json"
public_policy="${project_root}/nnunet/RESENC_S6_DUALWINDOW_PUBLIC_SELECTION_POLICY_20260827.json"
build_marker="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/DATASET_BUILD_COMPLETE.json"
preprocess_marker="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/PREPROCESSING_COMPLETE.json"
plans="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/nnUNetPlans.json"
dataset="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/dataset.json"
cache="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/nnUNetPlans_3d_fullres_S4D2W32"
parent_terminal="${project_root}/nnunet/runs/resenc_s6_positive_unlabeled_f020_100ep_20260827/TRAINING_TERMINAL.json"
parent="${project_root}/nnUNet_results/Dataset502_PanCancerPositiveUnlabeled/nnUNetTrainerResEncS6PanCancerPositiveUnlabeled__nnUNetPlans__3d_fullres_ResEncS6/fold_all/checkpoint_final.pth"
organ_public="${project_root}/nnunet/public_sweeps/resenc_s6_organ_target_endpoints_20260827/TERMINAL.json"
model="${project_root}/nnUNet_results/Dataset502_PanCancerPositiveUnlabeled/nnUNetTrainerResEncS6PanCancerDualWindow__nnUNetPlans__3d_fullres_ResEncS6"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
work="${project_root}/nnunet/runs/resenc_s6_dualwindow_f020_100ep_20260827"
terminal="${work}/TRAINING_TERMINAL.json"
public_root="${project_root}/nnunet/public_sweeps/resenc_s6_dualwindow_endpoints_20260827"
public_terminal="${public_root}/TERMINAL.json"
public_unit="miccai-resenc-s6-dualwindow-public-endpoints-20260827.service"
stage_end_epochs=(25 50 100)

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "dual-window ResEnc controller already active" >&2; exit 1; }

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

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"REPAIR_THEN_RESUME_EXACT_DUALWINDOW_RESENC",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "dual-window ResEnc failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${runner}" "${trainer_source}" "${authorization}" "${preflight}" \
  "${public_policy}" "${build_marker}" "${preprocess_marker}" "${plans}" \
  "${dataset}" "${parent_terminal}" "${parent}" "${organ_public}"; do
  [[ -s "${file}" ]] || { echo "missing dual-window ResEnc input: ${file}" >&2; exit 2; }
done
[[ -d "${cache}" ]]

"${python_bin}" - "${project_root}" "${authorization}" "${preflight}" \
  "${public_policy}" "${build_marker}" "${preprocess_marker}" "${plans}" \
  "${dataset}" "${cache}" "${parent_terminal}" "${parent}" "${organ_public}" <<'PY'
import hashlib,json,sys,torch
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1])
auth_path,preflight_path,policy_path,build_path,preprocess_path,plans_path,dataset_path,cache,parent_terminal_path,parent_path,organ_public_path=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
auth=json.load(open(auth_path,encoding="utf-8"))
if (auth.get("status")!="AUTHORIZED"
    or auth.get("decision")!="QUEUE_FUNCTION_PRESERVING_DUALWINDOW_RESENC_AFTER_ORGAN_AUDIT"
    or auth.get("child",{}).get("epochs")!=100
    or auth.get("child",{}).get("initial_lr")!=1e-4
    or auth.get("child",{}).get("global_batch_size")!=16
    or auth.get("child",{}).get("architecture_parameters")!=107468778
    or auth.get("supervision",{}).get("pseudo_case_exposure")!=0.2
    or auth.get("initialization",{}).get("stem_channels_1_and_2")!="zero"
    or auth.get("cache_policy")!="derive windows dynamically; do not duplicate Dataset502 preprocessing"
    or auth.get("public_used_for_training") or auth.get("hidden_used_for_training")):
    raise SystemExit("invalid dual-window ResEnc authorization")
preflight=json.load(open(preflight_path,encoding="utf-8"))
if (preflight.get("status")!="PASS" or preflight.get("gpu_used")
    or preflight.get("dataset",{}).get("identifier_count")!=19886
    or preflight.get("dataset",{}).get("identifier_sha256")!="37e3c103543fa4c737694be9a44b80c02aae6c6aa2161622e29bd517a4af728e"
    or preflight.get("tests",{}).get("total_passed")!=14
    or preflight.get("public_read") or preflight.get("hidden_read")):
    raise SystemExit("invalid dual-window ResEnc preflight")
for item in preflight.get("code",{}).values():
    path=root/item["path"]
    if sha(path)!=item["sha256"]:
        raise SystemExit(f"dual-window code hash mismatch: {path}")
policy=json.load(open(policy_path,encoding="utf-8"));qualification=policy.get("selection",{}).get("qualification",{})
if (policy.get("status")!="PREDECLARED_BEFORE_TRAINING"
    or policy.get("checkpoint_epochs")!=[25,50,100]
    or qualification.get("minimum_fixed_public_DSC")!=0.6366639257651584
    or qualification.get("minimum_fixed_public_NSD")!=0.5347520263176905
    or policy.get("threshold_search_after_public_read") is not False
    or policy.get("public_used_for_training") or policy.get("hidden_read")):
    raise SystemExit("invalid dual-window public-selection policy")
build=json.load(open(build_path,encoding="utf-8"))
if (build.get("status")!="COMPLETE" or build.get("labeled_cases")!=17575
    or build.get("positive_unlabeled_cases")!=2311 or build.get("total_cases")!=19886
    or build.get("public_read") or build.get("hidden_read")):
    raise SystemExit("invalid Dataset502 build marker")
preprocess=json.load(open(preprocess_path,encoding="utf-8"))
if (preprocess.get("status")!="COMPLETE" or preprocess.get("total_cases")!=19886
    or preprocess.get("positive_unlabeled_preprocessed_cases")!=2311
    or preprocess.get("build_marker",{}).get("sha256")!=sha(build_path)
    or preprocess.get("public_read") or preprocess.get("hidden_read")):
    raise SystemExit("invalid Dataset502 preprocessing marker")
plans=json.load(open(plans_path,encoding="utf-8"));dataset=json.load(open(dataset_path,encoding="utf-8"))
c=plans["configurations"]["3d_fullres_ResEncS6"]
if (plans.get("dataset_name")!="Dataset502_PanCancerPositiveUnlabeled"
    or c.get("data_identifier")!="nnUNetPlans_3d_fullres_S4D2W32"
    or c.get("patch_size")!=[128,160,160] or c.get("spacing")!=[2.4,1.6,1.6]
    or dataset.get("numTraining")!=19886):
    raise SystemExit("Dataset502 plan contract changed")
data={p.name.removesuffix(".b2nd") for p in cache.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")}
seg={p.name.removesuffix("_seg.b2nd") for p in cache.glob("*_seg.b2nd")}
properties={p.name.removesuffix(".pkl") for p in cache.glob("*.pkl")}
if len(data)!=19886 or data!=seg or data!=properties:
    raise SystemExit("Dataset502 cache triplet inventory mismatch")
pseudo=sorted(name for name in data if name.startswith("Pseudo-"))
if len(pseudo)!=2311 or Counter(name.rsplit("_",1)[0] for name in pseudo)!=Counter({"Pseudo-AMOS-2350":1843,"Pseudo-MSD-506":468}):
    raise SystemExit("Dataset502 pseudo cache inventory mismatch")
parent_terminal=json.load(open(parent_terminal_path,encoding="utf-8"))
if (parent_terminal.get("status")!="COMPLETE"
    or parent_terminal.get("final_checkpoint",{}).get("sha256")!=sha(parent_path)
    or parent_terminal.get("public_read") or parent_terminal.get("hidden_read")):
    raise SystemExit("invalid parent ResEnc terminal")
organ_public=json.load(open(organ_public_path,encoding="utf-8"))
if (organ_public.get("status")!="COMPLETE" or organ_public.get("hidden_read")
    or organ_public.get("submission_created") or organ_public.get("uploaded")):
    raise SystemExit("organ-target public audit has not released the GPU queue")
parent=torch.load(parent_path,map_location="cpu",weights_only=False)
if (parent.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPositiveUnlabeled"
    or int(parent.get("current_epoch",-1))!=100
    or not isinstance(parent.get("network_weights"),dict)):
    raise SystemExit("invalid parent ResEnc checkpoint")
PY

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${work}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected dual-window terminal status: ${status}" >&2; exit 2 ;;
  esac
fi

if ! systemctl --user is-active --quiet "${public_unit}"; then
  echo "staged dual-window ResEnc requires the public endpoint service: ${public_unit}" >&2
  exit 2
fi

checkpoint_epoch() {
  local checkpoint="$1"
  "${python_bin}" - "${checkpoint}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerDualWindow"
    or not isinstance(x.get("network_weights"),dict)
    or not isinstance(x.get("optimizer_state"),dict)):
    raise SystemExit("invalid dual-window optimizer-resume checkpoint")
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
      current="${candidate_epoch}"
      resume_checkpoint="${candidate}"
    fi
  done
  if [[ -z "${resume_checkpoint}" && -e "${fold}" ]]; then
    [[ -z "$(find "${fold}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
      echo "nonempty dual-window fold exists without a resumable checkpoint" >&2
      exit 2
    }
  fi
  if (( current > stage_end )); then
    [[ -s "${landmark}" ]] || {
      echo "missing dual-window endpoint ${stage_end} behind epoch ${current}" >&2
      exit 2
    }
  elif (( current < stage_end )); then
    export RESENC_DUALWINDOW_STAGE_END_EPOCH="${stage_end}"
    export PANCANCER_AUGMENTATION_SEED_OFFSET="$((current * 1000000))"
    launch_args=()
    if [[ -n "${resume_checkpoint}" ]]; then
      launch_args=(--c)
    else
      launch_args=()
    fi
    export RESENC_DUALWINDOW_PARENT_CHECKPOINT="${parent}"
    printf '%s starting dual-window ResEnc stage %s from epoch %s\n' \
      "$(date -Ins)" "${stage_end}" "${current}" | tee -a "${work}/stage_controller.log"
    "${python_bin}" -u "${runner}" Dataset502_PanCancerPositiveUnlabeled \
      3d_fullres_ResEncS6 all \
      -tr nnUNetTrainerResEncS6PanCancerDualWindow -p nnUNetPlans \
      "${launch_args[@]}" -num_gpus 2 >>"${work}/train.log" 2>&1
  fi

  [[ -s "${landmark}" ]]
  observed=$(checkpoint_epoch "${landmark}")
  [[ "${observed}" -eq "${stage_end}" ]] || {
    echo "dual-window endpoint checkpoint mismatch: ${observed} != ${stage_end}" >&2
    exit 2
  }
  landmark_sha=$(sha256sum "${landmark}" | awk '{print $1}')
  printf '%s waiting for hash-bound dual-window public endpoint %s\n' \
    "$(date -Ins)" "${stage_end}" | tee -a "${work}/stage_controller.log"
  while :; do
    if [[ -s "${endpoint_terminal}" ]] && "${python_bin}" - \
      "${endpoint_terminal}" "${landmark_sha}" "${stage_end}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"));sha=sys.argv[2];epoch=int(sys.argv[3])
if (p.get("status")!="COMPLETE" or p.get("epoch")!=epoch
    or p.get("checkpoint",{}).get("sha256")!=sha
    or p.get("case_count")!=50 or p.get("dataset_role")!="public"
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
PY
    then break; fi
    if [[ -s "${public_terminal}" ]]; then
      public_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${public_terminal}")
      [[ "${public_status}" != FAILED ]] || {
        echo "dual-window fixed-public endpoint controller failed" >&2
        exit 2
      }
    fi
    sleep 30
  done
done

unset RESENC_DUALWINDOW_STAGE_END_EPOCH RESENC_DUALWINDOW_PARENT_CHECKPOINT \
  PANCANCER_AUGMENTATION_SEED_OFFSET
[[ -s "${final}" ]]
"${python_bin}" - "${final}" "${parent}" "${plans}" "${dataset}" \
  "${trainer_source}" "${runner}" "${authorization}" "${preflight}" \
  "${public_policy}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,torch
from pathlib import Path
final,parent,plans,dataset,trainer,runner,authorization,preflight,policy,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=torch.load(final,map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerDualWindow"
    or int(x.get("current_epoch",-1))!=100 or not isinstance(x.get("optimizer_state"),dict)):
    raise SystemExit("invalid dual-window final checkpoint")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"nnU-Net v2 ResidualEncoderUNet S6 dynamic dual-window continuation",
 "recipe":{"parameters":107468778,"epochs":100,"initial_lr":1e-4,
   "optimizer":"fresh SGD Nesterov momentum 0.99","schedule":"polynomial",
   "schedule_horizon_epochs":100,"staged_public_endpoints":[25,50,100],
   "optimizer_state_preserved_between_stages":True,
   "augmentation_seed_policy":"base_2026082600_plus_stage_start_epoch_times_1000000_plus_ddp_rank_times_100000",
   "loss":"component-box CE plus Tversky a025 b075; outside boxes ignored",
   "input_views":["nnUNet_normalized_CT","soft_tissue_-175_250_HU","lung_-1000_400_HU"],
   "initialization":"strict parent load; original stem copied to channel 0; extra channels zero",
   "physical_gpus":[0,1],"global_batch_size":16,"labeled_cases":17575,
   "positive_unlabeled_cases":2311,"pseudo_exposure":0.2,
   "empty_unlabeled_cases":"excluded_unknown_not_negative"},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "parent_checkpoint":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "plans":{"path":str(plans.resolve()),"sha256":sha(plans)},
 "dataset":{"path":str(dataset.resolve()),"sha256":sha(dataset)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "authorization":{"path":str(authorization.resolve()),"sha256":sha(authorization)},
 "preflight":{"path":str(preflight.resolve()),"sha256":sha(preflight)},
 "public_policy":{"path":str(policy.resolve()),"sha256":sha(policy)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT

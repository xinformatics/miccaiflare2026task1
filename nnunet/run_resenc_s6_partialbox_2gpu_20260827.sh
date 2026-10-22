#!/usr/bin/env bash
set -euo pipefail

# Long, from-scratch ResEnc S6 run. This controller reads challenge training
# data only. Public/hidden inference is deliberately a separate gated phase.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${project_root}/nnunet/run_resenc_s6_partialbox.py"
trainer_source="${project_root}/nnunet/resenc_s6_partialbox.py"
authorization="${project_root}/nnunet/RESENC_S6_PARTIALBOX_AUTHORIZATION_20260827.json"
preflight="${project_root}/nnunet/RESENC_S6_PARTIALBOX_PREFLIGHT_20260827.json"
realdata_audit="${project_root}/nnunet/RESENC_S6_PARTIALBOX_REALDATA_AUDIT_20260827.json"
public_policy="${project_root}/nnunet/RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
old_stop="${project_root}/nnunet/STOP_RESENC"
plans="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json"
dataset="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/dataset.json"
cache="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/nnUNetTrainerResEncS6PanCancerPartialBox__nnUNetPlans__3d_fullres_ResEncS6"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
work="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827"
terminal="${work}/TRAINING_TERMINAL.json"
public_root="${project_root}/nnunet/public_sweeps/resenc_s6_partialbox_endpoints_20260827"
public_terminal="${public_root}/TERMINAL.json"
public_unit="miccai-resenc-s6-public-endpoints-20260827.service"
stage_end_epochs=(100 200 300 400 500)

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "corrected ResEnc S6 controller already active" >&2; exit 1; }

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
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RETRY_EXACT_RESENC_FROM_LATEST",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "corrected ResEnc S6 failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${runner}" "${trainer_source}" "${authorization}" "${preflight}" \
    "${realdata_audit}" "${public_policy}" "${old_stop}" "${plans}" "${dataset}"; do
  [[ -s "${file}" ]] || { echo "missing corrected ResEnc input: ${file}" >&2; exit 2; }
done
[[ -d "${cache}" ]]

"${python_bin}" - "${project_root}" "${authorization}" "${preflight}" \
  "${realdata_audit}" "${public_policy}" "${old_stop}" "${plans}" \
  "${dataset}" "${cache}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);authorization,preflight_path,audit_path,policy_path,old_stop,plans_path,dataset_path,cache=map(Path,sys.argv[2:])
auth=json.load(open(authorization,encoding="utf-8"))
if (auth.get("status")!="AUTHORIZED"
    or auth.get("decision")!="QUEUE_CORRECTED_RESENC_S6_AFTER_VALID_ACTIVE_CAMPAIGNS"
    or auth.get("parameters")!=107467050
    or auth.get("partial_label_policy")!="ignore_outside_union_of_16mm_component_boxes"
    or auth.get("public_used_for_training") or auth.get("hidden_used_for_training")):
    raise SystemExit("invalid corrected ResEnc authorization")
if "legacy" not in auth.get("old_stop_guard_scope","") or not old_stop.is_file():
    raise SystemExit("legacy ResEnc stop guard must remain preserved")
preflight=json.load(open(preflight_path,encoding="utf-8"))
if (preflight.get("status")!="PASS" or preflight.get("gpu_used")
    or preflight.get("public_read") or preflight.get("hidden_read")
    or preflight.get("architecture",{}).get("parameter_count")!=107467050
    or preflight.get("tests",{}).get("total_passed")!=32):
    raise SystemExit("invalid corrected ResEnc preflight")
for item in preflight.get("code",{}).values():
    path=root/item["path"]
    observed=hashlib.sha256(path.read_bytes()).hexdigest()
    if observed!=item["sha256"]:
        raise SystemExit(f"corrected ResEnc preflight hash mismatch: {path}")
audit=json.load(open(audit_path,encoding="utf-8"));measure=audit.get("measurements",{})
if (audit.get("status")!="PASS"
    or audit.get("decision")!="KEEP_16MM_COMPONENT_BOX_AND_TVERSKY_A025_B075"
    or audit.get("dataset",{}).get("training_cases")!=17575
    or audit.get("sample",{}).get("cases")!=40
    or audit.get("sample",{}).get("sources")!=28
    or measure.get("trusted_negative_to_positive_ratio",{}).get("median")!=24.67456449552064
    or audit.get("public_read") or audit.get("hidden_read")):
    raise SystemExit("invalid corrected ResEnc real-data target audit")
policy=json.load(open(policy_path,encoding="utf-8"));qualification=policy.get("selection",{}).get("qualification",{})
if (policy.get("status")!="PREDECLARED_BEFORE_TRAINING"
    or policy.get("checkpoint_epochs")!=[100,200,300,400,500]
    or len(policy.get("deployments",[]))!=4
    or qualification.get("minimum_fixed_public_DSC")!=0.6366639257651584
    or qualification.get("minimum_fixed_public_NSD")!=0.5347520263176905
    or qualification.get("minimum_DSC_gain_over_incumbent")!=0.015
    or policy.get("threshold_search_after_public_read") is not False
    or policy.get("public_used_for_training") or policy.get("hidden_read")):
    raise SystemExit("invalid corrected ResEnc public-selection policy")
plans=json.load(open(plans_path,encoding="utf-8"));dataset=json.load(open(dataset_path,encoding="utf-8"))
c=plans["configurations"]["3d_fullres_ResEncS6"];a=c["architecture"];k=a["arch_kwargs"]
if (not a["network_class_name"].endswith("ResidualEncoderUNet") or k["n_stages"]!=6
    or k["features_per_stage"]!=[32,64,128,256,320,320]
    or k["n_blocks_per_stage"]!=[1,3,4,6,6,6]
    or c["patch_size"]!=[128,160,160] or c["spacing"]!=[2.4,1.6,1.6]
    or c["data_identifier"]!="nnUNetPlans_3d_fullres_S4D2W32"):
    raise SystemExit("ResEnc S6 plan contract changed")
if dataset.get("numTraining")!=17575:
    raise SystemExit("training-case count changed")
if len(list(cache.glob("*.pkl")))!=17575 or len(list(cache.glob("*.b2nd")))!=35150:
    raise SystemExit("shared preprocessing cache is incomplete")
PY

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${work}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected corrected ResEnc terminal status: ${status}" >&2; exit 2 ;;
  esac
fi

if ! systemctl --user is-active --quiet "${public_unit}"; then
  echo "staged ResEnc requires the fixed-public endpoint service: ${public_unit}" >&2
  exit 2
fi

checkpoint_epoch() {
  local checkpoint="$1"
  "${python_bin}" - "${checkpoint}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPartialBox"
    or not isinstance(x.get("network_weights"),dict)
    or not isinstance(x.get("optimizer_state"),dict)):
    raise SystemExit("invalid corrected ResEnc optimizer-resume checkpoint")
print(int(x.get("current_epoch",-1)))
PY
}

for stage_end in "${stage_end_epochs[@]}"; do
  stage4=$(printf '%04d' "${stage_end}")
  landmark="${fold}/checkpoint_epoch_${stage4}.pth"
  [[ "${stage_end}" -eq 500 ]] && landmark="${final}"
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
      echo "nonempty corrected ResEnc fold exists without a resumable checkpoint" >&2
      exit 2
    }
  fi
  if (( current > stage_end )); then
    [[ -s "${landmark}" ]] || {
      echo "missing immutable endpoint ${stage_end} behind epoch ${current}" >&2
      exit 2
    }
  elif (( current < stage_end )); then
    export RESENC_STAGE_END_EPOCH="${stage_end}"
    export PANCANCER_AUGMENTATION_SEED_OFFSET="$((current * 1000000))"
    launch_args=()
    [[ -n "${resume_checkpoint}" ]] && launch_args=(--c)
    printf '%s starting corrected ResEnc stage %s from epoch %s\n' \
      "$(date -Ins)" "${stage_end}" "${current}" | tee -a "${work}/stage_controller.log"
    "${python_bin}" -u "${runner}" Dataset501_PanCancer 3d_fullres_ResEncS6 all \
      -tr nnUNetTrainerResEncS6PanCancerPartialBox -p nnUNetPlans \
      "${launch_args[@]}" -num_gpus 2 >>"${work}/train.log" 2>&1
  fi

  [[ -s "${landmark}" ]]
  observed=$(checkpoint_epoch "${landmark}")
  [[ "${observed}" -eq "${stage_end}" ]] || {
    echo "endpoint checkpoint epoch mismatch: ${observed} != ${stage_end}" >&2
    exit 2
  }
  landmark_sha=$(sha256sum "${landmark}" | awk '{print $1}')
  printf '%s waiting for hash-bound fixed-public endpoint %s\n' \
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
        echo "fixed-public endpoint controller failed while training was staged" >&2
        exit 2
      }
    fi
    sleep 30
  done
done

unset RESENC_STAGE_END_EPOCH PANCANCER_AUGMENTATION_SEED_OFFSET
[[ -s "${final}" ]]
"${python_bin}" - "${final}" "${plans}" "${dataset}" "${trainer_source}" "${runner}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,torch
from pathlib import Path
final,plans,dataset,trainer,runner,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=torch.load(final,map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPartialBox"
    or int(x.get("current_epoch",-1))!=500 or not isinstance(x.get("optimizer_state"),dict)):
    raise SystemExit("invalid corrected ResEnc final checkpoint")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"nnU-Net v2 ResidualEncoderUNet S6 partial-box",
 "recipe":{"parameters":107467050,"epochs":500,"initial_lr":1e-3,
   "optimizer":"SGD_Nesterov_momentum_0.99","schedule":"polynomial",
   "schedule_horizon_epochs":500,"staged_public_endpoints":[100,200,300,400,500],
   "optimizer_state_preserved_between_stages":True,
   "augmentation_seed_policy":"base_2026082600_plus_stage_start_epoch_times_1000000_plus_ddp_rank_times_100000",
   "loss":"component-box CE plus Tversky a025 b075; outside boxes ignored",
   "physical_gpus":[0,1],"global_batch_size":16,"case_count":17575,"source_count":28,
   "case_sampling":"75% uniform plus 25% size-bin-preserving bounded source target",
   "patch_center_sampling":"75% foreground-voxel uniform plus 25% annotated-component uniform"},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "plans":{"path":str(plans.resolve()),"sha256":sha(plans)},
 "dataset":{"path":str(dataset.resolve()),"sha256":sha(dataset)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT

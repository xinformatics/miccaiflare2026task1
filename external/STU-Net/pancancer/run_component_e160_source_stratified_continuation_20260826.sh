#!/usr/bin/env bash
set -euo pipefail

# Two-GPU, label-safe source/size-stratified continuation from the hidden-leading
# epoch-160 STU-Net-B weights. This script has no public/hidden input or upload.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${root}/run_pancancer_stunet_source_balanced.py"
trainer_source="${root}/STUNetTrainerPanCancerSourceBalanced.py"
audit="${root}/analysis/source_size_stratified_f025_cap3_20260826.json"
loader_smoke="${root}/analysis/component_center_source_loader_smoke_20260826/TERMINAL.json"
loss_audit="${root}/analysis/source_stratified_recall_loss_decision_20260826.json"
component_audit="${root}/analysis/component_center_sampling_headroom_20260826/report.json"
balanced_e40="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826/epoch_0040/TERMINAL.json"
balanced_e40_interim="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826/E40_INTERIM_COMPLETE.json"
parent="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"
parent_sha="81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceStratified__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
run_root="${root}/runs/stunetb_componentbox_e160_sourcesize_f025_cap3_80ep_20260826"
terminal="${run_root}/TRAINING_TERMINAL.json"

mkdir -p "${run_root}"
exec 9>"${run_root}/controller.lock"
flock -n 9 || { echo "source-stratified continuation already active" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA="${SOURCE_STRATIFIED_N_PROC_DA:-0}"
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 nnUNet_compile=false

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RETRY_SOURCE_STRATIFIED_FROM_LATEST",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "source-stratified continuation failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${parent}" "${runner}" "${trainer_source}" "${audit}" "${loader_smoke}" \
  "${loss_audit}" "${balanced_e40}" "${balanced_e40_interim}" "${component_audit}"; do
  [[ -s "${file}" ]] || { echo "missing source-stratified input: ${file}" >&2; exit 2; }
done
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${audit}" "${loader_smoke}" "${loss_audit}" \
  "${trainer_source}" "${balanced_e40}" "${balanced_e40_interim}" "${component_audit}" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit,smoke,loss=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:4])
trainer=Path(sys.argv[4]); balanced=Path(sys.argv[5]); interim=Path(sys.argv[6]); component_audit=Path(sys.argv[7])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
balanced_payload=json.load(open(balanced,encoding="utf-8"))
interim_payload=json.load(open(interim,encoding="utf-8"))
if (audit.get("status")!="analysis_only_not_launched"
    or audit.get("policy")!="size_preserving_bounded_source_mixture"
    or audit.get("case_count")!=17575 or audit.get("source_count")!=28
    or audit.get("source_fraction")!=0.25 or audit.get("raw_per_case_ratio_cap")!=3.0):
    raise SystemExit("invalid source/size-stratified audit")
if (smoke.get("status")!="COMPLETE" or smoke.get("training_case_count")!=17575
    or smoke.get("purpose")!="native_component_center_source_loader_one_batch_no_training"
    or smoke.get("training_loader_class")!="ComponentCenterLoader"
    or smoke.get("validation_loader_class")!="nnUNetDataLoader"
    or smoke.get("component_center_fraction")!=0.25
    or smoke.get("training_probability_sum")!=1.0
    or smoke.get("data_shape")!=[2,1,128,160,160]
    or smoke.get("segmentation_shape")!=[2,1,128,160,160]
    or smoke.get("batch_foreground_voxels",0)<=0
    or smoke.get("trainer_source",{}).get("sha256")!=sha(trainer)
    or smoke.get("gpu_used") or smoke.get("public_read") or smoke.get("hidden_read")
    or smoke.get("submission_created") or smoke.get("uploaded")):
    raise SystemExit("invalid component-center source native-loader smoke")
metric=loss.get("evidence",{}).get("gf_anchor_confirm_either_p70",{})
center=loss.get("component_center_sampling",{})
balanced_metric=balanced_payload.get("metrics",{}).get("gf_anchor_confirm_either_p70",{})
interim_metric=interim_payload.get("metrics",{}).get("gf_anchor_confirm_either_p70",{})
if (loss.get("status")!="COMPLETE"
    or loss.get("decision")!="USE_MILDLY_AGGRESSIVE_RECALL_LOSS_FOR_SOURCE_STRATIFIED_BRANCH"
    or loss.get("loss",{}).get("alpha_false_positive")!=0.25
    or loss.get("loss",{}).get("beta_false_negative")!=0.75
    or metric.get("Lesion_DSC")!=0.6174617981725372
    or metric.get("Lesion_NSD")!=0.5315017689410696
    or loss.get("trainer_source_sha256")!=sha(trainer)
    or balanced_payload.get("status")!="COMPLETE"
    or balanced_payload.get("evaluation_epoch")!=40
    or balanced_payload.get("branch")!="component_box_e160_balanced"
    or balanced_payload.get("public_slot_gate")
    or balanced_metric.get("Lesion_DSC")!=metric.get("Lesion_DSC")
    or balanced_metric.get("Lesion_NSD")!=metric.get("Lesion_NSD")
    or interim_payload.get("status")!="COMPLETE"
    or interim_payload.get("decision")!="INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME"
    or interim_payload.get("epoch_terminal",{}).get("sha256")!=loss.get("balanced_epoch40_terminal_sha256")
    or interim_metric.get("Lesion_DSC")!=metric.get("Lesion_DSC")
    or interim_metric.get("Lesion_NSD")!=metric.get("Lesion_NSD")
    or center.get("foreground_voxel_uniform_fraction")!=0.75
    or center.get("annotated_component_uniform_fraction")!=0.25
    or center.get("headroom_report_sha256")!=sha(component_audit)
    or balanced_payload.get("hidden_read") or balanced_payload.get("submission_created") or balanced_payload.get("uploaded")
    or interim_payload.get("hidden_read") or interim_payload.get("submission_created") or interim_payload.get("uploaded")
    or loss.get("hidden_read") or loss.get("submission_created") or loss.get("uploaded")):
    raise SystemExit("invalid source-stratified recall-loss decision audit")
PY

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${run_root}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected source-stratified terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi
[[ ! -e "${final}" ]]

while :; do
  read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
print(memory["MemAvailable"],avg10)
PY
  )
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" <<'PY'
import sys
available,pressure,gpu0,gpu1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu0<=1024 and gpu1<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe source-stratified launch: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

launch_args=(-pretrained_weights "${parent}")
if [[ -s "${latest}" ]]; then
  "${python_bin}" - "${latest}" <<'PY'
import sys,torch
from pathlib import Path
checkpoint=Path(sys.argv[1]); payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
epoch=int(payload.get("current_epoch",-1))
if (payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160SourceStratified"
    or not 1<=epoch<80 or not isinstance(payload.get("network_weights"),dict)
    or not isinstance(payload.get("optimizer_state"),dict)):
    raise SystemExit("invalid source-stratified optimizer resume checkpoint")
print(f"Validated source-stratified optimizer resume at current_epoch={epoch}",flush=True)
PY
  launch_args=(--c)
fi

"${python_bin}" -u "${runner}" Dataset501_PanCancer \
  3d_fullres_STUNetB_PartialBox all \
  -tr STUNetTrainerPanCancerComponentBoxE160SourceStratified -p nnUNetPlans \
  "${launch_args[@]}" -num_gpus 2 >>"${run_root}/train.log" 2>&1

[[ -s "${final}" ]]
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${parent}" "${final}" "${trainer_source}" "${runner}" \
  "${audit}" "${loader_smoke}" "${loss_audit}" "${component_audit}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
parent,final,trainer,runner,audit,smoke,loss_audit,component_audit,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B epoch-160 source/size-stratified continuation",
 "recipe":{"parent_epoch":160,"epochs":80,"initial_lr":3e-5,
   "optimizer":"SGD_Nesterov_momentum_0.99_fresh_state","schedule":"polynomial",
   "loss":"component-box CE plus Tversky a025 b075; outside boxes ignored for every case",
   "physical_gpus":[0,1],"per_rank_batch_size":1,"global_batch_size":2,
   "case_count":17575,"source_count":28,
   "case_sampling":"75% uniform plus 25% size-bin-preserving bounded source target",
   "patch_center_sampling":"75% foreground-voxel uniform plus 25% annotated-component uniform",
   "raw_per_case_ratio_cap":3.0,"foreground_oversample_fraction":1.0},
 "parent":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "sampling_audit":{"path":str(audit.resolve()),"sha256":sha(audit)},
 "native_loader_smoke":{"path":str(smoke.resolve()),"sha256":sha(smoke)},
 "recall_loss_decision":{"path":str(loss_audit.resolve()),"sha256":sha(loss_audit)},
 "component_center_audit":{"path":str(component_audit.resolve()),"sha256":sha(component_audit)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY

trap - EXIT

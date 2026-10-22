#!/usr/bin/env bash
set -euo pipefail

# Two-GPU completeness-aware continuation from the hidden-leading component
# epoch-160 weights. This script has no public/hidden input or upload path.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${root}/run_pancancer_stunet_complete_hybrid.py"
trainer_source="${root}/STUNetTrainerPanCancerCompleteHybrid.py"
inventory_audit="${root}/analysis/complete_hybrid4x_inventory_20260826.json"
loader_smoke="${root}/analysis/complete_hybrid4x_native_loader_smoke_20260826/TERMINAL.json"
plans="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json"
dataset_json="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/dataset.json"
parent="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"
parent_sha="81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160CompleteHybrid__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
run_root="${root}/runs/stunetb_componentbox_e160_completehybrid4x_80ep_20260826"
terminal="${run_root}/TRAINING_TERMINAL.json"
label_correction="${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/LABEL_CONTRACT_CORRECTION_20260825.json"

mkdir -p "${run_root}"
exec 9>"${run_root}/controller.lock"
flock -n 9 || { echo "complete-hybrid continuation already active" >&2; exit 1; }

# Fail closed even under a direct manual launch. The purported complete cohorts
# were distributed under the challenge-wide partial-label contract, so empty
# regions cannot be used as authoritative background supervision.
"${python_bin}" - "${label_correction}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
correction,output=map(Path,sys.argv[1:]); p=json.load(open(correction,encoding="utf-8"))
if p.get("status")!="INVALIDATE_COMPLETE_LABEL_INTERPRETATION":
 raise SystemExit("missing authoritative partial-label correction")
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES",
 "reason":"complete-hybrid empty-region supervision violates info.txt",
 "label_contract_correction":{"path":str(correction.resolve()),"sha256":sha(correction)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
exit 0

export CUDA_VISIBLE_DEVICES=0,1
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA="${COMPLETE_HYBRID_N_PROC_DA:-0}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export nnUNet_compile=false

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RETRY_COMPLETE_HYBRID_FROM_LATEST",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "complete-hybrid continuation failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${parent}" "${runner}" "${trainer_source}" "${inventory_audit}" \
  "${loader_smoke}" "${plans}" "${dataset_json}"; do
  [[ -s "${file}" ]] || { echo "missing complete-hybrid input: ${file}" >&2; exit 2; }
done
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${inventory_audit}" "${loader_smoke}" "${trainer_source}" \
  "${runner}" "${plans}" "${dataset_json}" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit,smoke=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:3])
trainer,runner,plans,dataset=map(Path,sys.argv[3:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (audit.get("status")!="COMPLETE" or audit.get("decision")!="IMPLEMENTED_NOT_LAUNCHED"
    or audit.get("training_inventory",{}).get("cases")!=17575
    or audit.get("training_inventory",{}).get("verified_complete_cases")!=3145):
    raise SystemExit("invalid complete-hybrid inventory audit")
inputs=smoke.get("inputs",{})
if (smoke.get("status")!="COMPLETE"
    or smoke.get("purpose")!="native_nnunet_complete_hybrid_loader_and_target_smoke_no_training"
    or smoke.get("training_case_count")!=17575
    or smoke.get("complete_case_count")!=3145 or smoke.get("partial_case_count")!=14430
    or smoke.get("identifier_inventory_sha256")!="28f4d1bfd095422f546f0f53134eb4f226963243bc89fcdd78a6d7614bceb509"
    or smoke.get("native_loader_class")!="nnUNetDataLoader"
    or smoke.get("data_shape")!=[2,1,128,160,160]
    or smoke.get("segmentation_shape")!=[2,1,128,160,160]
    or smoke.get("training_probability_sum")!=1.0
    or smoke.get("validation_sampling_probabilities") is not None
    or smoke.get("complete_valid_voxels")!=3276800
    or smoke.get("partial_valid_voxels",0)<=0 or smoke.get("partial_foreground_voxels",0)<=0
    or smoke.get("maximum_probability_absolute_error")!=0.0 or smoke.get("gpu_used")
    or smoke.get("public_or_hidden_labels_read") or smoke.get("submission_created")
    or smoke.get("uploaded")
    or inputs.get("trainer_source",{}).get("sha256")!=sha(trainer)
    or inputs.get("runner_source",{}).get("sha256")!=sha(runner)
    or inputs.get("audit",{}).get("sha256")!=sha(Path(sys.argv[1]))
    or inputs.get("plans",{}).get("sha256")!=sha(plans)
    or inputs.get("dataset_json",{}).get("sha256")!=sha(dataset)):
    raise SystemExit("invalid complete-hybrid native-loader smoke")
PY

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${run_root}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected complete-hybrid terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi
[[ ! -e "${final}" ]]

# Two DDP ranks plus W&B are launched only after all prior GPU work has exited
# and host memory pressure is low. The gate is repeated on service restart.
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
  printf '%s waiting for safe complete-hybrid launch: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

launch_args=(-pretrained_weights "${parent}")
if [[ -s "${latest}" ]]; then
  "${python_bin}" - "${latest}" <<'PY'
import sys,torch
from pathlib import Path
checkpoint=Path(sys.argv[1]);payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
epoch=int(payload.get("current_epoch",-1))
if (payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160CompleteHybrid"
    or not 1<=epoch<80 or not isinstance(payload.get("network_weights"),dict)
    or not isinstance(payload.get("optimizer_state"),dict)):
    raise SystemExit("invalid complete-hybrid optimizer resume checkpoint")
print(f"Validated complete-hybrid optimizer resume at current_epoch={epoch}",flush=True)
PY
  launch_args=(--c)
fi

"${python_bin}" -u "${runner}" Dataset501_PanCancer \
  3d_fullres_STUNetB_PartialBox all \
  -tr STUNetTrainerPanCancerComponentBoxE160CompleteHybrid -p nnUNetPlans \
  "${launch_args[@]}" -num_gpus 2 >>"${run_root}/train.log" 2>&1

[[ -s "${final}" ]]
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${parent}" "${final}" "${trainer_source}" "${runner}" \
  "${inventory_audit}" "${loader_smoke}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
parent,final,trainer,runner,audit,smoke,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B epoch-160 completeness-aware continuation",
 "recipe":{"parent_epoch":160,"epochs":80,"initial_lr":3e-5,
   "optimizer":"SGD_Nesterov_momentum_0.99_fresh_state","schedule":"polynomial",
   "loss":"verified-complete full patch else component-box CE plus Tversky a05 b05",
   "physical_gpus":[0,1],"global_batch_size":4,"complete_case_multiplier":4.0,
   "foreground_oversample_fraction":0.5,"verified_complete_cases":3145,"partial_cases":14430},
 "parent":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "inventory_audit":{"path":str(audit.resolve()),"sha256":sha(audit)},
 "native_loader_smoke":{"path":str(smoke.resolve()),"sha256":sha(smoke)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT

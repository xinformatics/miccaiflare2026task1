#!/usr/bin/env bash
set -euo pipefail

# Conditional two-GPU alpha=0.20/beta=0.80 continuation. This is a causal
# sibling of the alpha=0.25/beta=0.75 source-stratified run: exact same parent,
# sampler, optimizer, schedule, and endpoint epochs; only the Tversky weights
# differ. No public/hidden input, packaging, submission, or upload.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${root}/run_pancancer_stunet_source_aggressive.py"
trainer="${root}/STUNetTrainerPanCancerSourceAggressive.py"
source_trainer="${root}/STUNetTrainerPanCancerSourceBalanced.py"
diagnostic="${root}/analysis/source_recall_loss_diagnostic_20260826/TERMINAL.json"
parent="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"
parent_sha="81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceAggressive__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
run_root="${root}/runs/stunetb_componentbox_e160_source_a020b080_80ep_20260827"
terminal="${run_root}/TRAINING_TERMINAL.json"

mkdir -p "${run_root}"
exec 9>"${run_root}/controller.lock"
flock -n 9 || { echo "source-aggressive continuation already active" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0,1
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA="${SOURCE_AGGRESSIVE_N_PROC_DA:-0}"
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
 "status":"FAILED","reason":reason,"decision":"RETRY_SOURCE_AGGRESSIVE_FROM_LATEST",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "source-aggressive continuation failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for file in "${diagnostic}" "${parent}" "${runner}" "${trainer}" "${source_trainer}"; do
  [[ -s "${file}" ]] || { echo "missing source-aggressive input: ${file}" >&2; exit 2; }
done
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${diagnostic}" "${trainer}" "${source_trainer}" <<'PY'
import hashlib,json,sys
from pathlib import Path
diagnostic,trainer,source_trainer=map(Path,sys.argv[1:])
p=json.load(open(diagnostic,encoding="utf-8")); selected=p.get("selected") or {}
if (p.get("status")!="COMPLETE" or p.get("decision")!="QUALIFY_TVERSKY_A020_B080_FOLLOWUP"
    or not selected.get("qualifies") or not all(selected.get("checks",{}).values())
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("aggressive-loss launch gate is not qualified")
text=trainer.read_text(encoding="utf-8")
if ("PartialBoxTverskyCELoss(alpha=0.20, beta=0.80)" not in text
    or "STUNetTrainerPanCancerComponentBoxE160SourceStratified" not in text
    or "def get_dataloaders" in text or "def __init__" in text):
    raise SystemExit("aggressive trainer no longer differs by loss only")
if not source_trainer.is_file(): raise SystemExit("missing audited source trainer")
PY

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${run_root}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected source-aggressive terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi
[[ ! -e "${final}" ]]

while :; do
  read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
m={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
p=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
print(m["MemAvailable"],float(next(x.split("=",1)[1] for x in p.split() if x.startswith("avg10="))))
PY
  )
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" <<'PY'
import sys
a,p,g0,g1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if a>=30*1024*1024 and p<=2.0 and g0<=1024 and g1<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe source-aggressive launch: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
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
if (payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160SourceAggressive"
    or not 1<=epoch<80 or not isinstance(payload.get("network_weights"),dict)
    or not isinstance(payload.get("optimizer_state"),dict)):
    raise SystemExit("invalid source-aggressive optimizer resume checkpoint")
print(f"Validated source-aggressive optimizer resume at current_epoch={epoch}",flush=True)
PY
  launch_args=(--c)
fi

"${python_bin}" -u "${runner}" Dataset501_PanCancer \
  3d_fullres_STUNetB_PartialBox all \
  -tr STUNetTrainerPanCancerComponentBoxE160SourceAggressive -p nnUNetPlans \
  "${launch_args[@]}" -num_gpus 2 >>"${run_root}/train.log" 2>&1

[[ -s "${final}" ]]
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${parent}" "${final}" "${trainer}" "${source_trainer}" \
  "${runner}" "${diagnostic}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
parent,final,trainer,source_trainer,runner,diagnostic,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B epoch-160 source-stratified aggressive-loss continuation",
 "recipe":{"parent_epoch":160,"epochs":80,"initial_lr":3e-5,
   "optimizer":"SGD_Nesterov_momentum_0.99_fresh_state","schedule":"polynomial",
   "loss":"component-box CE plus Tversky a020 b080; outside boxes ignored for every case",
   "physical_gpus":[0,1],"per_rank_batch_size":1,"global_batch_size":2,
   "case_count":17575,"source_count":28,
   "case_sampling":"75% uniform plus 25% size-bin-preserving bounded source target",
   "patch_center_sampling":"75% foreground-voxel uniform plus 25% annotated-component uniform",
   "raw_per_case_ratio_cap":3.0,"foreground_oversample_fraction":1.0,
   "causal_difference_from_source_a025b075":"Tversky alpha/beta only"},
 "parent":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "inherited_source_trainer":{"path":str(source_trainer.resolve()),"sha256":sha(source_trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "launch_gate":{"path":str(diagnostic.resolve()),"sha256":sha(diagnostic)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY

trap - EXIT

#!/usr/bin/env bash
set -euo pipefail

# Precision-preserving learned continuation of the hidden-leading component-box
# epoch-160 state. This intentionally replaces the weaker global-box control on
# GPU 0; the component epoch-300 public endpoint evaluator remains on GPU 1.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${root}/run_pancancer_stunet.py"
trainer_source="${root}/STUNetTrainerPanCancer.py"
dual_launcher="${root}/launch_stunetb_dual_after_e100_20260826.sh"
validator="${root}/validate_post_e100_branch.py"
parent="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"
parent_sha="81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
base_source="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0100.pth"
base_latest="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_latest.pth"
base_unit="miccai-stunetb-base-gpu0-post-e100-20260826.service"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
latest="${fold}/checkpoint_latest.pth"
final="${fold}/checkpoint_final.pth"
run_root="${root}/runs/stunetb_componentbox_e160_balanced_80ep_20260826"
pause_marker="${run_root}/BASE_CONTROL_REPURPOSED.json"
terminal="${run_root}/TRAINING_TERMINAL.json"

mkdir -p "${run_root}"
exec 9>"${run_root}/controller.lock"
flock -n 9 || { echo "epoch-160 balanced continuation already active" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export MPLCONFIGDIR="${run_root}/.matplotlib_gpu0"
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
# This host's systemd-oomd killed the first process tree at epoch 11 and the
# reduced compiled process tree again at epoch 13 while system swap was full.
# Serialize augmentation and disable the retained compiler pool. This is a
# numerical execution/resource change only; the full optimizer state resumes.
export nnUNet_n_proc_DA="${BALANCED_N_PROC_DA:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export nnUNet_compile=false

failure_line=unknown
base_stopped=false
restore_base() {
  if [[ "${base_stopped}" == true ]]; then
    /usr/bin/bash "${dual_launcher}" || true
    base_stopped=false
  fi
}
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RESTORE_BASE_CONTROL",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 ]]; then
    if [[ ! -s "${terminal}" ]]; then
      write_failure "epoch-160 balanced continuation failed at line ${line}, status ${status}"
    fi
    restore_base
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

[[ -s "${parent}" ]]
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${run_root}/TRAINING_TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected continuation terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi
[[ ! -e "${final}" ]]

if systemctl --user is-active --quiet "${base_unit}"; then
  systemctl --user stop "${base_unit}"
  base_stopped=true
fi
while systemctl --user is-active --quiet "${base_unit}"; do sleep 2; done

base_validation=$("${python_bin}" "${validator}" --mode base \
  --source "${base_source}" --latest "${base_latest}")
"${python_bin}" - "${parent}" "${parent_sha}" "${base_latest}" \
  "${base_validation}" "${pause_marker}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
parent=Path(sys.argv[1]); parent_sha=sys.argv[2]; latest=Path(sys.argv[3])
validation=json.loads(sys.argv[4]); output=Path(sys.argv[5])
observed=hashlib.sha256(parent.read_bytes()).hexdigest()
if observed != parent_sha:
    raise SystemExit("parent checkpoint changed")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "reason":"replace weaker global-box control with hidden-leading component-box continuation",
 "parent":{"path":str(parent.resolve()),"sha256":parent_sha,"epoch":160},
 "base_resume":{"path":str(latest.resolve()),"sha256":hashlib.sha256(latest.read_bytes()).hexdigest(),
                "epoch":validation["latest_epoch"]},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

# On an automatic restart after systemd-oomd, do not immediately rebuild the
# CUDA/data process tree while the host is still thrashing. This gate reads
# only kernel memory telemetry and holds no model/data allocation.
while :; do
  read -r memory_available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
print(memory["MemAvailable"],avg10)
PY
  )
  if "${python_bin}" - "${memory_available_kib}" "${full_avg10}" <<'PY'
import sys
raise SystemExit(0 if int(sys.argv[1]) >= 20*1024*1024 and float(sys.argv[2]) <= 5.0 else 1)
PY
  then
    break
  fi
  printf '%s waiting for host memory recovery: MemAvailable=%s KiB full.avg10=%s\n' \
    "$(date -Ins)" "${memory_available_kib}" "${full_avg10}"
  sleep 30
done

while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for idle physical GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 10
done

launch_args=(-pretrained_weights "${parent}")
if [[ -s "${latest}" ]]; then
  "${python_bin}" - "${latest}" <<'PY'
import sys,torch
from pathlib import Path
checkpoint=Path(sys.argv[1])
# This is a locally generated optimizer checkpoint from this exact run.
payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
epoch=int(payload.get("current_epoch",-1))
if (payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160Balanced"
    or not 1 <= epoch < 80 or not isinstance(payload.get("network_weights"),dict)
    or not isinstance(payload.get("optimizer_state"),dict)):
    raise SystemExit("invalid balanced continuation resume checkpoint")
print(f"Validated balanced optimizer resume at current_epoch={epoch}",flush=True)
PY
  launch_args=(--c)
fi

"${python_bin}" -u "${runner}" \
  Dataset501_PanCancer 3d_fullres_STUNetB_PartialBox all \
  -tr STUNetTrainerPanCancerComponentBoxE160Balanced -p nnUNetPlans \
  "${launch_args[@]}" -num_gpus 1 \
  >>"${run_root}/train.log" 2>&1

[[ -s "${final}" ]]
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]
"${python_bin}" - "${parent}" "${final}" "${trainer_source}" "${runner}" \
  "${pause_marker}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
parent,final,trainer,runner,pause,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","model":"STU-Net-B component-box epoch-160 balanced continuation",
 "recipe":{"parent_epoch":160,"epochs":80,"initial_lr":3e-5,
   "optimizer":"SGD_Nesterov_momentum_0.99_fresh_state","schedule":"polynomial",
   "loss":"trusted_component_box_CE_plus_Tversky_a05_b05","physical_gpu":0,
   "global_batch_size":2,"trusted_region":"union of expanded 26-connected component bboxes"},
 "parent":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "final_checkpoint":{"path":str(final.resolve()),"sha256":sha(final)},
 "trainer_source":{"path":str(trainer.resolve()),"sha256":sha(trainer)},
 "runner":{"path":str(runner.resolve()),"sha256":sha(runner)},
 "base_control_pause":{"path":str(pause.resolve()),"sha256":sha(pause)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

base_stopped=false
trap - EXIT

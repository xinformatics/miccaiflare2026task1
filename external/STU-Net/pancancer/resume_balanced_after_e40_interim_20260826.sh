#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
interim="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826/E40_INTERIM_COMPLETE.json"
checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0040.pth"
checkpoint_sha="3015cdf2d9b267989fd2fde2939cdac8670c55c31b96fd50f59eadf9da2a29f6"
latest="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_latest.pth"
trainer_unit="miccai-stunetb-component-e160-balanced-gpu0-20260826.service"
trainer_script="${root}/run_component_e160_balanced_continuation_20260826.sh"
trainer_pid=2087594
wandb_pid=2087741
work="${root}/runs/resume_balanced_after_e40_interim_20260826"
terminal="${work}/TERMINAL.json"
evaluator_unit="miccai-stunetb-balanced-e40-e80-public-20260826.service"
evaluator="${root}/run_balanced_e40_e80_public_20260826.sh"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "balanced interim resume watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${interim}" ]]; do
  printf '%s waiting for epoch-40 interim metrics before resuming training\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" - "${interim}" "${checkpoint}" "${checkpoint_sha}" <<'PY'
import hashlib,json,sys
from pathlib import Path
interim,checkpoint=map(Path,sys.argv[1:3]); expected=sys.argv[3]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
p=json.load(open(interim,encoding="utf-8")); endpoint=Path(p.get("epoch_terminal",{}).get("path",""))
if (p.get("status")!="COMPLETE"
    or p.get("decision")!="INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME"
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")
    or p.get("epoch_terminal",{}).get("sha256")!=sha(endpoint)
    or sha(checkpoint)!=expected):
    raise SystemExit("invalid epoch-40 interim resume contract")
e=json.load(open(endpoint,encoding="utf-8"))
if (e.get("status")!="COMPLETE" or e.get("evaluation_epoch")!=40
    or e.get("checkpoint",{}).get("sha256")!=expected):
    raise SystemExit("invalid epoch-40 endpoint provenance")
PY

resume_mode="in_memory_process_group"
if [[ -r "/proc/${trainer_pid}/stat" && -r "/proc/${wandb_pid}/stat" ]]; then
  for pid in "${trainer_pid}" "${wandb_pid}"; do
    state=$(awk '{print $3}' "/proc/${pid}/stat")
    [[ "${state}" == T || "${state}" == t ]] || {
      echo "process ${pid} is not frozen: ${state}" >&2; exit 2;
    }
  done
  kill -CONT -- "-${trainer_pid}" "-${wandb_pid}"
else
  # A frozen CUDA process still retained enough host memory for systemd-oomd
  # to kill inference on a large public CT. In that case the trainer service is
  # deliberately stopped without its failure trap and restarted from the last
  # exact optimizer checkpoint only after interim scoring finishes.
  resume_mode="optimizer_checkpoint_current_epoch_47"
  [[ ! -e "/proc/${trainer_pid}" && ! -e "/proc/${wandb_pid}" ]]
  [[ -s "${latest}" ]]
  "${python_bin}" - "${latest}" <<'PY'
import sys,torch
payload=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
if (payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160Balanced"
    or payload.get("current_epoch")!=47
    or not isinstance(payload.get("network_weights"),dict)
    or not isinstance(payload.get("optimizer_state"),dict)):
    raise SystemExit("invalid durable balanced optimizer checkpoint")
PY
  if systemctl --user cat "${trainer_unit}" >/dev/null 2>&1; then
    systemctl --user start "${trainer_unit}"
  else
    systemd-run --user --unit="${trainer_unit%.service}" \
      --description="MICCAI-balanced-STUNet-e160-continuation-GPU0" \
      --property=MemoryHigh=32G --property=MemoryMax=40G --property=TasksMax=128 \
      --property=Restart=on-failure --property=RestartSec=60s \
      --property=StartLimitBurst=6 --property=StartLimitIntervalSec=3600s \
      /usr/bin/bash "${trainer_script}"
  fi
fi

if ! systemctl --user is-active --quiet "${evaluator_unit}"; then
  if systemctl --user cat "${evaluator_unit}" >/dev/null 2>&1; then
    systemctl --user start "${evaluator_unit}"
  else
    systemd-run --user --unit="${evaluator_unit%.service}" \
      --property=MemoryHigh=16G --property=MemoryMax=24G --property=TasksMax=64 \
      --property=Restart=on-failure --property=RestartSec=60s \
      --property=StartLimitBurst=3 --property=StartLimitIntervalSec=3600s \
      /usr/bin/bash "${evaluator}"
  fi
fi

"${python_bin}" - "${interim}" "${checkpoint}" "${terminal}" "${resume_mode}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
interim,checkpoint,output=map(Path,sys.argv[1:4]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.load(open(interim,encoding="utf-8"))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"BALANCED_TRAINING_RESUMED_AFTER_E40_PUBLIC",
 "interim":{"path":str(interim.resolve()),"sha256":sha(interim)},
 "epoch40_checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "metrics":p["metrics"],"public_slot_gate":p["public_slot_gate"],
 "training_resume_mode":sys.argv[4],
 "trainer_process_group":2087594 if sys.argv[4]=="in_memory_process_group" else None,
 "wandb_process_group":2087741 if sys.argv[4]=="in_memory_process_group" else None,
 "restored_evaluator":"miccai-stunetb-balanced-e40-e80-public-20260826.service",
 "hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY

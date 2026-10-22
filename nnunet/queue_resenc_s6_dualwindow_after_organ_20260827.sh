#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
organ_public="${project_root}/nnunet/public_sweeps/resenc_s6_organ_target_endpoints_20260827/TERMINAL.json"
parent_terminal="${project_root}/nnunet/runs/resenc_s6_positive_unlabeled_f020_100ep_20260827/TRAINING_TERMINAL.json"
parent="${project_root}/nnUNet_results/Dataset502_PanCancerPositiveUnlabeled/nnUNetTrainerResEncS6PanCancerPositiveUnlabeled__nnUNetPlans__3d_fullres_ResEncS6/fold_all/checkpoint_final.pth"
controller="${project_root}/nnunet/run_resenc_s6_dualwindow_2gpu_20260827.sh"
training_terminal="${project_root}/nnunet/runs/resenc_s6_dualwindow_f020_100ep_20260827/TRAINING_TERMINAL.json"
work="${project_root}/nnunet/runs/queue_resenc_s6_dualwindow_after_organ_20260827"
terminal="${work}/TERMINAL.json"
unit="miccai-resenc-s6-dualwindow-20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "dual-window ResEnc queue already active" >&2; exit 1; }

while [[ ! -s "${organ_public}" || ! -s "${parent_terminal}" || ! -s "${parent}" ]]; do
  printf '%s waiting for organ public audit and deterministic positive parent\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" - "${organ_public}" "${parent_terminal}" "${parent}" <<'PY'
import hashlib,json,sys
from pathlib import Path
organ_path,parent_terminal_path,parent_path=map(Path,sys.argv[1:])
organ=json.load(open(organ_path,encoding="utf-8"))
parent_terminal=json.load(open(parent_terminal_path,encoding="utf-8"))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if (organ.get("status")!="COMPLETE" or organ.get("hidden_read")
    or organ.get("submission_created") or organ.get("uploaded")):
    raise SystemExit("organ-target audit is not safely terminal")
if (parent_terminal.get("status")!="COMPLETE"
    or parent_terminal.get("final_checkpoint",{}).get("sha256")!=sha(parent_path)
    or parent_terminal.get("public_read") or parent_terminal.get("hidden_read")):
    raise SystemExit("positive-only deterministic parent is invalid")
PY

if [[ -s "${training_terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${training_terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
fi

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
  util0=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  util1=$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" "${util0}" "${util1}" <<'PY'
import sys
available,pressure,gpu0,gpu1,util0,util1=(int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]))
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu0<=3072 and gpu1<=3072 and util0<=5 and util1<=5 else 1)
PY
  then break; fi
  printf '%s waiting for safe dual-window launch: available=%s pressure=%s gpu0=%s/%s%% gpu1=%s/%s%%\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${util0}" "${used1}" "${util1}"
  sleep 30
done

if ! systemctl --user is-active --quiet "${unit}.service"; then
  systemd-run --user --unit="${unit}" \
    --property=MemoryHigh=48G --property=MemoryMax=64G --property=TasksMax=192 \
    --property=Restart=on-failure --property=RestartSec=60s \
    --property=StartLimitBurst=3 --property=StartLimitIntervalSec=3600s \
    /usr/bin/bash "${controller}"
fi

"${python_bin}" - "${organ_public}" "${parent_terminal}" "${parent}" \
  "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
organ,parent_terminal,parent,controller,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"START_DUALWINDOW_RESENC_AFTER_ORGAN_PUBLIC_AUDIT",
 "organ_public_terminal":{"path":str(organ.resolve()),"sha256":sha(organ)},
 "parent_training_terminal":{"path":str(parent_terminal.resolve()),"sha256":sha(parent_terminal)},
 "parent_checkpoint":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-resenc-s6-dualwindow-20260827.service",
 "public_used_for_training":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY


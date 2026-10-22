#!/usr/bin/env bash
set -euo pipefail

# Preserve the current evidence-driven order. Corrected ResEnc S6 starts only
# after the valid source/aggressive STU branches, long Focus continuation, and
# any public-qualified label-blind hidden save have fully released the GPUs.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus="${project_root}/external/GF-Screen/public_sweeps/gf_focus_pu_long30_e283_to313_20260825/TERMINAL.json"
focus_saved="${project_root}/external/GF-Screen/public_sweeps/gf_focus_pu_long30_trajectory_saved_20260825/TERMINAL.json"
controller="${project_root}/nnunet/run_resenc_s6_partialbox_2gpu_20260827.sh"
training_terminal="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827/TRAINING_TERMINAL.json"
work="${project_root}/nnunet/runs/queue_resenc_s6_partialbox_after_focus_20260827"
terminal="${work}/TERMINAL.json"
unit="miccai-resenc-s6-partialbox-20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "corrected ResEnc queue already active" >&2; exit 1; }

while [[ ! -s "${focus}" || ! -s "${focus_saved}" ]]; do
  printf '%s waiting for long Focus controller and save terminals\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${focus}" "${focus_saved}" <<'PY'
import json,sys
focus,saved=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if (focus.get("status") not in {"COMPLETE","SKIPPED"}
    or saved.get("status") not in {"COMPLETE","SKIPPED"}
    or focus.get("uploaded") or saved.get("uploaded") or saved.get("hidden_labels_read")):
    raise SystemExit("long Focus chain has not ended in an upload-safe terminal")
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
  # The exact two-rank smoke test peaked at 43.5 GiB reserved per 49.1 GiB
  # device.  Permit the stable <=3 GiB background allocation observed on GPU0
  # while retaining >2.5 GiB measured headroom; utilization and host-pressure
  # guards below still prevent launch during active competing work.
  util0=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  util1=$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" "${util0}" "${util1}" <<'PY'
import sys
available,pressure,gpu0,gpu1,util0,util1=(
    int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),
    int(sys.argv[5]),int(sys.argv[6]),
)
raise SystemExit(
    0
    if available>=30*1024*1024 and pressure<=2.0
    and gpu0<=3072 and gpu1<=3072 and util0<=5 and util1<=5
    else 1
)
PY
  then break; fi
  printf '%s waiting for safe corrected ResEnc launch: available=%s KiB pressure=%s gpu0=%s/%s%% gpu1=%s/%s%% MiB\n' \
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

"${python_bin}" - "${focus}" "${focus_saved}" "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
focus,saved,controller,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"START_CORRECTED_RESENC_S6_AFTER_VALID_CAMPAIGNS",
 "focus_terminal":{"path":str(focus.resolve()),"sha256":sha(focus)},
 "focus_save_terminal":{"path":str(saved.resolve()),"sha256":sha(saved)},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-resenc-s6-partialbox-20260827.service",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

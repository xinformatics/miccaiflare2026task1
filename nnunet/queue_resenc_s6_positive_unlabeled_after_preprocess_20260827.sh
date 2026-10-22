#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
preprocess="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/PREPROCESSING_COMPLETE.json"
parent="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827/TRAINING_TERMINAL.json"
controller="${project_root}/nnunet/run_resenc_s6_positive_unlabeled_2gpu_20260827.sh"
training_terminal="${project_root}/nnunet/runs/resenc_s6_positive_unlabeled_f020_100ep_20260827/TRAINING_TERMINAL.json"
work="${project_root}/nnunet/runs/queue_resenc_s6_positive_unlabeled_after_preprocess_20260827"
terminal="${work}/TERMINAL.json"
unit="miccai-resenc-s6-positive-unlabeled-20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "positive-unlabeled ResEnc queue already active" >&2; exit 1; }

while [[ ! -s "${preprocess}" || ! -s "${parent}" ]]; do
  printf '%s waiting for Dataset502 preprocessing and parent ResEnc terminal\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${preprocess}" "${parent}" <<'PY'
import json,sys
preprocess,parent=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if (preprocess.get("status")!="COMPLETE" or preprocess.get("total_cases")!=19886
    or preprocess.get("positive_unlabeled_preprocessed_cases")!=2311
    or preprocess.get("public_read") or preprocess.get("hidden_read")
    or parent.get("status")!="COMPLETE" or parent.get("public_read")
    or parent.get("hidden_read")):
    raise SystemExit("positive-unlabeled prerequisites are not safe and complete")
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
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" <<'PY'
import sys
available,pressure,gpu0,gpu1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu0<=1024 and gpu1<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe positive-unlabeled launch: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

if ! systemctl --user is-active --quiet "${unit}.service"; then
  systemd-run --user --unit="${unit}" \
    --property=MemoryHigh=48G --property=MemoryMax=64G --property=TasksMax=192 \
    --property=Restart=on-failure --property=RestartSec=60s \
    --property=StartLimitBurst=3 --property=StartLimitIntervalSec=3600s \
    /usr/bin/bash "${controller}"
fi

"${python_bin}" - "${preprocess}" "${parent}" "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
preprocess,parent,controller,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"START_POSITIVE_UNLABELED_RESENC_AFTER_PARENT_AND_PREPROCESSING",
 "preprocessing_terminal":{"path":str(preprocess.resolve()),"sha256":sha(preprocess)},
 "parent_training_terminal":{"path":str(parent.resolve()),"sha256":sha(parent)},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-resenc-s6-positive-unlabeled-20260827.service",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

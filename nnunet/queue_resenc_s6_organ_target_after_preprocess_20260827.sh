#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
preprocess="${project_root}/nnUNet_preprocessed/Dataset503_PanCancerOrganTarget/PREPROCESSING_COMPLETE.json"
positive_public="${project_root}/nnunet/public_sweeps/resenc_s6_positive_unlabeled_endpoints_20260827/TERMINAL.json"
controller="${project_root}/nnunet/run_resenc_s6_organ_target_2gpu_20260827.sh"
training_terminal="${project_root}/nnunet/runs/resenc_s6_organ_target_f001_25ep_20260827/TRAINING_TERMINAL.json"
work="${project_root}/nnunet/runs/queue_resenc_s6_organ_target_after_preprocess_20260827"
terminal="${work}/TERMINAL.json"
unit="miccai-resenc-s6-organ-target-20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "organ-target ResEnc queue already active" >&2; exit 1; }

while [[ ! -s "${preprocess}" || ! -s "${positive_public}" ]]; do
  printf '%s waiting for Dataset503 preprocessing and positive selector\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${preprocess}" "${positive_public}" <<'PY'
import json,sys
preprocess,positive=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if (preprocess.get("status")!="COMPLETE" or preprocess.get("total_cases")!=17579
    or preprocess.get("organ_target_preprocessed_cases")!=4
    or preprocess.get("public_read") or preprocess.get("hidden_read")
    or positive.get("status")!="COMPLETE" or not isinstance(positive.get("diagnostic_best"),dict)
    or positive.get("hidden_read") or positive.get("submission_created")
    or positive.get("uploaded")):
    raise SystemExit("organ-target prerequisites are not safe and complete")
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
  printf '%s waiting for safe organ-target launch: available=%s pressure=%s gpu0=%s/%s%% gpu1=%s/%s%%\n' \
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

"${python_bin}" - "${preprocess}" "${positive_public}" "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
preprocess,positive,controller,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"START_ORGAN_TARGET_RESENC_AFTER_POSITIVE_AND_PREPROCESSING",
 "preprocessing_terminal":{"path":str(preprocess.resolve()),"sha256":sha(preprocess)},
 "positive_public_selector":{"path":str(positive.resolve()),"sha256":sha(positive)},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-resenc-s6-organ-target-20260827.service",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

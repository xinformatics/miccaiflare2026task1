#!/usr/bin/env bash
set -euo pipefail

# Finish the already predeclared long-Focus positive-unlabeled trajectory after
# the valid partial-box source/aggressive STU campaigns. Historical exact-707
# additive and complete-hybrid branches are excluded because they used missing
# annotations as negatives. This controller reads no challenge image and
# cannot upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
aggressive_terminal="${project}/external/STU-Net/pancancer/runs/source_aggressive_campaign_20260827/TERMINAL.json"
checkpoint="${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0303.pt"
checkpoint_sha="76022fa403700a38ce2aa9042aad3446687020e259c0bb5d1a65d8f698dd0bbc"
deferral="${root}/public_sweeps/gf_focus_pu_e303_interim_saved_20260825/TRAINING_DEFERRED_FOR_ENSEMBLE.json"
final_checkpoint="${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0313.pt"
training_controller="${root}/run_long_focus_pu_after_direct_public_20260825.sh"
resume_unit="miccai-long-focus-pu-after-balanced-20260826"
work="${root}/public_sweeps/gf_focus_e303_to313_after_balanced_20260826"
terminal="${work}/TERMINAL.json"
lock="${root}/runs/queue_focus_e303_to313_after_balanced_20260826.lock"

mkdir -p "${work}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "post-balanced Focus queue already active" >&2; exit 1; }

while [[ ! -s "${aggressive_terminal}" ]]; do
  printf '%s waiting for valid partial-box aggressive-loss campaign\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${aggressive_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"}
    or p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}
    or p.get("hidden_labels_read") or p.get("uploaded")):
 raise SystemExit("invalid aggressive-loss terminal")
PY

for file in "${checkpoint}" "${deferral}" "${training_controller}"; do
  [[ -s "${file}" ]] || { echo "missing preserved Focus continuation asset: ${file}" >&2; exit 2; }
done
[[ ! -e "${final_checkpoint}" ]] || {
  printf '%s Focus checkpoint 313 already exists; no continuation launch needed\n' "$(date -Ins)"
  exit 0
}
"${python_bin}" - "${checkpoint}" "${checkpoint_sha}" "${deferral}" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
checkpoint=Path(sys.argv[1]); expected=sys.argv[2]; deferral=Path(sys.argv[3])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(checkpoint)!=expected: raise SystemExit("checkpoint-303 SHA256 mismatch")
d=json.load(open(deferral,encoding="utf-8")); item=d.get("resume_checkpoint",{})
if (d.get("status")!="DEFERRED" or Path(item.get("path","")).resolve()!=checkpoint.resolve()
    or item.get("sha256")!=expected or d.get("hidden_labels_read") or d.get("uploaded")):
    raise SystemExit("invalid checkpoint-303 deferral contract")
state=torch.load(checkpoint,map_location="cpu",weights_only=True)
if int(state.get("epoch",-1))!=302 or not state.get("optimizer_seg") or not state.get("scheduler_seg"):
    raise SystemExit("checkpoint-303 lacks exact optimizer/scheduler resume state")
PY

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
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu0<=4096 and gpu1<=1024 else 1)
PY
  then
    break
  fi
  printf '%s waiting for safe two-GPU Focus launch: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

if systemctl --user is-active --quiet "${resume_unit}.service"; then exit 0; fi
systemd-run --user --unit="${resume_unit}" \
  --property=MemoryHigh=48G --property=MemoryMax=64G --property=TasksMax=512 \
  --property=Restart=on-failure --property=RestartSec=60s \
  --property=StartLimitBurst=3 --property=StartLimitIntervalSec=3600s \
  /usr/bin/bash "${training_controller}"

"${python_bin}" - "${aggressive_terminal}" "${checkpoint}" "${checkpoint_sha}" \
  "${training_controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
aggressive,checkpoint=map(Path,sys.argv[1:3])
expected=sys.argv[3]; controller=Path(sys.argv[4]); output=Path(sys.argv[5])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"FINISH_PREDECLARED_FOCUS_E303_TO_E313",
 "aggressive_loss_campaign":{"path":str(aggressive.resolve()),"sha256":sha(aggressive)},
 "resume_checkpoint":{"path":str(checkpoint.resolve()),"sha256":expected},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

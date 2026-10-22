#!/usr/bin/env bash
set -euo pipefail

# Use one bounded idle-GPU window to advance the exact-707 learned additive
# full-volume OOF cache. The underlying controller validates every reused NPZ.
# This wrapper never reads public/hidden data and deliberately stops before the
# balanced endpoint is expected to release the two-GPU training critical path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
controller="${root}/run_focus_classifier_fusion_fullvolume_oof_20260824.sh"
output="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
predictions="${output}/fold0/work_bbox24/predictions"
balanced_terminal="${project}/external/STU-Net/pancancer/runs/stunetb_componentbox_e160_balanced_80ep_20260826/TRAINING_TERMINAL.json"
window_terminal="${output}/RESUME_WINDOW_20260826.json"
window_log="${output}/resume_window_20260826.log"
window_seconds="${FOCUS_RESUME_WINDOW_SECONDS:-3300}"

mkdir -p "${output}"
exec 9>"${root}/runs/run_focus_fusion_resume_window_20260826.lock"
flock -n 9 || { echo "Focus-fusion resume window already active" >&2; exit 1; }
[[ ! -s "${balanced_terminal}" ]] || exit 0
[[ "${window_seconds}" =~ ^[0-9]+$ && "${window_seconds}" -ge 600 \
    && "${window_seconds}" -le 3600 ]] \
  || { echo "invalid bounded resume duration" >&2; exit 2; }

read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
print(memory["MemAvailable"],avg10)
PY
)
gpu1_used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
"${python_bin}" - "${available_kib}" "${full_avg10}" "${gpu1_used}" <<'PY'
import sys
available,pressure,gpu=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu<=1024 else 1)
PY

before=$(find "${predictions}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
started_at=$(date -Ins)
set +e
/usr/bin/timeout --signal=TERM --kill-after=60s "${window_seconds}s" \
  /usr/bin/bash "${controller}" frozen >>"${window_log}" 2>&1
controller_status=$?
set -e
after=$(find "${predictions}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)

"${python_bin}" - "${started_at}" "${window_seconds}" "${controller_status}" \
  "${before}" "${after}" "${balanced_terminal}" "${window_log}" \
  "${window_terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
started,duration,status,before,after=sys.argv[1:6]
balanced,log,output=map(Path,sys.argv[6:])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if int(status)==0 else "BOUNDED_PAUSE",
 "decision":"CONTINUE_FROM_VALIDATED_PARTIAL_CACHE",
 "started_at":started,"window_seconds":int(duration),"controller_exit_status":int(status),
 "fold0_prediction_count_before":int(before),"fold0_prediction_count_after":int(after),
 "new_predictions":int(after)-int(before),"balanced_training_terminal_present":balanced.is_file(),
 "log":{"path":str(log.resolve()),"sha256":hashlib.sha256(log.read_bytes()).hexdigest()},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

case "${controller_status}" in 0|124|137|143) exit 0 ;; *) exit "${controller_status}" ;; esac

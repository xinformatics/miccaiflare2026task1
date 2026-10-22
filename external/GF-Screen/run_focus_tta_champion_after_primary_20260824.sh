#!/usr/bin/env bash
set -euo pipefail

# Run the bounded public-only flip-TTA audit as soon as its CPU preflight has
# passed and a GPU is naturally free. It is independent of exact-707 training,
# so waiting for all five primary folds would unnecessarily delay a potentially
# useful submission. The terminal is written for both success and failure so
# longer follow-up training cannot deadlock. This wrapper has no hidden
# inference, archive creation, network request, or upload operation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/public_sweeps/gf_champion_focus_tta_20260824"
terminal="${output}/PUBLIC_TTA_TERMINAL.json"
preflight="${root}/public_sweeps/gf_champion_focus_tta_preflight_20260824/PREFLIGHT_COMPLETE.json"

mkdir -p "${output}" "${root}/runs"
exec 9>"${root}/runs/run_focus_tta_champion_after_primary_20260824.lock"
flock -n 9 || { echo "champion TTA audit already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${preflight}" ]]; do
  printf '%s waiting for TTA deployment preflight\n' "$(date -Ins)"
  sleep 30
done

status=0
preflight_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${preflight}")
if [[ "${preflight_status}" == PASS ]]; then
  (
    cd "${root}"
    PYTHON_BIN="${python_bin}" OUTPUT_ROOT="public_sweeps/gf_champion_focus_tta_20260824" \
      bash run_focus_tta_champion_public_20260824.sh
  ) > "${output}/controller.log" 2>&1 || status=$?
else
  status=2
  echo "TTA deployment preflight did not pass" > "${output}/controller.log"
fi

"${python_bin}" - "${status}" "${preflight}" "${output}/summary.tsv" \
  "${output}/runtime.tsv" "${terminal}" <<'PY'
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

status = int(sys.argv[1])
preflight, summary, runtime, output = map(Path, sys.argv[2:])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

valid = False
rows = []
if status == 0 and summary.is_file() and runtime.is_file():
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    valid = {row.get("variant") for row in rows} == {"notta", "tta_x", "tta_xy"}

payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE" if valid else "ERROR",
    "experiment": "fixed-public champion Focus flip-TTA audit",
    "exit_status": status,
    "scheduling": "preflight passed, then first naturally free GPU",
    "preflight": str(preflight.resolve()),
    "preflight_sha256": sha(preflight),
    "summary": str(summary.resolve()) if summary.is_file() else None,
    "summary_sha256": sha(summary) if summary.is_file() else None,
    "runtime": str(runtime.resolve()) if runtime.is_file() else None,
    "runtime_sha256": sha(runtime) if runtime.is_file() else None,
    "rows": rows,
    "public_read": True,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(payload, indent=2))
PY

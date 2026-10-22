#!/usr/bin/env bash
set -euo pipefail

# Preserve the fold-0 route admission decision after pooled fivefold scoring.
# CPU-only and local-only; it cannot infer challenge data or create a ZIP.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824"
fivefold_terminal="${work}/FIVEFOLD_TERMINAL.json"
pilot_terminal="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_pilot_20260824/ROUTE_TERMINAL.json"
decision="${work}/RESTRICTED_DECISION.json"
terminal="${work}/RESTRICTED_TERMINAL.json"

exec 9>"${root}/runs/run_focus_encoder_upgrade_route_restriction_20260824.lock"
flock -n 9 || { echo "encoder route restriction watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${fivefold_terminal}" || ! -s "${pilot_terminal}" ]]; do
  printf '%s waiting for pilot and fivefold encoder route terminals\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" -u "${root}/restrict_focus_encoder_upgrade_routes.py" \
  --fivefold-terminal "${fivefold_terminal}" --pilot-terminal "${pilot_terminal}" \
  --output "${decision}" > "${work}/restrict.log" 2>&1
"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "qualified_routes":decision["qualified_routes"],
 "restricted_decision":str(source.resolve()),
 "restricted_decision_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

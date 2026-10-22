#!/usr/bin/env bash
set -euo pipefail

# Compose the qualified encoder-adapted Focus routes with the already frozen
# B32/Focus deployment. CPU-only; no challenge inference or submission path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
base_root="${root}/analysis/focus_composite_routes_20260824"
base_terminal="${base_root}/TERMINAL.json"
base_composite="${base_root}/DECISION.json"
upgrade_root="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824"
upgrade_terminal="${upgrade_root}/RESTRICTED_TERMINAL.json"
candidate_routes="${upgrade_root}/focus_epoch_routes.json"
upgrade_decision="${upgrade_root}/RESTRICTED_DECISION.json"
tta_root="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_routed_tta_fivefold_20260824"
tta_terminal="${tta_root}/TERMINAL.json"
candidate_tta="${tta_root}/DECISION.json"
output="${root}/analysis/focus_encoder_upgrade_composite_routes_20260824"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_focus_encoder_upgrade_composite_route_decision_20260824.lock"
flock -n 9 || { echo "encoder composite route watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" observed="$2" reason="$3"
  "${python_bin}" - "${terminal}" "${status}" "${observed}" "${reason}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":sys.argv[2],"decision":sys.argv[3],"reason":sys.argv[4],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_FROZEN_COMPOSITE \
    "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${base_terminal}" || ! -s "${upgrade_terminal}" || ! -s "${tta_terminal}" ]]; do
  printf '%s waiting for base composite and encoder route/TTA terminals\n' "$(date -Ins)"
  sleep 30
done
base_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${base_terminal}")
upgrade_result=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' "${upgrade_terminal}")
tta_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${tta_terminal}")
if [[ "${base_status}" != COMPLETE || \
      "${upgrade_result}" != PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF || \
      "${tta_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_FROZEN_COMPOSITE \
    "base composite or encoder route/TTA evidence did not qualify"
  trap - EXIT
  exit 0
fi
for required in "${base_composite}" "${candidate_routes}" \
  "${upgrade_decision}" "${candidate_tta}"; do
  [[ -s "${required}" ]] || { echo "missing encoder composite input: ${required}" >&2; exit 2; }
done
"${python_bin}" -u "${root}/compose_focus_encoder_upgrade_routes.py" \
  --base-composite "${base_composite}" --candidate-routes "${candidate_routes}" \
  --upgrade-decision "${upgrade_decision}" --candidate-tta "${candidate_tta}" \
  --output "${decision}" > "${output}/compose.log" 2>&1
observed=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
write_terminal COMPLETE "${observed}" \
  "exact local encoder-upgraded composite route is ready for full-volume OOF"
trap - EXIT
cat "${decision}"

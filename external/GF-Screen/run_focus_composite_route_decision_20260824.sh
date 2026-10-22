#!/usr/bin/env bash
set -euo pipefail

# Join exact case-held-out B32/Focus architecture routing, per-source Focus
# epoch routing, and pooled five-fold x/y TTA qualification. This watcher is
# CPU-only and cannot read public/hidden volumes or create a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
tta_root="${root}/analysis/focus_dualwindow_cell707_routed_tta_fivefold_20260824"
output="${root}/analysis/focus_composite_routes_20260824"
epoch_routes="${focus}/focus_epoch_routes.json"
epoch_gate="${focus}/focus_epoch_routes_decision.json"
mixed_route="${focus}/focus_b32_source_hybrid_decision.json"
tta_decision="${tta_root}/DECISION.json"
tta_terminal="${tta_root}/TERMINAL.json"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}" "${root}/runs"
exec 9>"${root}/runs/run_focus_composite_route_decision_20260824.lock"
flock -n 9 || { echo "composite Focus route watcher already active" >&2; exit 1; }
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_EXISTING_ROUTE \
    "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${tta_terminal}" || ! -s "${epoch_gate}" ]]; do
  printf '%s waiting for exact five-fold Focus/TTA decisions\n' "$(date -Ins)"
  sleep 30
done
tta_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${tta_terminal}")
epoch_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${epoch_gate}")
if [[ "${tta_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_EXISTING_ROUTE \
    "five-fold routed TTA audit did not complete"
  trap - EXIT
  exit 0
fi
case "${epoch_status}" in
  QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME|KEEP_GLOBAL_FOCUS_EPOCHS) ;;
  *)
    write_terminal SKIPPED KEEP_EXISTING_ROUTE \
      "Focus epoch-route evidence was not a terminal local decision"
    trap - EXIT
    exit 0
    ;;
esac
for required in "${epoch_routes}" "${mixed_route}" "${tta_decision}"; do
  [[ -s "${required}" ]] || { echo "missing composite route input: ${required}" >&2; exit 2; }
done
mixed_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${mixed_route}")
if [[ "${mixed_status}" != QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME ]]; then
  write_terminal SKIPPED KEEP_EXISTING_ROUTE \
    "mixed B32/Focus architecture gate did not qualify"
  trap - EXIT
  exit 0
fi

"${python_bin}" -u "${root}/compose_focus_b32_epoch_tta_routes.py" \
  --epoch-routes "${epoch_routes}" --epoch-gate "${epoch_gate}" \
  --mixed-route "${mixed_route}" \
  --tta-decision "${tta_decision}" --output "${decision}" \
  > "${output}/compose.log" 2>&1
observed=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
write_terminal COMPLETE "${observed}" \
  "exact local composite architecture/epoch/TTA route is ready for full-volume OOF"
trap - EXIT
cat "${decision}"

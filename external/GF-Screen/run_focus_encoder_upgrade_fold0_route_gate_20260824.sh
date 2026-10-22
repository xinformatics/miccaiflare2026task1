#!/usr/bin/env bash
set -euo pipefail

# Retrospective route-corrected gate for the already-running batch-8 fold-0
# encoder experiment. The gate was frozen after the epoch-1 source diagnostic
# and does not alter training. It reads internal held-out patches only.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
run="${root}/runs/focus_dualwindow_encoder01b8_upgrade_fold0_20260824"
analysis="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_pilot_20260824"
plan="${analysis}/ROUTE_PLAN.json"
audits="${analysis}/route_gate_audits"
candidate_routes="${analysis}/ROUTE_CANDIDATE.json"
frozen_routes="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_epoch_routes.json"
decision="${analysis}/ROUTE_DECISION.json"
terminal="${analysis}/ROUTE_TERMINAL.json"

mkdir -p "${audits}"
exec 9>"${root}/runs/run_focus_encoder_upgrade_fold0_route_gate_20260824.lock"
flock -n 9 || {
  echo "Focus encoder-upgrade fold-0 route gate already active" >&2
  exit 1
}
[[ -s "${terminal}" ]] && exit 0
for required in "${index}" "${plan}" "${frozen_routes}"; do
  [[ -s "${required}" ]] || {
    echo "missing route-gate asset: ${required}" >&2
    exit 2
  }
done
while [[ ! -s "${run}/COMPLETE.txt" ]]; do
  printf '%s waiting for batch-8 encoder-upgrade fold-0 completion\n' "$(date -Ins)"
  sleep 30
done

audit_args=()
for epoch in $(seq 0 10); do
  epoch4=$(printf '%04d' "${epoch}")
  checkpoint="${run}/checkpoint_epoch_${epoch4}.pt"
  prediction="${run}/validation_predictions_epoch_${epoch4}.npz"
  output="${audits}/epoch_${epoch4}.json"
  for required in "${checkpoint}" "${prediction}"; do
    [[ -s "${required}" ]] || {
      echo "missing completed epoch artifact: ${required}" >&2
      exit 3
    }
  done
  "${python_bin}" -u "${root}/audit_focus_fold_epoch_sources.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --prediction "${prediction}" --folds 5 --fold 0 --seed 20260823 \
    --epoch "${epoch}" --output "${output}" \
    > "${audits}/epoch_${epoch4}.log" 2>&1
  audit_args+=(--audit "${output}")
done
"${python_bin}" -u "${root}/select_focus_single_fold_routes.py" \
  "${audit_args[@]}" --expected-epochs 11 --output "${candidate_routes}" \
  > "${analysis}/route_candidate.log" 2>&1
"${python_bin}" -u "${root}/select_focus_encoder_upgrade_fold0_routes.py" \
  --plan "${plan}" --candidate-routes "${candidate_routes}" \
  --frozen-routes "${frozen_routes}" --output "${decision}" \
  > "${analysis}/route_select.log" 2>&1

"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "qualified_routes":decision["qualified_routes"],
 "route_decision":str(source.resolve()),
 "route_decision_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

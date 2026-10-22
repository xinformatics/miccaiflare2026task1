#!/usr/bin/env bash
set -euo pipefail

# CPU-only exact707 source-family checkpoint-routing audit. It cannot read
# public/hidden data, run deployment inference, package, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${train}/patches_bbox24_ss_min5_v4_cell48/index.csv"
output="${train}/source_classifier_routed_oof"
decision="${output}/DECISION.json"
predictions="${output}/classification_oof.npz"
decoupled="${train}/decoupled_oof"
decoupled_terminal="${decoupled}/TERMINAL.json"
baseline="${decoupled}/five_fold_oof_summary.json"
summary="${output}/five_fold_oof_summary.json"
selection="${output}/checkpoint_selection.json"
terminal="${output}/TERMINAL.json"
source_audits_terminal="${train}/LIVE_SOURCE_AUDITS_COMPLETE.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_source_routed_classifier_oof_20260824.lock"
flock -n 9 || { echo "source-routed classifier audit already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${train}/TRAINING_COMPLETE.json" ]]; do
  if [[ -s "${train}/TRAINING_SKIPPED.json" ]]; then
    "${python_bin}" - "${train}/TRAINING_SKIPPED.json" "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,decision,terminal=map(Path,sys.argv[1:])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "decision":"KEEP_GLOBAL_CLASSIFIER","status":"SKIPPED",
 "reason":"primary exact707 training skipped","source":str(source.resolve()),
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False}
for output in (decision,terminal):
 temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
 temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
    exit 0
  fi
  printf '%s waiting for exact707 primary training terminal\n' "$(date -Ins)"
  sleep 30
done
while [[ ! -s "${source_audits_terminal}" ]]; do
  printf '%s waiting for all 55 exact707 scored-epoch source audits\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${source_audits_terminal}" "${train}/TRAINING_COMPLETE.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit,training=map(Path,sys.argv[1:])
payload=json.loads(audit.read_text(encoding="utf-8"))
expected={(fold,epoch) for fold in range(5) for epoch in (1,4,7,10,13,16,19,22,25,28,30)}
observed={(int(row.get("fold",-1)),int(row.get("epoch",-1))) for row in payload.get("audits",[])}
if (payload.get("status")!="COMPLETE" or payload.get("folds")!=5
    or payload.get("scored_epochs")!=11 or observed!=expected
    or payload.get("training_terminal_sha256")!=hashlib.sha256(training.read_bytes()).hexdigest()
    or payload.get("public_read") is not False or payload.get("hidden_read") is not False
    or payload.get("submission_created") is not False):
    raise SystemExit("live source-audit terminal is incomplete or mismatched")
PY
if [[ ! -s "${decision}" ]]; then
  args=()
  for fold in 0 1 2 3 4; do
    args+=(--run "${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold${fold}_20260823")
  done
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export MKL_THREADING_LAYER=GNU
  "${python_bin}" -u "${root}/analyze_source_routed_classifier_oof.py" \
    --index "${index}" "${args[@]}" --audit-root "${train}" \
    --folds 5 --seed 20260823 --bootstrap 2000 \
    --predictions-output "${predictions}" --output "${decision}" \
    > "${output}/analyze.log" 2>&1
fi
route_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
if [[ "${route_decision}" != QUALIFY_SOURCE_ROUTED_CLASSIFIER ]]; then
  "${python_bin}" - "${decision}" "${route_decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); decision=sys.argv[2]; output=Path(sys.argv[3])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"KEEP_GLOBAL_CLASSIFIER",
 "reason":f"patch OOF decision was {decision}","source":str(source.resolve()),
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
  exit 0
fi
while [[ ! -s "${decoupled_terminal}" ]]; do
  printf '%s waiting for exact707 global decoupled patch OOF\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" -u "${root}/build_source_routed_classifier_patch_oof.py" \
  --index "${index}" --decision "${decision}" --baseline "${baseline}" \
  --output "${summary}" > "${output}/build_summary.log" 2>&1
"${python_bin}" - "${summary}" "${selection}" "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
summary,selection,decision,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload=json.loads(summary.read_text(encoding="utf-8"))["checkpoint_selection"]
temporary=selection.with_name(f".{selection.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,selection)
terminal_payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_CLASSIFIER_ROUTED_FULL_VOLUME",
 "summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "selection":str(selection.resolve()),"selection_sha256":sha(selection),
 "classifier_route_decision":str(decision.resolve()),
 "classifier_route_decision_sha256":sha(decision),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(terminal_payload,indent=2)+"\n");os.replace(temporary,output)
print(json.dumps(terminal_payload,indent=2))
PY

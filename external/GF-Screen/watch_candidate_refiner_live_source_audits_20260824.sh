#!/usr/bin/env bash
set -euo pipefail

# CPU-only live source audit for exact707 folds 1--4. Every scored checkpoint
# is bound to its prediction archive and evaluated only on that fold's grouped
# internal holdout. This watcher performs no inference and cannot read public
# or hidden data, package a submission, or upload anything.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${work}/patches_bbox24_ss_min5_v4_cell48/index.csv"
training_complete="${work}/TRAINING_COMPLETE.json"
training_skipped="${work}/TRAINING_SKIPPED.json"
terminal="${work}/LIVE_SOURCE_AUDITS_COMPLETE.json"
epochs=(1 4 7 10 13 16 19 22 25 28 30)

run_dir() {
  printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' \
    "${root}" "$1"
}

mkdir -p "${work}"
exec 9>"${root}/runs/watch_candidate_refiner_live_source_audits_20260824.lock"
flock -n 9 || { echo "live exact707 source audit watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
[[ -s "${index}" ]] || { echo "missing exact707 index: ${index}" >&2; exit 2; }

audit_is_valid() {
  local audit="$1" fold="$2" epoch="$3"
  "${python_bin}" - "${audit}" "${fold}" "${epoch}" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); fold=int(sys.argv[2]); epoch=int(sys.argv[3])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
payload=json.loads(path.read_text(encoding="utf-8"))
required={"mature","weak_sources","lung_cells"}
valid=(payload.get("fold")==fold and payload.get("epoch")==epoch
       and payload.get("cases",0)>0 and payload.get("patches",0)>0
       and required <= set(payload.get("source_classification_groups",{}))
       and required <= set(payload.get("source_segmentation_groups",{}))
       and payload.get("public_read") is False
       and payload.get("hidden_read") is False
       and payload.get("submission_created") is False)
raise SystemExit(0 if valid else 1)
PY
}

audit_available() {
  local fold="$1" epoch="$2" padded run prediction checkpoint output log
  printf -v padded '%04d' "${epoch}"
  run="$(run_dir "${fold}")"
  prediction="${run}/validation_predictions_epoch_${padded}.npz"
  checkpoint="${run}/checkpoint_epoch_${padded}.pt"
  output="${work}/fold${fold}_source_epoch${epoch}.json"
  log="${work}/fold${fold}_source_epoch${epoch}.log"
  [[ -s "${prediction}" && -s "${checkpoint}" ]] || return 0
  if audit_is_valid "${output}" "${fold}" "${epoch}"; then
    return 0
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_fold_sources.py" \
    --index "${index}" --prediction "${prediction}" --checkpoint "${checkpoint}" \
    --folds 5 --fold "${fold}" --seed 20260823 --epoch "${epoch}" \
    --output "${output}" > "${log}" 2>&1
  audit_is_valid "${output}" "${fold}" "${epoch}"
}

sweep() {
  local fold epoch
  for fold in 1 2 3 4; do
    for epoch in "${epochs[@]}"; do
      audit_available "${fold}" "${epoch}"
    done
  done
}

while [[ ! -s "${training_complete}" && ! -s "${training_skipped}" ]]; do
  sweep
  sleep 30
done
sweep

"${python_bin}" - "${work}" "${training_complete}" "${training_skipped}" \
  "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
root,complete,skipped,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if skipped.is_file():
    payload={
      "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
      "status":"SKIPPED","reason":"exact707 primary training skipped",
      "training_terminal":str(skipped.resolve()),"training_terminal_sha256":sha(skipped),
      "public_read":False,"hidden_read":False,"submission_created":False,
    }
else:
    if not complete.is_file(): raise SystemExit("missing training terminal")
    records=[]
    for fold in range(5):
      for epoch in (1,4,7,10,13,16,19,22,25,28,30):
        path=root/f"fold{fold}_source_epoch{epoch}.json"
        if not path.is_file():
            raise SystemExit(f"missing source audit: {path}")
        record=json.loads(path.read_text(encoding="utf-8"))
        if record.get("fold")!=fold or record.get("epoch")!=epoch:
            raise SystemExit(f"source audit identity mismatch: {path}")
        records.append({"fold":fold,"epoch":epoch,"path":str(path.resolve()),
                        "sha256":sha(path)})
    payload={
      "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
      "status":"COMPLETE","audits":records,"folds":5,"scored_epochs":11,
      "training_terminal":str(complete.resolve()),"training_terminal_sha256":sha(complete),
      "public_read":False,"hidden_read":False,"submission_created":False,
    }
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

#!/usr/bin/env bash
set -euo pipefail

# Prepare and audit the exact-210 OOF candidate caches while fold 4 trains.
# This is checkpoint-independent and reads no public or hidden data.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823"
staging="${output}/staging"
terminal="${output}/PREWARM_COMPLETE.json"

mkdir -p "${output}"
exec 9>"${root}/runs/prewarm_candidate_refiner_v2_oof_caches_20260823.lock"
flock -n 9 || { echo "OOF cache prewarm already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
[[ -s "${staging}/STAGING_COMPLETE.json" ]] || {
  echo "exact-210 staging is not complete" >&2
  exit 2
}

prepare_fold() {
  local fold="$1" fold_stage="${staging}/fold${fold}"
  local work="${output}/fold${fold}/work_bbox24"
  mkdir -p "${work}"
  exec 8>"${work}/PREWARM.lock"
  flock 8
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${fold_stage}/images" \
      --proposal "epoch105=${fold_stage}/proposals" \
      --base "${fold_stage}/proposals" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 \
      --bbox-margin-mm 24 --tile-stride-mm 72 \
      > "${output}/fold${fold}.prepare.log" 2>&1
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    > "${output}/fold${fold}.audit.log" 2>&1
}

prepare_sequence() {
  local fold
  for fold in "$@"; do prepare_fold "${fold}"; done
}
prepare_sequence 0 2 4 & pid0=$!
prepare_sequence 1 3 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

"${python_bin}" - "${output}" "${terminal}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
root,output=map(Path,sys.argv[1:])
folds=[]
for fold in range(5):
    work=root/f"fold{fold}"/"work_bbox24"
    for required in (work/"PREPARE_COMPLETE.txt",work/"AUDIT.json",work/"manifest.json"):
        if not required.is_file(): raise SystemExit(f"missing cache asset: {required}")
    audit=json.loads((work/"AUDIT.json").read_text(encoding="utf-8"))
    folds.append({"fold":fold,"work":str(work.resolve()),"audit":audit})
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","folds":folds,"checkpoint_independent":True,
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
PY

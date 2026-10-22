#!/usr/bin/env bash
set -euo pipefail

# CPU-only source-stratified audit watcher for the exact primary fold 0.
# It reads only internal held-out labels/predictions and creates no inference,
# public evaluation, packaging, network request, submission, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${work}/patches_bbox24_ss_min5_v4_cell48/index.csv"
run="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
terminal="${work}/FOLD0_SOURCE_AUDITS_COMPLETE.json"

mkdir -p "${work}"
exec 9>"${root}/runs/watch_candidate_refiner_fold0_source_audits_20260824.lock"
flock -n 9 || { echo "fold-0 source audit watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

audit_available() {
  local epoch="$1" padded prediction checkpoint output
  printf -v padded '%04d' "${epoch}"
  prediction="${run}/validation_predictions_epoch_${padded}.npz"
  checkpoint="${run}/checkpoint_epoch_${padded}.pt"
  output="${work}/fold0_source_epoch${epoch}.json"
  [[ -s "${prediction}" && -s "${checkpoint}" ]] || return 0
  if [[ -s "${output}" ]] && "${python_bin}" - "${output}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
required={"mature","weak_sources","lung_cells"}
raise SystemExit(0 if required <= set(p.get("source_segmentation_groups",{})) else 1)
PY
  then
    return 0
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_fold_sources.py" \
    --index "${index}" --prediction "${prediction}" --checkpoint "${checkpoint}" \
    --folds 5 --fold 0 --seed 20260823 --epoch "${epoch}" --output "${output}" \
    > "${work}/fold0_source_epoch${epoch}.log" 2>&1
}

while :; do
  for epoch in 1 4 7 10 13 16 19 22 25 28 30; do audit_available "${epoch}"; done
  # The legacy fold-0 process writes COMPLETE.txt before the compatibility
  # watcher can freeze its final scored epoch.  Do not seal the source audit
  # in that short interval: epoch 30 must exist when training is complete.
  if [[ -s "${run}/COMPLETE.txt" \
      && -s "${run}/checkpoint_epoch_0030.pt" \
      && -s "${run}/validation_predictions_epoch_0030.npz" ]]; then
    break
  fi
  sleep 30
done
for epoch in 1 4 7 10 13 16 19 22 25 28 30; do audit_available "${epoch}"; done
audit_args=()
for epoch in 1 4 7 10 13 16 19 22 25 28 30; do
  audit="${work}/fold0_source_epoch${epoch}.json"
  [[ -s "${audit}" ]] || { echo "missing required source audit: ${audit}" >&2; exit 3; }
  audit_args+=(--audit "${audit}")
done
current_selection="${work}/fold0_currentgf_checkpoint_selection.json"
epoch_selection="${work}/fold0_epoch105_checkpoint_selection.json"
"${python_bin}" -u "${root}/select_candidate_refiner_source_checkpoint.py" \
  "${audit_args[@]}" --source-group currentgf \
  --checkpoint-output "${run}/checkpoint_best_currentgf_segmentation.pt" \
  --prediction-output "${run}/validation_predictions_best_currentgf_segmentation.npz" \
  --output "${current_selection}" > "${work}/fold0_currentgf_checkpoint_selection.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_source_checkpoint.py" \
  "${audit_args[@]}" --source-group epoch105 \
  --checkpoint-output "${run}/checkpoint_best_epoch105_segmentation.pt" \
  --prediction-output "${run}/validation_predictions_best_epoch105_segmentation.npz" \
  --output "${epoch_selection}" > "${work}/fold0_epoch105_checkpoint_selection.log" 2>&1
"${python_bin}" - "${run}/COMPLETE.txt" "${work}" "${current_selection}" \
  "${epoch_selection}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
training,root,current_selection,epoch_selection,output=map(Path,sys.argv[1:]); records=[]
for epoch in (1,4,7,10,13,16,19,22,25,28,30):
    path=root/f"fold0_source_epoch{epoch}.json"
    if path.is_file():
        records.append({"epoch":epoch,"path":str(path.resolve()),
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","training_terminal":str(training.resolve()),"audits":records,
 "currentgf_checkpoint_selection":str(current_selection.resolve()),
 "currentgf_checkpoint_selection_sha256":hashlib.sha256(current_selection.read_bytes()).hexdigest(),
 "epoch105_checkpoint_selection":str(epoch_selection.resolve()),
 "epoch105_checkpoint_selection_sha256":hashlib.sha256(epoch_selection.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

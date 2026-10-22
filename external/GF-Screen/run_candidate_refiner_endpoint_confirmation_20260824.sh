#!/usr/bin/env bash
set -euo pipefail

# Conditional exact-707 confirmation above the frozen 0.95 contour ceiling.
# This is an anchor-currentGF, label-internal OOF audit only. It has no public,
# hidden, packaging, submission, or upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
anchor="${primary}/anchor_currentgf"
work="${anchor}/endpoint_confirmation"
plan="${primary}/ENDPOINT_CONFIRMATION_PLAN.json"
primary_complete="${primary}/FULL_VOLUME_OOF_COMPLETE.json"
primary_skipped="${primary}/FULL_VOLUME_OOF_SKIPPED.json"
anchor_selection="${anchor}/SEGMENTATION_THRESHOLD_SELECTION.json"
anchor_sweep="${anchor}/segmentation_threshold_sweep.tsv"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
trigger="${work}/TRIGGER.json"
grid="${work}/grid.json"
decision="${work}/DECISION.json"
terminal="${work}/TERMINAL.json"

mkdir -p "${work}"
exec 9>"${root}/runs/run_candidate_refiner_endpoint_confirmation_20260824.lock"
flock -n 9 || { echo "endpoint confirmation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${primary_complete}" && ! -s "${primary_skipped}" ]]; do
  printf '%s waiting for exact-707 full-volume terminal\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${primary_skipped}" ]]; then
  "${python_bin}" - "${primary_skipped}" "${terminal}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"SKIP_ENDPOINT_CONFIRMATION",
 "reason":"primary exact-707 full-volume chain skipped",
 "primary_terminal":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,"upload_authorized":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
  exit 0
fi

for required in "${plan}" "${anchor_selection}" "${anchor_sweep}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing endpoint input: ${required}" >&2; exit 2; }
done
"${python_bin}" -u "${root}/select_candidate_refiner_endpoint_confirmation.py" trigger \
  --plan "${plan}" --selection "${anchor_selection}" --sweep "${anchor_sweep}" \
  --output "${trigger}" > "${work}/trigger.log" 2>&1
trigger_decision=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${trigger}")
if [[ "${trigger_decision}" != RUN_ENDPOINT_CONFIRMATION ]]; then
  "${python_bin}" - "${trigger}" "${terminal}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload=json.loads(source.read_text(encoding="utf-8"))
payload["status"]="COMPLETE"; payload["created_at_terminal"]=datetime.now().astimezone().isoformat()
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
  exit 0
fi

"${python_bin}" - "${grid}" <<'PY'
import json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); configurations=[]
for threshold in (0.96,0.97,0.98):
 tag=f"weak707_endpoint_c0p75_s{threshold:.2f}_refined".replace(".","p")
 configurations.append({"tag":tag,"classification_threshold":0.75,
  "segmentation_threshold":threshold,"mode":"refined",
  "source_classification_thresholds":{},"source_segmentation_thresholds":{}})
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps({"schema_version":1,"configurations":configurations},indent=2)+"\n")
os.replace(temporary,output)
PY

status=0
for fold_group in "0 1 2" "3 4"; do
  pids=()
  for fold in ${fold_group}; do
    fold_work="${primary}/fold${fold}/work_bbox24"
    base="${primary}/staging/fold${fold}/proposals/currentgf"
    validation_log="${work}/fold${fold}.reuse.log"
    validator_args=(
      --configurations "${grid}" --output-root "${work}" --fold "${fold}"
      --expected-dir "${primary}/staging/fold${fold}/labels" --base "${base}"
      --segmentation-threshold 0.95 --min-seed-coverage 0
      --min-refined-overlap 0 --min-volume-ratio 0
      --max-volume-ratio 1000000000000
    )
    if "${python_bin}" -u "${root}/validate_candidate_refiner_grid_fold.py" \
        "${validator_args[@]}" > "${validation_log}" 2>&1; then
      printf '%s reusing endpoint fold %s\n' "$(date -Ins)" "${fold}"
      continue
    fi
    "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
      --work "${fold_work}" --base "${base}" --output-root "${work}" \
      --configurations "${grid}" --fold "${fold}" --segmentation-threshold 0.95 \
      --min-seed-coverage 0 --min-refined-overlap 0 --min-volume-ratio 0 \
      --max-volume-ratio 1000000000000 \
      > "${work}/fold${fold}.materialize.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

for threshold in 0.96 0.97 0.98; do
  threshold_tag=${threshold/./p}
  tag="weak707_endpoint_c0p75_s${threshold_tag}_refined"
  combined="${work}/${tag}/combined"
  "${python_bin}" - "${work}/${tag}" "${combined}" \
    "${primary}/staging/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); combined.mkdir(parents=True,exist_ok=True); seen={}
for fold in range(5):
 for source in (root/f"fold{fold}").glob("*.nii.gz"):
  if source.name in seen: raise SystemExit(f"duplicate case: {source.name}")
  target=combined/source.name
  if target.is_symlink() or target.exists():
   if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
  else: target.symlink_to(source.resolve(strict=True))
  seen[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=707 or set(seen)!=expected: raise SystemExit("endpoint OOF coverage mismatch")
PY
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined}" --gt "${primary}/staging/all/labels" \
    --csv "${work}/${tag}.metrics.csv" --workers 6 \
    > "${work}/${tag}.score.log" 2>&1
done

"${python_bin}" -u "${root}/select_candidate_refiner_endpoint_confirmation.py" select \
  --plan "${plan}" --trigger "${trigger}" --manifest "${manifest}" \
  --metrics "0.95=${anchor}/weak707_segcal_c0p75_s0p95_refined.metrics.csv" \
  --metrics "0.96=${work}/weak707_endpoint_c0p75_s0p96_refined.metrics.csv" \
  --metrics "0.97=${work}/weak707_endpoint_c0p75_s0p97_refined.metrics.csv" \
  --metrics "0.98=${work}/weak707_endpoint_c0p75_s0p98_refined.metrics.csv" \
  --bootstrap 20000 --seed 20260824 --output "${decision}" \
  > "${work}/select.log" 2>&1

"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload=json.loads(source.read_text(encoding="utf-8"))
payload["status"]="COMPLETE"; payload["created_at_terminal"]=datetime.now().astimezone().isoformat()
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
cat "${terminal}"

#!/usr/bin/env bash
set -euo pipefail

# CPU-only exact-707 patch-OOF composition for independently selected
# confidence and contour checkpoints. No public/hidden inference or packaging.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
training_complete="${train}/TRAINING_COMPLETE.json"
training_skipped="${train}/TRAINING_SKIPPED.json"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
baseline="${train}/five_fold_oof_summary.json"
output="${train}/decoupled_oof"
summary="${output}/five_fold_oof_summary.json"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v4_weak707_decoupled_oof_20260823.lock"
flock -n 9 || { echo "exact-707 decoupled OOF controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${training_complete}" && ! -s "${training_skipped}" ]]; do
  printf '%s waiting for exact-707 five-fold training terminal\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${training_skipped}" ]]; then
  "${python_bin}" - "${training_skipped}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_COUPLED","reason":"primary exact-707 training skipped",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
for required in "${patches}/COMPLETE.txt" "${patches}/index.csv" \
  "${patches}/AUDIT.json" "${baseline}"; do [[ -s "${required}" ]]; done

summary_args=()
for fold in 0 1 2 3 4; do
  run="$(fold_dir "${fold}")"
  fold_output="${output}/fold${fold}"
  mkdir -p "${fold_output}"
  class_checkpoint="${run}/checkpoint_best_classification.pt"
  contour_checkpoint="${run}/checkpoint_best_segmentation.pt"
  class_predictions="${run}/validation_predictions_best_classification.npz"
  contour_predictions="${run}/validation_predictions_best_segmentation.npz"
  composed="${fold_output}/validation_predictions_decoupled.npz"
  audit="${fold_output}/COMPOSITION_AUDIT.json"
  for required in "${class_checkpoint}" "${contour_checkpoint}" \
    "${class_predictions}" "${contour_predictions}"; do [[ -s "${required}" ]]; done
  "${python_bin}" -u "${root}/compose_candidate_refiner_validation_predictions.py" \
    --classification "${class_predictions}" --contour "${contour_predictions}" \
    --classification-checkpoint "${class_checkpoint}" \
    --contour-checkpoint "${contour_checkpoint}" \
    --output "${composed}" --audit "${audit}" \
    > "${fold_output}/compose.log" 2>&1
  summary_args+=(--checkpoint "${class_checkpoint}")
  summary_args+=(--contour-checkpoint "${contour_checkpoint}")
  summary_args+=(--prediction "${composed}")
  summary_args+=(--composition-audit "${audit}")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${summary_args[@]}" --output "${summary}" \
  --source-group mature=epoch105,currentgf \
  --source-group weak_sources=autopet_weak,lndb_weak \
  --source-group lung_cells=lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7 \
  > "${output}/summarize.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_decoupled_oof.py" \
  --baseline "${baseline}" --candidate "${summary}" --output "${decision}" \
  > "${output}/select.log" 2>&1
"${python_bin}" - "${training_complete}" "${summary}" "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
upstream,summary,decision,output=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","upstream":str(upstream.resolve()),
 "summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "decision":str(decision.resolve()),"decision_payload":json.loads(decision.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

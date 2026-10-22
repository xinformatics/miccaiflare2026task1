#!/usr/bin/env bash
set -euo pipefail

# CPU-only composition and five-fold patch-OOF gate for independently selected
# candidate-confidence and contour checkpoints. No public/hidden inference.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_union_native_training_20260823"
terminal="${train}/fivefold/UNION_NATIVE_FIVEFOLD_TERMINAL.json"
prep="${train}/PREPARATION_TERMINAL.json"
output="${train}/decoupled_oof"
summary="${output}/five_fold_oof_summary.json"
decision="${output}/DECISION.json"
complete="${output}/TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_union_native_b32_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_union_native_decoupled_oof_20260823.lock"
flock -n 9 || { echo "native decoupled OOF controller already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${terminal}" ]]; do
  printf '%s waiting for union-native five-fold terminal\n' "$(date -Ins)"
  sleep 30
done
status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
[[ "${status}" == COMPLETE ]] || { echo "native five-fold did not complete" >&2; exit 2; }
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep}")
[[ -s "${index}" ]]

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
  --index "${index}" "${summary_args[@]}" --output "${summary}" \
  > "${output}/summarize.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_decoupled_oof.py" \
  --baseline "${train}/fivefold/five_fold_oof_summary.json" \
  --candidate "${summary}" --output "${decision}" \
  > "${output}/select.log" 2>&1
"${python_bin}" - "${terminal}" "${summary}" "${decision}" "${complete}" <<'PY'
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

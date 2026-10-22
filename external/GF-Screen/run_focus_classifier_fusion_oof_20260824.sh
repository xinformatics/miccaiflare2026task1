#!/usr/bin/env bash
set -euo pipefail

# CPU-only stacked candidate-classification audit. It cross-fits a fixed L2
# logistic model from B32 probability, inference-visible Focus contour
# confidence, proposal geometry/support, and source family. Inputs are grouped
# held-out exact707 archives only; no public/hidden data or deployment occurs.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
variant="${1:-}"
primary="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${primary}/patches_bbox24_ss_min5_v4_cell48/index.csv"

case "${variant}" in
  frozen)
    focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
    focus_run_prefix="${root}/runs/focus_dualwindow_cell707_release10_fold"
    focus_run_suffix="20260823"
    output="${root}/analysis/focus_classifier_fusion_frozen_20260824"
    ;;
  discriminative)
    focus_train="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
    focus_run_prefix="${root}/runs/focus_dualwindow_disc01_release10_fold"
    focus_run_suffix="20260824"
    output="${root}/analysis/focus_classifier_fusion_disc01_20260824"
    ;;
  *) echo "usage: $0 frozen|discriminative" >&2; exit 2 ;;
esac
focus_terminal="${focus_train}/FIVEFOLD_TERMINAL.json"
decision="${output}/DECISION.json"
predictions="${output}/classification_oof.npz"
lock="${root}/runs/run_focus_classifier_fusion_${variant}_20260824.lock"

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "${variant} Focus classifier-fusion audit already active" >&2; exit 1; }
[[ -s "${decision}" ]] && exit 0
while [[ ! -s "${focus_terminal}" ]]; do
  printf '%s waiting for %s Focus five-fold terminal\n' "$(date -Ins)" "${variant}"
  sleep 30
done
focus_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${focus_terminal}")
if [[ "${focus_decision}" != PROMOTE_TO_FULL_VOLUME_OOF ]]; then
  "${python_bin}" - "${variant}" "${focus_terminal}" "${focus_decision}" \
    "${decision}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
variant=sys.argv[1]; source=Path(sys.argv[2]); observed=sys.argv[3]; output=Path(sys.argv[4])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","variant":variant,"decision":"KEEP_B32_CLASSIFIER",
 "reason":f"Focus five-fold decision was {observed}","source":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
while [[ ! -s "${primary}/TRAINING_COMPLETE.json" ]]; do
  printf '%s waiting for exact707 primary training terminal\n' "$(date -Ins)"
  sleep 30
done

args=()
for fold in 0 1 2 3 4; do
  b32="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold${fold}_20260823/validation_predictions_best_classification.npz"
  focus="${focus_run_prefix}${fold}_${focus_run_suffix}/validation_predictions_best.npz"
  [[ -s "${b32}" && -s "${focus}" ]] || {
    echo "missing matched fold ${fold} classifier-fusion input" >&2; exit 2;
  }
  args+=(--b32-prediction "${b32}" --focus-prediction "${focus}")
done
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MKL_THREADING_LAYER=GNU
"${python_bin}" -u "${root}/analyze_focus_classifier_fusion_oof.py" \
  --index "${index}" "${args[@]}" --folds 5 --seed 20260823 \
  --bootstrap 2000 --predictions-output "${predictions}" \
  --output "${decision}" > "${output}/analyze.log" 2>&1
cat "${decision}"

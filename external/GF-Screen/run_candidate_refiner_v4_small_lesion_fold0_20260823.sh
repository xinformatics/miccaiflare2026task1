#!/usr/bin/env bash
set -euo pipefail

# Matched exact-707 fold-0 small-lesion loss ablation. It waits until the
# unchanged five-fold primary run is complete, then uses one genuinely free
# GPU. It performs no public/hidden inference, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${work}/patches_bbox24_ss_min5_v4_cell48"
source_audit="${work}/SOURCE_DOMAIN_AUDIT.json"
primary_complete="${work}/TRAINING_COMPLETE.json"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
tta_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_SAVE_TERMINAL.json"
tta_fallback_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_FALLBACK_SAVE_TERMINAL.json"
focus_terminal="${root}/public_sweeps/focus_dualwindow_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
prompt_pool_terminal="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_20260824/FOLD0_TERMINAL.json"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
output="${root}/runs/candidate_refiner_v4_cell707_b32_t05_smallloss30_fold0_20260823"
decision_dir="${root}/analysis/candidate_refiner_v4_cell707_small_loss_20260823"
decision="${decision_dir}/FOLD0_DECISION.json"
terminal="${decision_dir}/FOLD0_TERMINAL.json"
gpu="${SMALL_LESION_GPU:-1}"

mkdir -p "${output}/matplotlib" "${decision_dir}"
exec 9>"${root}/runs/run_candidate_refiner_v4_small_lesion_fold0_20260823.lock"
flock -n 9 || { echo "small-lesion fold-0 ablation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${primary_complete}" || ! -s "${baseline}/COMPLETE.txt" \
    || ! -s "${primary_pipeline_complete}" || ! -s "${tta_terminal}" \
    || ! -s "${tta_fallback_terminal}" \
    || ! -s "${focus_terminal}" || ! -s "${prompt_pool_terminal}" ]]; do
  printf '%s waiting for exact-707 save-only terminal and complete higher-priority Focus/prompt-pool chains\n' "$(date -Ins)"
  sleep 30
done
for required in \
  "${patches}/COMPLETE.txt" "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${source_audit}" "${baseline}/history.json" \
  "${baseline}/validation_predictions_best_segmentation.npz"; do
  [[ -s "${required}" ]] || { echo "missing prerequisite: ${required}" >&2; exit 3; }
done

"${python_bin}" - "${patches}/index.csv" "${source_audit}" <<'PY'
import hashlib,json,sys
from pathlib import Path
index,audit=map(Path,sys.argv[1:])
payload=json.loads(audit.read_text(encoding="utf-8"))
if payload.get("index_sha256") != hashlib.sha256(index.read_bytes()).hexdigest():
    raise SystemExit("source audit/index identity mismatch")
if payload.get("recommendations",{}).get("test_small_lesion_loss") is not True:
    raise SystemExit("source audit does not recommend the small-lesion ablation")
PY

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
  "${python_bin}" -u "${root}/train_candidate_refiner.py" \
  --index "${patches}/index.csv" --output "${output}" \
  --folds 5 --fold 0 --epochs 30 --batch-size 8 --workers 2 \
  --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
  --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260823 \
  --segmentation-threshold-max 0.95 --validation-interval 3 \
  --validation-batch-size 32 \
  --small-lesion-loss --small-lesion-positive-weight-cap 30 \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_v4_cell707_b32_t05_smallloss30_fold0 \
  --resume > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_small_lesion.py" \
  --index "${patches}/index.csv" --source-audit "${source_audit}" \
  --baseline-history "${baseline}/history.json" \
  --candidate-history "${output}/history.json" \
  --baseline-prediction "${baseline}/validation_predictions_best_segmentation.npz" \
  --candidate-prediction "${output}/validation_predictions_best_segmentation.npz" \
  --folds 5 --fold 0 --seed 20260823 \
  --minimum-overall-segmentation-gain 0.005 \
  --minimum-small-source-gain 0.010 \
  --minimum-composite-gain 0.0 --maximum-classification-drop 0.010 \
  --output "${decision}" > "${decision_dir}/select.log" 2>&1

"${python_bin}" - "${patches}/index.csv" "${source_audit}" \
  "${baseline}/history.json" "${output}/history.json" "${decision}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
index,audit,baseline,candidate,decision,output=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","experiment":"matched exact-707 fold-0 small-lesion loss cap30",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "source_audit":str(audit.resolve()),"source_audit_sha256":sha(audit),
 "baseline_history":str(baseline.resolve()),"baseline_history_sha256":sha(baseline),
 "candidate_history":str(candidate.resolve()),"candidate_history_sha256":sha(candidate),
 "decision":str(decision.resolve()),"decision_payload":json.loads(decision.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

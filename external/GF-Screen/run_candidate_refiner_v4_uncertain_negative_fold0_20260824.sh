#!/usr/bin/env bash
set -euo pipefail

# Matched fold-0 test of asymmetric annotation-negative classification loss.
# This is a training-only gate: it cannot read public/hidden data or package.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
primary_complete="${train}/TRAINING_COMPLETE.json"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
source_condition_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_source_conditioning_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
smallloss_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_small_loss_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
output="${root}/runs/candidate_refiner_v4_cell707_b32_t05_uneg025_fold0_20260824"
decision_dir="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_20260824"
decision="${decision_dir}/FOLD0_DECISION.json"
terminal="${decision_dir}/FOLD0_TERMINAL.json"
gpu="${UNCERTAIN_NEGATIVE_GPU:-1}"

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }
mkdir -p "${output}/matplotlib" "${decision_dir}"
exec 9>"${root}/runs/run_candidate_refiner_v4_uncertain_negative_fold0_20260824.lock"
flock -n 9 || { echo "uncertain-negative fold-0 ablation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${primary_complete}" || ! -s "${primary_pipeline_complete}" \
    || ! -s "${source_condition_terminal}" || ! -s "${smallloss_terminal}" ]]; do
  printf '%s waiting for exact-707 primary and higher-priority ablation terminals\n' "$(date -Ins)"
  sleep 30
done
for required in "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${baseline}/COMPLETE.txt" "${baseline}/history.json"; do
  [[ -s "${required}" ]] || { echo "missing prerequisite: ${required}" >&2; exit 3; }
done
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
  --validation-batch-size 32 --classification-negative-weight 0.25 \
  --negative-segmentation-weight 0.025 \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_v4_cell707_b32_t05_uneg025_fold0 \
  --resume > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_uncertain_negative.py" \
  --baseline-history "${baseline}/history.json" \
  --candidate-history "${output}/history.json" \
  --output "${decision}" > "${decision_dir}/select.log" 2>&1

"${python_bin}" - "${patches}/index.csv" "${baseline}/history.json" \
  "${output}/history.json" "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,baseline,candidate,decision,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","variant":"uncertain_negative_025",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "baseline_history":str(baseline.resolve()),"baseline_history_sha256":sha(baseline),
 "candidate_history":str(candidate.resolve()),"candidate_history_sha256":sha(candidate),
 "decision":json.loads(decision.read_text()),"decision_path":str(decision.resolve()),
 "classification_negative_weight":0.25,
 "negative_segmentation_weight":0.025,
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

#!/usr/bin/env bash
set -euo pipefail

# Controlled fold-0 contour-specialist pilot. Optimizer batches contain only
# weak AutoPET, weak LNDb, and pooled lung-cell proposals. Held-out validation
# still covers every proposal source. This script performs no public/hidden
# inference, packaging, network request, submission, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
primary_complete="${train}/TRAINING_COMPLETE.json"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
decoupled_chain_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_decoupled_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
source_routed_chain_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_source_routed_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
tta_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_SAVE_TERMINAL.json"
tta_fallback_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_FALLBACK_SAVE_TERMINAL.json"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
output="${root}/runs/candidate_refiner_v4_cell707_b32_t05_discovery_fold0_20260824"
work="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_20260824"
decision="${work}/FOLD0_DECISION.json"
terminal="${work}/FOLD0_TERMINAL.json"
gpu="${DISCOVERY_SPECIALIST_GPU:-0}"
allow_early_gpu="${DISCOVERY_SPECIALIST_ALLOW_EARLY_GPU:-0}"

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }
[[ "${allow_early_gpu}" == 0 || "${allow_early_gpu}" == 1 ]] || {
  echo "invalid early-GPU dispatch flag: ${allow_early_gpu}" >&2
  exit 2
}
mkdir -p "${output}/matplotlib" "${work}"
exec 9>"${root}/runs/run_candidate_refiner_v4_discovery_specialist_fold0_20260824.lock"
flock -n 9 || { echo "discovery-specialist fold-0 pilot already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

# The original ordering conserves GPU-hours by waiting for every higher-
# priority deployment chain.  An explicit early mode may use an otherwise idle
# GPU once the immutable primary training and TTA evidence are complete.  It
# changes scheduling only: inputs, optimizer recipe, held-out fold, and the
# predeclared promotion gate remain identical.  Keep GPU 0 free for primary
# public/hidden inference when using this mode.
if [[ "${allow_early_gpu}" == 1 ]]; then
  while [[ ! -s "${primary_complete}" || ! -s "${tta_terminal}" \
      || ! -s "${tta_fallback_terminal}" ]]; do
    printf '%s early dispatch waiting for primary training and TTA terminals\n' \
      "$(date -Ins)"
    sleep 30
  done
else
  while [[ ! -s "${primary_complete}" || ! -s "${primary_pipeline_complete}" \
      || ! -s "${tta_terminal}" || ! -s "${tta_fallback_terminal}" \
      || ! -s "${decoupled_chain_terminal}" \
      || ! -s "${source_routed_chain_terminal}" ]]; do
    printf '%s waiting for exact-707 save-only, TTA, decoupled, and source-routed chain terminals\n' "$(date -Ins)"
    sleep 30
  done
fi
for required in "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${baseline}/COMPLETE.txt" "${baseline}/checkpoint_best_cohort_segmentation.pt" \
  "${baseline}/validation_predictions_best_cohort_segmentation.npz"; do
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
  --validation-batch-size 32 \
  --train-source-group autopet_weak \
  --train-source-group lndb_weak \
  --train-source-group lung_cells \
  --segmentation-selection-source-group autopet_weak \
  --segmentation-selection-source-group lndb_weak \
  --segmentation-selection-source-group lung_cells \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_v4_cell707_discovery_fold0 \
  --resume > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_discovery_specialist.py" \
  --index "${patches}/index.csv" \
  --baseline-checkpoint "${baseline}/checkpoint_best_cohort_segmentation.pt" \
  --baseline-prediction "${baseline}/validation_predictions_best_cohort_segmentation.npz" \
  --candidate-checkpoint "${output}/checkpoint_best_routed_segmentation.pt" \
  --candidate-prediction "${output}/validation_predictions_best_routed_segmentation.npz" \
  --folds 5 --fold 0 --seed 20260823 \
  --minimum-combined-gain 0.010 --maximum-group-drop 0.005 \
  --output "${decision}" > "${work}/select.log" 2>&1

"${python_bin}" - "${patches}/index.csv" "${baseline}/history.json" \
  "${output}/history.json" "${decision}" "${allow_early_gpu}" \
  "${gpu}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,baseline,candidate,decision=map(Path,sys.argv[1:5])
allow_early=bool(int(sys.argv[5])); gpu=int(sys.argv[6]); output=Path(sys.argv[7])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","experiment":"exact-707 fold-0 discovery contour specialist",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "baseline_history":str(baseline.resolve()),"baseline_history_sha256":sha(baseline),
 "candidate_history":str(candidate.resolve()),"candidate_history_sha256":sha(candidate),
 "decision":str(decision.resolve()),"decision_payload":json.loads(decision.read_text()),
 "training_source_groups":["autopet_weak","lndb_weak","lung_cells"],
 "validation_scope":"all held-out proposal sources",
 "classification_route":"primary","mature_contour_route":"primary",
 "discovery_contour_route":"specialist",
 "scheduling":{"early_gpu_dispatch":allow_early,"physical_gpu":gpu,
               "model_recipe_changed":False},
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

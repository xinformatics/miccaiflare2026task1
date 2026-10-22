#!/usr/bin/env bash
set -euo pipefail

# Matched fold-0 test of prompt-masked classification pooling. This is an
# internal training gate only: it cannot read public/hidden cases or package.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${work}/patches_bbox24_ss_min5_v4_cell48"
primary_complete="${work}/TRAINING_COMPLETE.json"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
tta_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_SAVE_TERMINAL.json"
tta_fallback_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_FALLBACK_SAVE_TERMINAL.json"
focus_terminal="${root}/public_sweeps/focus_dualwindow_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
output="${root}/runs/candidate_refiner_v4_cell707_b32_t05_promptpool_fold0_20260823"
decision_dir="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_20260824"
baseline_audit="${decision_dir}/baseline_best_classification_source.json"
candidate_audit="${decision_dir}/candidate_best_classification_source.json"
superseded_plan="${decision_dir}/PLAN_V2.json"
plan="${decision_dir}/PLAN_V3.json"
decision="${decision_dir}/FOLD0_DECISION.json"
terminal="${decision_dir}/FOLD0_TERMINAL.json"
gpu="${PROMPT_POOL_GPU:-1}"

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }
mkdir -p "${output}/matplotlib" "${decision_dir}"
exec 9>"${root}/runs/run_candidate_refiner_v4_prompt_pool_fold0_20260824.lock"
flock -n 9 || { echo "prompt-pool fold-0 ablation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

# Freeze all model-selection inputs and gates before an optimizer can run.
for required in "${patches}/index.csv" "${baseline}/history.json" \
  "${baseline}/checkpoint_best_classification.pt" \
  "${baseline}/validation_predictions_best_classification.npz"; do
  [[ -s "${required}" ]] || { echo "missing plan input: ${required}" >&2; exit 3; }
done
"${python_bin}" - "${patches}/index.csv" "${baseline}/history.json" \
  "${baseline}/checkpoint_best_classification.pt" \
  "${baseline}/validation_predictions_best_classification.npz" \
  "${superseded_plan}" "${plan}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,history,checkpoint,prediction,superseded,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if not superseded.is_file():
 raise SystemExit("missing superseded prompt-pool plan revision")
expected={
 "schema_version":1,
 "selection_basis":"matched exact-707 grouped held-out fold 0; public/hidden blind",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "baseline_history":str(history.resolve()),"baseline_history_sha256":sha(history),
 "baseline_classification_checkpoint":str(checkpoint.resolve()),
 "baseline_classification_checkpoint_sha256":sha(checkpoint),
 "baseline_classification_prediction":str(prediction.resolve()),
 "baseline_classification_prediction_sha256":sha(prediction),
 "superseded_plan":str(superseded.resolve()),
 "superseded_plan_sha256":sha(superseded),
 "training":{"fold":0,"folds":5,"epochs":30,"batch_size":8,"workers":2,
  "lr":3e-4,"weight_decay":1e-4,"base_channels":32,
  "sampling_temperature":0.5,"samples_per_epoch":8192,"seed":20260823,
  "segmentation_threshold_max":0.95,"validation_interval":3,
  "validation_batch_size":32,"prompt_masked_classifier":True,
  "prompt_channel":2},
 "architecture":{"classifier_pooling":["global_average","global_max",
  "prompt_average","prompt_max"],
  "input_channel_layout":["soft","lung","seed_support"],
  "prompt_channel":2,"prompt_channel_semantic":"seed_support",
  "prompt_downsampling":"adaptive_average_occupancy",
  "reason":"guarantee every nonempty tiny prompt reaches the encoder grid"},
 "gates":{"minimum_classification_gain":0.005,
  "minimum_weak_source_gain":0.010,"minimum_lung_cell_gain":0.005,
  "maximum_mature_source_drop":0.005,"maximum_segmentation_drop":0.005,
  "minimum_composite_gain":0.0},
 "public_read":False,"hidden_read":False,"submission_created":False,
}
if output.is_file():
 observed=json.loads(output.read_text(encoding="utf-8"))
 comparable={key:observed.get(key) for key in expected if key!="created_at"}
 if comparable!=expected:
  raise SystemExit("frozen prompt-pool plan does not match current inputs/recipe")
else:
 payload={"created_at":datetime.datetime.now().astimezone().isoformat(),**expected}
 temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
 temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
 os.replace(temporary,output)
PY

while [[ ! -s "${primary_complete}" || ! -s "${primary_pipeline_complete}" \
    || ! -s "${tta_terminal}" || ! -s "${tta_fallback_terminal}" \
    || ! -s "${focus_terminal}" ]]; do
  printf '%s waiting for exact-707 primary and higher-priority Focus/TTA terminals\n' "$(date -Ins)"
  sleep 30
done
for required in "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${plan}" \
  "${baseline}/COMPLETE.txt" "${baseline}/history.json" \
  "${baseline}/checkpoint_best_classification.pt" \
  "${baseline}/validation_predictions_best_classification.npz"; do
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
  --validation-batch-size 32 --prompt-masked-classifier \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_v4_cell707_b32_t05_promptpool_fold0 \
  --resume > "${output}/train.log" 2>&1

audit_best_classification() {
  local run="$1" destination="$2" label="$3" epoch
  epoch=$("${python_bin}" - "${run}/checkpoint_best_classification.pt" <<'PY'
import sys,torch
payload=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
epoch=int(payload.get("epoch",-1))
if epoch<1: raise SystemExit("checkpoint lacks a valid epoch")
print(epoch)
PY
)
  "${python_bin}" -u "${root}/audit_candidate_refiner_fold_sources.py" \
    --index "${patches}/index.csv" \
    --prediction "${run}/validation_predictions_best_classification.npz" \
    --checkpoint "${run}/checkpoint_best_classification.pt" \
    --folds 5 --fold 0 --seed 20260823 --epoch "${epoch}" \
    --output "${destination}" > "${decision_dir}/${label}_audit.log" 2>&1
}
audit_best_classification "${baseline}" "${baseline_audit}" baseline
audit_best_classification "${output}" "${candidate_audit}" candidate

"${python_bin}" -u "${root}/select_candidate_refiner_prompt_pool.py" \
  --baseline-history "${baseline}/history.json" \
  --candidate-history "${output}/history.json" \
  --baseline-source-audit "${baseline_audit}" \
  --candidate-source-audit "${candidate_audit}" \
  --output "${decision}" > "${decision_dir}/select.log" 2>&1

"${python_bin}" - "${patches}/index.csv" "${baseline}/history.json" \
  "${output}/history.json" "${baseline_audit}" "${candidate_audit}" \
  "${plan}" "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,baseline,candidate,baseline_audit,candidate_audit,plan,decision,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","variant":"prompt_masked_classifier",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "baseline_history":str(baseline.resolve()),"baseline_history_sha256":sha(baseline),
 "candidate_history":str(candidate.resolve()),"candidate_history_sha256":sha(candidate),
 "baseline_source_audit":str(baseline_audit.resolve()),"baseline_source_audit_sha256":sha(baseline_audit),
 "candidate_source_audit":str(candidate_audit.resolve()),"candidate_source_audit_sha256":sha(candidate_audit),
 "plan":str(plan.resolve()),"plan_sha256":sha(plan),
 "decision":json.loads(decision.read_text()),"decision_path":str(decision.resolve()),
 "prompt_masked_classifier":True,"prompt_channel":2,
 "prompt_channel_semantic":"seed_support",
 "public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

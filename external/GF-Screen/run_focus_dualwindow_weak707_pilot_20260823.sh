#!/usr/bin/env bash
set -euo pipefail

# Local-only contour pilot on the balanced 707-case, source-separated cache.
# It starts after the matched fold-0 B32 contour baseline and patch audit are
# complete; other B32 folds may continue on the other GPU. It cannot read
# public or hidden cases, package a ZIP, or upload anything.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
training_skipped="${training}/TRAINING_SKIPPED.json"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
patch_audit="${training}/patches_bbox24_ss_min5_v4_cell48/AUDIT.json"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823/checkpoint_best_segmentation.pt"
baseline_prediction="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823/validation_predictions_best_segmentation.npz"
baseline_complete="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823/COMPLETE.txt"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
tta_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_SAVE_TERMINAL.json"
tta_fallback_terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_FALLBACK_SAVE_TERMINAL.json"
allow_early="${FOCUS_DUALWINDOW_ALLOW_EARLY_GPU:-0}"
gpu="${FOCUS_DUALWINDOW_GPU:-1}"
run="${root}/runs/focus_dualwindow_cell707_release10_fold0_20260823"
analysis="${root}/analysis/focus_dualwindow_cell707_pilot_20260823"
preflight="${analysis}/PREFLIGHT.json"
plan="${analysis}/PLAN.json"
decision="${analysis}/DECISION.json"
mixed_decision="${analysis}/MIXED_DECISION.json"
skipped="${analysis}/SKIPPED.json"

mkdir -p "${run}" "${analysis}"
exec 9>"${root}/runs/run_focus_dualwindow_weak707_pilot_20260823.lock"
flock -n 9 || { echo "dual-window weak707 Focus pilot already active" >&2; exit 1; }
[[ "${allow_early}" == 0 || "${allow_early}" == 1 ]] \
  || { echo "FOCUS_DUALWINDOW_ALLOW_EARLY_GPU must be 0 or 1" >&2; exit 2; }
[[ "${gpu}" =~ ^[0-9]+$ ]] \
  || { echo "FOCUS_DUALWINDOW_GPU must be a nonnegative integer" >&2; exit 2; }
[[ -s "${decision}" && -s "${mixed_decision}" ]] && exit 0
[[ -s "${skipped}" ]] && exit 0
while [[ ! -s "${baseline_complete}" && ! -s "${training_skipped}" ]]; do
  printf '%s waiting for weak707 patch audit and B32 fold-0 baseline\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${training_skipped}" ]]; then
  "${python_bin}" - "${training_skipped}" "${skipped}" <<'PY'
import datetime,json,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "decision":"SKIP_FOCUS_PILOT",
 "reason":"weak-domain proposal scale-out did not pass its frozen gate",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi
if [[ "${allow_early}" == 1 ]]; then
  while [[ ! -s "${tta_terminal}" || ! -s "${tta_fallback_terminal}" ]]; do
    printf '%s early dispatch waiting for completed champion-TTA decision\n' \
      "$(date -Ins)"
    sleep 30
  done
else
  while [[ ! -s "${primary_pipeline_complete}" || ! -s "${tta_terminal}" \
      || ! -s "${tta_fallback_terminal}" ]]; do
    printf '%s waiting for primary deployment and completed champion-TTA decision\n' \
      "$(date -Ins)"
    sleep 30
  done
fi
for required in "${index}" "${patch_audit}" "${pretrained}" "${baseline}" \
  "${baseline_prediction}" "${preflight}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

# Freeze exact inputs, recipe, and promotion gate before the first optimizer
# step. The baseline is the same fold's B32 segmentation-selected checkpoint.
if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${index}" "${pretrained}" "${baseline}" \
    "${baseline_prediction}" "${preflight}" "${allow_early}" "${gpu}" \
    "${plan}" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
index,pretrained,baseline,baseline_prediction,preflight=map(Path,sys.argv[1:6])
allow_early,gpu=sys.argv[6:8]
output=Path(sys.argv[8])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
preflight_payload=json.loads(preflight.read_text(encoding="utf-8"))
if preflight_payload.get("status") != "PASS":
 raise SystemExit("Focus checkpoint preflight did not pass")
if preflight_payload.get("checkpoint_sha256") != sha(pretrained):
 raise SystemExit("Focus preflight/checkpoint identity mismatch")
payload={
 "schema_version":1,
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"grouped held-out fold 0 of balanced complete-label 707 only",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "pretrained":str(pretrained.resolve()),"pretrained_sha256":sha(pretrained),
 "baseline_checkpoint":str(baseline.resolve()),
 "baseline_checkpoint_sha256":sha(baseline),
 "baseline_prediction":str(baseline_prediction.resolve()),
 "baseline_prediction_sha256":sha(baseline_prediction),
 "preflight":str(preflight.resolve()),"preflight_sha256":sha(preflight),
 "training":{
  "fold":0,"folds":5,"epochs":10,"batch_size":8,"workers":2,
  "lr":3e-5,"weight_decay":1e-5,"loss_recipe":"release_dice_ce",
  "sampling_temperature":0.5,"samples_per_epoch":2048,
  "segmentation_threshold_max":0.95,
  "freeze_encoder":True,"input_mode":"soft_lung_seed","seed":20260823,
 },
 "fold0_gate":{
  "minimum_case_dice_gain_over_b32":0.01,
  "minimum_cohort_dice_gain_over_b32":0.01,
  "minimum_case_dice_gain_over_released":0.03,
  "maximum_negative_nonempty_rate":0.95,
 },
 "mixed_source_gate":{
  "target_groups":["weak_sources","lung_cells"],
  "minimum_case_dice_gain_over_b32":0.02,
  "minimum_cohort_dice_gain_over_b32":0.01,
  "maximum_focus_negative_nonempty_rate":0.95,
  "threshold_selection":"separate group-optimal held-out case-balanced Dice",
  "mature_route":"b32",
 },
 "rationale":"A lung-window channel targets LNDb while the seed channel localizes broad weak proposals; extra input kernels are zero initialized so released CT behavior is preserved at epoch 0.",
 "scheduling":{
  "allow_early_gpu":allow_early == "1","physical_gpu":int(gpu),
  "early_dispatch_dependencies":["fold0_baseline","checkpoint_preflight","strict_tta_terminal","tta_fallback_terminal"],
  "omitted_ordering_only_dependency":"primary exact-707 public/save terminal" if allow_early == "1" else None,
 },
 "public_read":False,"hidden_read":False,"submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
fi

while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
export MPLCONFIGDIR="${run}/matplotlib"
export TMPDIR="/tmp/gf_focus_dualwindow_weak707_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  "${root}/train_focus_candidate_refiner.py" \
  --index "${index}" --pretrained "${pretrained}" --output "${run}" \
  --folds 5 --fold 0 --epochs 10 --batch-size 8 --workers 2 \
  --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
  --sampling-temperature 0.5 --samples-per-epoch 2048 \
  --segmentation-threshold-max 0.95 \
  --freeze-encoder --input-mode soft_lung_seed --seed 20260823 \
  --wandb-project pancancer-ct \
  --wandb-name focus_dualwindow_cell707_release10_fold0_20260823 --resume \
  > "${run}/train.log" 2>&1

"${python_bin}" -u "${root}/select_focus_dualwindow_weak707_pilot.py" \
  --plan "${plan}" --focus-checkpoint "${run}/checkpoint_best.pt" \
  --baseline-checkpoint "${baseline}" --output "${decision}" \
  > "${analysis}/select.log" 2>&1
"${python_bin}" -u "${root}/select_focus_b32_source_hybrid_pilot.py" \
  --plan "${plan}" --focus-checkpoint "${run}/checkpoint_best.pt" \
  --focus-prediction "${run}/validation_predictions_best.npz" \
  --baseline-checkpoint "${baseline}" \
  --baseline-prediction "${baseline_prediction}" \
  --output "${mixed_decision}" > "${analysis}/select_mixed.log" 2>&1
cat "${decision}"
cat "${mixed_decision}"

#!/usr/bin/env bash
set -euo pipefail

# Controlled model-level upgrade to the already-qualified frozen-encoder
# dual-window Focus route.  This is not the older fallback controller: it is
# allowed only after the frozen Focus and composite source route have promoted.
# Selection uses grouped internal fold 0 only.  There is no public/hidden,
# packaging, submission, or upload path in this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823/checkpoint_best_segmentation.pt"
frozen_analysis="${root}/analysis/focus_dualwindow_cell707_pilot_20260823"
frozen_decision="${frozen_analysis}/DECISION.json"
frozen_checkpoint="${root}/runs/focus_dualwindow_cell707_release10_fold0_20260823/checkpoint_best.pt"
composite_decision="${root}/analysis/focus_composite_routes_20260824/DECISION.json"
run="${root}/runs/focus_dualwindow_encoder01b8_upgrade_fold0_20260824"
analysis="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_pilot_20260824"
plan="${analysis}/PLAN.json"
decision="${analysis}/DECISION.json"
terminal="${analysis}/TERMINAL.json"

mkdir -p "${run}" "${analysis}"
exec 9>"${root}/runs/run_focus_dualwindow_encoder_upgrade_fold0_20260824.lock"
flock -n 9 || {
  echo "Focus encoder-upgrade fold-0 controller already active" >&2
  exit 1
}
[[ -s "${terminal}" ]] && exit 0

for required in "${index}" "${pretrained}" "${baseline}" \
  "${frozen_decision}" "${frozen_checkpoint}" "${composite_decision}"; do
  [[ -s "${required}" ]] || {
    echo "missing required asset: ${required}" >&2
    exit 2
  }
done

"${python_bin}" - "${frozen_decision}" "${composite_decision}" <<'PY'
import json,sys
frozen=json.load(open(sys.argv[1],encoding="utf-8"))
composite=json.load(open(sys.argv[2],encoding="utf-8"))
if frozen.get("decision") != "PROMOTE_TO_FIVEFOLD":
    raise SystemExit("frozen Focus was not qualified")
if composite.get("decision") != "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME":
    raise SystemExit("qualified composite route is unavailable")
PY

if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${index}" "${pretrained}" "${baseline}" \
    "${frozen_checkpoint}" "${frozen_decision}" "${composite_decision}" \
    "${plan}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,pretrained,baseline,frozen,frozen_decision,composite,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"predeclared grouped held-out fold 0 of internal exact707 only",
 "experiment_role":"encoder adaptation upgrade to qualified frozen Focus route",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "pretrained":str(pretrained.resolve()),"pretrained_sha256":sha(pretrained),
 "baseline_checkpoint":str(baseline.resolve()),
 "baseline_checkpoint_sha256":sha(baseline),
 "frozen_checkpoint":str(frozen.resolve()),
 "frozen_checkpoint_sha256":sha(frozen),
 "frozen_decision":str(frozen_decision.resolve()),
 "frozen_decision_sha256":sha(frozen_decision),
 "qualified_composite_decision":str(composite.resolve()),
 "qualified_composite_decision_sha256":sha(composite),
 "training":{
  "fold":0,"folds":5,"epochs":10,"batch_size":8,"workers":2,
  "lr":3e-5,"encoder_lr_scale":0.1,"weight_decay":1e-5,
  "loss_recipe":"release_dice_ce","sampling_temperature":0.5,
  "samples_per_epoch":2048,"freeze_encoder":False,
  "input_mode":"soft_lung_seed","segmentation_threshold_max":0.95,
  "seed":20260823
 },
 "fold0_gate":{
  "minimum_case_dice_gain_over_b32":0.01,
  "minimum_cohort_dice_gain_over_b32":0.01,
  "minimum_case_dice_gain_over_frozen":0.005,
  "minimum_cohort_dice_gain_over_frozen":0.005,
  "maximum_negative_nonempty_rate":0.95
 },
 "rationale":"Test whether a 0.1 encoder LR multiplier improves the qualified dual-window Focus route while preserving the released CT representation and frozen decoder LR recipe.",
 "public_read":False,"hidden_read":False,"submission_created":False
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
fi

gpu="${FOCUS_ENCODER_UPGRADE_GPU:-1}"
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${run}/matplotlib"
export TMPDIR="/tmp/gf_focus_encoder01b8_upgrade_fold0_20260824"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  "${root}/train_focus_candidate_refiner.py" \
  --index "${index}" --pretrained "${pretrained}" --output "${run}" \
  --folds 5 --fold 0 --epochs 10 --batch-size 8 --workers 2 \
  --lr 3e-5 --encoder-lr-scale 0.1 --weight-decay 1e-5 \
  --loss-recipe release_dice_ce --sampling-temperature 0.5 \
  --samples-per-epoch 2048 --segmentation-threshold-max 0.95 \
  --input-mode soft_lung_seed --seed 20260823 \
  --wandb-project pancancer-ct \
  --wandb-name focus_dualwindow_encoder01b8_upgrade_fold0_20260824 --resume \
  > "${run}/train.log" 2>&1

"${python_bin}" -u "${root}/select_focus_discriminative_pilot.py" \
  --plan "${plan}" --candidate-checkpoint "${run}/checkpoint_best.pt" \
  --baseline-checkpoint "${baseline}" --frozen-checkpoint "${frozen_checkpoint}" \
  --expected-batch-size 8 \
  --output "${decision}" > "${analysis}/select.log" 2>&1

"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "pilot_decision":str(source.resolve()),
 "pilot_decision_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

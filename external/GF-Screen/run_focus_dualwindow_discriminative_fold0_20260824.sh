#!/usr/bin/env bash
set -euo pipefail

# Model-level fallback for the exact707 dual-window Focus pilot. It runs only
# when the frozen-encoder pilot fails its predeclared fold-0 gate. Decoder and
# output tensors retain the 3e-5 warmup/cosine recipe while pretrained encoder
# tensors receive a 0.1 LR multiplier. No public/hidden inference, packaging,
# or upload path is reachable from this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823/checkpoint_best_segmentation.pt"
frozen_analysis="${root}/analysis/focus_dualwindow_cell707_pilot_20260823"
frozen_decision="${frozen_analysis}/DECISION.json"
frozen_mixed_decision="${frozen_analysis}/MIXED_DECISION.json"
frozen_checkpoint="${root}/runs/focus_dualwindow_cell707_release10_fold0_20260823/checkpoint_best.pt"
run="${root}/runs/focus_dualwindow_disc01_release10_fold0_20260824"
analysis="${root}/analysis/focus_dualwindow_disc01_pilot_20260824"
plan="${analysis}/PLAN.json"
decision="${analysis}/DECISION.json"
terminal="${analysis}/FOCUS_DISC_TERMINAL.json"

mkdir -p "${run}" "${analysis}"
exec 9>"${root}/runs/run_focus_dualwindow_discriminative_fold0_20260824.lock"
flock -n 9 || { echo "discriminative Focus fold-0 controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${frozen_decision}" || ! -s "${frozen_mixed_decision}" ]]; do
  printf '%s waiting for frozen global and mixed Focus pilot decisions\n' "$(date -Ins)"
  sleep 30
done
frozen_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${frozen_decision}")
frozen_mixed_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${frozen_mixed_decision}")
if [[ "${frozen_status}" != KEEP_B32 || \
      "${frozen_mixed_status}" == PROMOTE_MIXED_TO_FIVEFOLD ]]; then
  "${python_bin}" - "${frozen_decision}" "${frozen_mixed_decision}" \
    "${frozen_status}" "${frozen_mixed_status}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); mixed=Path(sys.argv[2]); observed=sys.argv[3]
mixed_observed=sys.argv[4]; output=Path(sys.argv[5])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_FROZEN_FOCUS",
 "reason":f"frozen global/mixed decisions were {observed}/{mixed_observed}",
 "source":str(source.resolve()),"mixed_source":str(mixed.resolve()),
 "public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
for required in "${index}" "${pretrained}" "${baseline}" \
  "${frozen_checkpoint}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${index}" "${pretrained}" "${baseline}" \
    "${frozen_checkpoint}" "${frozen_decision}" "${plan}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,pretrained,baseline,frozen,frozen_decision,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
frozen_payload=json.loads(frozen_decision.read_text(encoding="utf-8"))
if frozen_payload.get("decision")!="KEEP_B32":
    raise SystemExit("discriminative fallback requires a rejected frozen pilot")
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"grouped held-out fold 0 of balanced complete-label 707 only",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "pretrained":str(pretrained.resolve()),"pretrained_sha256":sha(pretrained),
 "baseline_checkpoint":str(baseline.resolve()),
 "baseline_checkpoint_sha256":sha(baseline),
 "frozen_checkpoint":str(frozen.resolve()),
 "frozen_checkpoint_sha256":sha(frozen),
 "frozen_decision":str(frozen_decision.resolve()),
 "frozen_decision_sha256":sha(frozen_decision),
 "training":{
  "fold":0,"folds":5,"epochs":10,"batch_size":4,"workers":2,
  "lr":3e-5,"encoder_lr_scale":0.1,"weight_decay":1e-5,
  "loss_recipe":"release_dice_ce","sampling_temperature":0.5,
  "samples_per_epoch":2048,"freeze_encoder":False,
  "input_mode":"soft_lung_seed","segmentation_threshold_max":0.95,
  "seed":20260823,
 },
 "fold0_gate":{
  "minimum_case_dice_gain_over_b32":0.01,
  "minimum_cohort_dice_gain_over_b32":0.01,
  "minimum_case_dice_gain_over_frozen":0.005,
  "minimum_cohort_dice_gain_over_frozen":0.005,
  "maximum_negative_nonempty_rate":0.95,
 },
 "rationale":"Prior full-encoder adaptation improved frozen Focus by 0.0075 Dice. A 0.1 encoder LR preserves released features while allowing the weak/lung distribution shift to propagate beyond the zero-initialized input adapters.",
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
fi

gpu="${FOCUS_DISC_GPU:-1}"
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
export MPLCONFIGDIR="${run}/matplotlib"
export TMPDIR="/tmp/gf_focus_dualwindow_disc01_20260824"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  "${root}/train_focus_candidate_refiner.py" \
  --index "${index}" --pretrained "${pretrained}" --output "${run}" \
  --folds 5 --fold 0 --epochs 10 --batch-size 4 --workers 2 \
  --lr 3e-5 --encoder-lr-scale 0.1 --weight-decay 1e-5 \
  --loss-recipe release_dice_ce --sampling-temperature 0.5 \
  --samples-per-epoch 2048 --segmentation-threshold-max 0.95 \
  --input-mode soft_lung_seed --seed 20260823 \
  --wandb-project pancancer-ct \
  --wandb-name focus_dualwindow_disc01_release10_fold0_20260824 --resume \
  > "${run}/train.log" 2>&1

"${python_bin}" -u "${root}/select_focus_discriminative_pilot.py" \
  --plan "${plan}" --candidate-checkpoint "${run}/checkpoint_best.pt" \
  --baseline-checkpoint "${baseline}" \
  --frozen-checkpoint "${frozen_checkpoint}" --output "${decision}" \
  > "${analysis}/select.log" 2>&1
"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "pilot_decision":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

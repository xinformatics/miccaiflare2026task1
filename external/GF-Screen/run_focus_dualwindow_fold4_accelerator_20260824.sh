#!/usr/bin/env bash
set -euo pipefail

# Opportunistically train the already-qualified Focus fold 4 on physical GPU 1.
# The primary five-fold controller trains folds 1..4 on GPU 0 and will skip this
# fold after COMPLETE.txt appears. This changes scheduling only; the model,
# data, seed, optimizer, and retained checkpoints are identical.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
pilot="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/DECISION.json"
mixed_pilot="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/MIXED_DECISION.json"
output="${root}/runs/focus_dualwindow_cell707_release10_fold4_20260823"
work="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
terminal="${work}/FOLD4_ACCELERATOR_TERMINAL.json"
gpu="${FOCUS_DUALWINDOW_ACCELERATOR_GPU:-1}"

mkdir -p "${output}" "${work}"
exec 9>"${root}/runs/run_focus_dualwindow_fold4_accelerator_20260824.lock"
flock -n 9 || { echo "Focus fold-4 accelerator already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }

for required in "${index}" "${pretrained}" "${pilot}" "${mixed_pilot}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
"${python_bin}" - "${pilot}" "${mixed_pilot}" <<'PY'
import json,sys
global_decision=json.load(open(sys.argv[1],encoding="utf-8"))
mixed_decision=json.load(open(sys.argv[2],encoding="utf-8"))
if global_decision.get("decision")!="PROMOTE_TO_FIVEFOLD":
    raise SystemExit("global Focus pilot did not qualify")
if mixed_decision.get("decision")!="PROMOTE_MIXED_TO_FIVEFOLD":
    raise SystemExit("mixed Focus pilot did not qualify")
for payload in (global_decision,mixed_decision):
    if payload.get("public_read") or payload.get("hidden_read") or payload.get("submission_created"):
        raise SystemExit("pilot decision is not local-only")
PY

while [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best.pt" ]]; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

# The primary controller may have finished fold 4 while this service waited.
if [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best.pt" ]]; then
  export MPLCONFIGDIR="${output}/matplotlib"
  export TMPDIR="/tmp/gf_focus_dualwindow_weak707_fold4_accelerator_20260824"
  mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    "${root}/train_focus_candidate_refiner.py" \
    --index "${index}" --pretrained "${pretrained}" --output "${output}" \
    --folds 5 --fold 4 --epochs 10 --batch-size 8 --workers 2 \
    --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
    --sampling-temperature 0.5 --samples-per-epoch 2048 \
    --segmentation-threshold-max 0.95 \
    --freeze-encoder --input-mode soft_lung_seed --seed 20260823 \
    --wandb-project pancancer-ct \
    --wandb-name focus_dualwindow_cell707_release10_fold4_20260823 \
    --resume > "${output}/train.log" 2>&1
fi

for required in "${output}/COMPLETE.txt" "${output}/checkpoint_best.pt" \
  "${output}/validation_predictions_best.npz"; do
  [[ -s "${required}" ]] || { echo "fold-4 output incomplete: ${required}" >&2; exit 3; }
done
"${python_bin}" - "${index}" "${pretrained}" "${pilot}" "${mixed_pilot}" \
  "${output}/checkpoint_best.pt" "${output}/validation_predictions_best.npz" \
  "${gpu}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
index,pretrained,pilot,mixed,checkpoint,prediction=map(Path,sys.argv[1:7])
gpu=int(sys.argv[7]); output=Path(sys.argv[8])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","experiment":"Focus fold-4 scheduling accelerator",
 "fold":4,"folds":5,"physical_gpu":gpu,
 "recipe":{
  "epochs":10,"batch_size":8,"workers":2,"lr":3e-5,
  "weight_decay":1e-5,"loss_recipe":"release_dice_ce",
  "sampling_temperature":0.5,"samples_per_epoch":2048,
  "segmentation_threshold_max":0.95,"freeze_encoder":True,
  "input_mode":"soft_lung_seed","seed":20260823,
 },
 "index":str(index.resolve()),"index_sha256":sha(index),
 "pretrained":str(pretrained.resolve()),"pretrained_sha256":sha(pretrained),
 "pilot":str(pilot.resolve()),"pilot_sha256":sha(pilot),
 "mixed_pilot":str(mixed.resolve()),"mixed_pilot_sha256":sha(mixed),
 "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha(checkpoint),
 "prediction":str(prediction.resolve()),"prediction_sha256":sha(prediction),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

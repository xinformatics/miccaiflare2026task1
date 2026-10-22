#!/usr/bin/env bash
set -euo pipefail

# One-fold, decoder-only architecture gate for a pretrained GF Focus contour
# refiner. Training normally starts after the primary exact-100 saver, before
# the lower-headroom proposal-transfer and 707-case scale-outs. If the exact
# A/B artifacts were precomputed on an otherwise idle GPU, their label-isolated
# decision can be written immediately. It cannot infer public/hidden data or
# create/upload a ZIP.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
small="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold0_20260823/checkpoint_best_segmentation.pt"
small_prediction="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold0_20260823/validation_predictions_best_segmentation.npz"
primary_saved="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
boundary_output="${root}/runs/focus_candidate_refiner_boundary_fold0_20260823"
release_output="${root}/runs/focus_candidate_refiner_release_fold0_20260823"
decision="${root}/analysis/focus_candidate_refiner_pilot_20260823/PILOT_DECISION.json"

mkdir -p "${boundary_output}" "${boundary_output}/matplotlib" \
  "${release_output}" "${release_output}/matplotlib" "$(dirname "${decision}")"
exec 9>"${root}/runs/run_focus_candidate_refiner_pilot_before_707_20260823.lock"
flock -n 9 || { echo "pre-707 Focus candidate pilot already active" >&2; exit 1; }
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${decision}" ]]; then
    "${python_bin}" - "${decision}" "${status}" "${line}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); output.parent.mkdir(parents=True,exist_ok=True)
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_SMALL_REFINER","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_read":False,"submission_created":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  fi
  exit "${status}"
}
trap write_failure EXIT
[[ -s "${decision}" ]] && exit 0
for required in "${index}" "${pretrained}" "${small}" "${small_prediction}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

train_variant() {
  local gpu="$1" recipe="$2" output="$3"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
    flock "${output}/train.lock" \
    "${python_bin}" -u "${root}/train_focus_candidate_refiner.py" \
    --index "${index}" --pretrained "${pretrained}" --output "${output}" \
    --folds 5 --fold 0 --epochs 3 --batch-size "${batch_size}" --workers 2 \
    --lr 3e-5 --weight-decay 1e-5 --loss-recipe "${recipe}" \
    --sampling-temperature 0.5 --samples-per-epoch 2048 \
    --freeze-encoder --seed 20260822 --wandb-project pancancer-ct \
    --wandb-name "focus_candidate_refiner_${recipe}_fold0_20260823" \
    --resume > "${output}/train.log" 2>&1
}
if [[ ! -s "${boundary_output}/COMPLETE.txt" \
    || ! -s "${boundary_output}/checkpoint_best.pt" \
    || ! -s "${release_output}/COMPLETE.txt" \
    || ! -s "${release_output}/checkpoint_best.pt" ]]; then
  while [[ ! -s "${primary_saved}" ]]; do
    printf '%s waiting for terminal primary exact-100 saver\n' "$(date -Ins)"
    sleep 30
  done
  while :; do
    used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
    printf '%s waiting for both GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
  batch_size=$(CUDA_VISIBLE_DEVICES=0 bash "${root}/ensure_focus_batch_size_20260823.sh")
  train_variant 0 boundary_balanced "${boundary_output}" & pid0=$!
  train_variant 1 release_dice_ce "${release_output}" & pid1=$!
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
fi

"${python_bin}" - "${boundary_output}/checkpoint_best.pt" \
  "${release_output}/checkpoint_best.pt" "${small}" \
  "${small_prediction}" "${pretrained}" "${index}" "${root}" "${decision}" <<'PY'
import csv,hashlib,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

boundary_path,release_path,small_path,small_prediction,pretrained,index,root,output=map(Path,sys.argv[1:])
sys.path.insert(0,str(root))
from focus_validation_metrics import aggregate_positive_dice
from train_candidate_refiner import grouped_fold_rows
variants=[]
for recipe,path in (("boundary_balanced",boundary_path),("release_dice_ce",release_path)):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("loss_recipe")!=recipe:
        raise SystemExit(f"loss recipe mismatch for {path}")
    variants.append((recipe,path,checkpoint))
small=torch.load(small_path,map_location="cpu",weights_only=False)
if any(int(checkpoint.get("fold",-1))!=0 for _,_,checkpoint in variants) or int(small.get("fold",-1))!=0:
    raise SystemExit("pilot checkpoints do not share grouped fold zero")
index_sha=hashlib.sha256(index.read_bytes()).hexdigest()
if any(checkpoint.get("index_sha256")!=index_sha for _,_,checkpoint in variants) or small.get("index_sha256")!=index_sha:
    raise SystemExit("pilot checkpoint index mismatch")
with index.open(newline="",encoding="utf-8") as stream:
    rows=list(csv.DictReader(stream))
_,validation_rows=grouped_fold_rows(rows,5,0,20260822)
with np.load(small_prediction,allow_pickle=False) as archive:
    small_summary=aggregate_positive_dice(
        archive["positive_patch_id"],archive["segmentation_dice_grid"],
        archive["segmentation_thresholds"],validation_rows,
    )
recipe,focus_path,focus=max(
    variants,
    key=lambda item:(
        float(item[2]["metrics"]["selection_score"]),
        float(item[2]["metrics"]["segmentation_dice"]),
        -float(item[2]["metrics"]["negative_nonempty_rate"]),
        item[0]=="release_dice_ce",
    ),
)
if any(int(candidate["config"]["batch_size"])!=int(focus["config"]["batch_size"]) for _,_,candidate in variants):
    raise SystemExit("Focus loss variants used different batch sizes")
focus_dice=float(focus["metrics"]["segmentation_dice"])
small_dice=float(small_summary["case_balanced_dice"])
gain=focus_dice-small_dice
negative_rate=float(focus["metrics"]["negative_nonempty_rate"])
qualifies=gain>=0.03 and negative_rate<=0.95
payload={
 "schema_version":1,
 "created_at":datetime.now().astimezone().isoformat(),
 "selection_basis":"same grouped fold-0 complete-label validation cases; patches averaged within case",
 "architecture":"72.76M-parameter pretrained GF Focus SwinUNETR; decoder/head fine-tune; controlled loss A/B",
 "pretrained":str(pretrained.resolve()),
 "pretrained_sha256":hashlib.sha256(pretrained.read_bytes()).hexdigest(),
 "focus_checkpoint":str(focus_path.resolve()),
 "focus_prediction":str((focus_path.parent/"validation_predictions_best.npz").resolve()),
 "selected_loss_recipe":recipe,
 "focus_batch_size":int(focus["config"]["batch_size"]),
 "small_checkpoint":str(small_path.resolve()),
 "small_prediction":str(small_prediction.resolve()),
 "small_patch_mean_segmentation_dice":float(small_summary["patch_mean_dice"]),
 "small_positive_cases":int(small_summary["positive_cases"]),
 "focus_segmentation_dice":focus_dice,
 "small_segmentation_dice":small_dice,
 "dice_gain":gain,
 "focus_negative_nonempty_rate":negative_rate,
 "loss_variants":[
   {"loss_recipe":candidate_recipe,
    "checkpoint":str(candidate_path.resolve()),
    "checkpoint_epoch":int(candidate["epoch"]),
    "segmentation_dice":float(candidate["metrics"]["segmentation_dice"]),
    "patch_mean_segmentation_dice":float(candidate["metrics"]["patch_mean_segmentation_dice"]),
    "cohort_balanced_segmentation_dice":float(candidate["metrics"]["cohort_balanced_segmentation_dice"]),
    "negative_nonempty_rate":float(candidate["metrics"]["negative_nonempty_rate"]),
    "selection_score":float(candidate["metrics"]["selection_score"])}
   for candidate_recipe,candidate_path,candidate in variants
 ],
 "rule":{"minimum_dice_gain":0.03,"maximum_negative_nonempty_rate":0.95},
 "decision":"PROMISING_FOR_FIVE_FOLD" if qualifies else "KEEP_SMALL_REFINER",
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload,indent=2))
PY

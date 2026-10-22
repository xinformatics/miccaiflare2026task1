#!/usr/bin/env bash
set -euo pipefail

# Expand the discriminative-LR dual-window Focus fallback only after its
# grouped fold-0 gate promotes. All selection remains on exact707 internal
# held-out patches; no public/hidden inference, packaging, or upload occurs.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${primary}/patches_bbox24_ss_min5_v4_cell48/index.csv"
primary_complete="${primary}/TRAINING_COMPLETE.json"
primary_summary="${primary}/five_fold_oof_summary.json"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
pilot_root="${root}/analysis/focus_dualwindow_disc01_pilot_20260824"
pilot_terminal="${pilot_root}/FOCUS_DISC_TERMINAL.json"
work="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
summary="${work}/focus_patch_oof.json"
decision="${work}/DECISION.json"
terminal="${work}/FIVEFOLD_TERMINAL.json"

fold_dir() {
  printf '%s/runs/focus_dualwindow_disc01_release10_fold%s_20260824' \
    "${root}" "$1"
}

mkdir -p "${work}"
exec 9>"${root}/runs/run_focus_dualwindow_discriminative_fivefold_20260824.lock"
flock -n 9 || { echo "discriminative Focus five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${pilot_terminal}" ]]; do
  printf '%s waiting for discriminative Focus fold-0 terminal\n' "$(date -Ins)"
  sleep 30
done
pilot_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${pilot_terminal}")
if [[ "${pilot_decision}" != PROMOTE_TO_FIVEFOLD ]]; then
  "${python_bin}" - "${pilot_terminal}" "${pilot_decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); observed=sys.argv[2]; output=Path(sys.argv[3])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_B32",
 "reason":f"discriminative Focus fold-0 decision was {observed}",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
for required in "${index}" "${pretrained}" "$(fold_dir 0)/checkpoint_best.pt"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

gpu="${FOCUS_DISC_FIVEFOLD_GPU:-1}"
for fold in 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  mkdir -p "${output}"
  if [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best.pt" ]]; then
    while :; do
      used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d '[:space:]')
      [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
      printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
      sleep 30
    done
    export MPLCONFIGDIR="${output}/matplotlib"
    export TMPDIR="/tmp/gf_focus_dualwindow_disc01_fold${fold}_20260824"
    mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      "${root}/train_focus_candidate_refiner.py" \
      --index "${index}" --pretrained "${pretrained}" --output "${output}" \
      --folds 5 --fold "${fold}" --epochs 10 --batch-size 4 --workers 2 \
      --lr 3e-5 --encoder-lr-scale 0.1 --weight-decay 1e-5 \
      --loss-recipe release_dice_ce --sampling-temperature 0.5 \
      --samples-per-epoch 2048 --segmentation-threshold-max 0.95 \
      --input-mode soft_lung_seed --seed 20260823 \
      --wandb-project pancancer-ct \
      --wandb-name "focus_dualwindow_disc01_release10_fold${fold}_20260824" \
      --resume > "${output}/train.log" 2>&1
  fi
done

args=()
for fold in 0 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  args+=(--checkpoint "${output}/checkpoint_best.pt")
  args+=(--prediction "${output}/validation_predictions_best.npz")
done
"${python_bin}" -u "${root}/summarize_focus_candidate_refiner_oof.py" \
  --index "${index}" "${args[@]}" --folds 5 --seed 20260823 \
  --output "${summary}" > "${work}/summarize.log" 2>&1
while [[ ! -s "${primary_complete}" ]]; do
  printf '%s waiting for matched B32 five-fold OOF summary\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${primary_summary}" ]] || { echo "missing B32 summary" >&2; exit 2; }
"${python_bin}" -u "${root}/select_focus_dualwindow_weak707_fivefold.py" \
  --focus-summary "${summary}" --b32-summary "${primary_summary}" \
  --output "${decision}" > "${work}/select.log" 2>&1
"${python_bin}" - "${decision}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,summary,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
recipe=json.loads(summary.read_text()).get("training_recipe",{})
if recipe.get("freeze_encoder") is not False or recipe.get("encoder_lr_scale")!=0.1:
    raise SystemExit("discriminative Focus summary recipe mismatch")
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "fivefold_decision":str(source.resolve()),
 "fivefold_decision_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "summary":str(summary.resolve()),
 "summary_sha256":hashlib.sha256(summary.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

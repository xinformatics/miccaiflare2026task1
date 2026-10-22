#!/usr/bin/env bash
set -euo pipefail

# Expand the batch-8 encoder-adaptation upgrade only after its deployed-route
# fold-0 gate promotes. Final selection compares independently selected exact
# five-fold weak/lung patch OOF against frozen Focus; mature remains B32. No
# public/hidden inference, packaging, submission, or upload is reachable.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${primary}/patches_bbox24_ss_min5_v4_cell48/index.csv"
b32_summary="${primary}/five_fold_oof_summary.json"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
pilot_root="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_pilot_20260824"
pilot_terminal="${pilot_root}/ROUTE_TERMINAL.json"
frozen_summary="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_epoch_routes.json"
work="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824"
summary="${work}/focus_epoch_routes.json"
decision="${work}/DECISION.json"
terminal="${work}/FIVEFOLD_TERMINAL.json"

fold_dir() {
  printf '%s/runs/focus_dualwindow_encoder01b8_upgrade_fold%s_20260824' \
    "${root}" "$1"
}

mkdir -p "${work}"
exec 9>"${root}/runs/run_focus_dualwindow_encoder_upgrade_fivefold_20260824.lock"
flock -n 9 || {
  echo "Focus encoder-upgrade five-fold controller already active" >&2
  exit 1
}
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${pilot_terminal}" ]]; do
  printf '%s waiting for batch-8 encoder-upgrade fold-0 terminal\n' "$(date -Ins)"
  sleep 30
done

pilot_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${pilot_terminal}")
if [[ "${pilot_decision}" != PROMOTE_ROUTES_TO_FIVEFOLD ]]; then
  "${python_bin}" - "${pilot_terminal}" "${pilot_decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); observed=sys.argv[2]; output=Path(sys.argv[3])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_FROZEN_FOCUS",
 "reason":f"encoder-upgrade fold-0 decision was {observed}",
 "source":str(source.resolve()),
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi

for required in "${index}" "${pretrained}" "${b32_summary}" \
  "${frozen_summary}" "$(fold_dir 0)/COMPLETE.txt" \
  "$(fold_dir 0)/checkpoint_best.pt"; do
  [[ -s "${required}" ]] || {
    echo "missing required asset: ${required}" >&2
    exit 2
  }
done

gpu0="${FOCUS_ENCODER_UPGRADE_FIVEFOLD_GPU0:-0}"
gpu1="${FOCUS_ENCODER_UPGRADE_FIVEFOLD_GPU1:-1}"
train_fold() {
  local gpu="$1" fold="$2" output used
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
    mkdir -p "${output}/matplotlib" \
      "/tmp/gf_focus_encoder01b8_upgrade_fold${fold}_20260824"
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
      TMPDIR="/tmp/gf_focus_encoder01b8_upgrade_fold${fold}_20260824" \
      "${python_bin}" -u \
      "${root}/train_focus_candidate_refiner.py" \
      --index "${index}" --pretrained "${pretrained}" --output "${output}" \
      --folds 5 --fold "${fold}" --epochs 10 --batch-size 8 --workers 2 \
      --lr 3e-5 --encoder-lr-scale 0.1 --weight-decay 1e-5 \
      --loss-recipe release_dice_ce --sampling-temperature 0.5 \
      --samples-per-epoch 2048 --segmentation-threshold-max 0.95 \
      --input-mode soft_lung_seed --seed 20260823 \
      --wandb-project pancancer-ct \
      --wandb-name "focus_dualwindow_encoder01b8_upgrade_fold${fold}_20260824" \
      --resume > "${output}/train.log" 2>&1
  fi
}
train_sequence() {
  local gpu="$1"; shift
  local fold
  for fold in "$@"; do train_fold "${gpu}" "${fold}"; done
}
# Fold 0 has released GPU 1 before the route terminal can exist. Split the
# four untouched confirmations over both devices to preserve the exact recipe
# while halving continuation wall time. GPU memory guards serialize safely
# behind any earlier owner.
train_sequence "${gpu0}" 1 3 & pid0=$!
train_sequence "${gpu1}" 2 4 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

args=()
for fold in 0 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  args+=(--run "${output}")
done
"${python_bin}" -u "${root}/summarize_focus_epoch_routes.py" \
  --index "${index}" "${args[@]}" --folds 5 --seed 20260823 \
  --expected-epochs 11 \
  --output "${summary}" > "${work}/summarize.log" 2>&1
"${python_bin}" -u "${root}/select_focus_encoder_upgrade_fivefold.py" \
  --candidate-summary "${summary}" --frozen-summary "${frozen_summary}" \
  --b32-summary "${b32_summary}" --output "${decision}" \
  > "${work}/select.log" 2>&1

"${python_bin}" - "${decision}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,summary,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "fivefold_decision":str(source.resolve()),"fivefold_decision_sha256":sha(source),
 "summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "public_read":False,"hidden_read":False,"submission_created":False
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

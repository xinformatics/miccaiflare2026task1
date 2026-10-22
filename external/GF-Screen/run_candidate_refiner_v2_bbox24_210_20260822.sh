#!/usr/bin/env bash
set -euo pipefail

# High-ceiling bbox+24 mm tiled refiner. This training-only controller uses
# label-blind epoch-105 proposals and labels only from documented complete-label
# cohorts. It performs no public-label selection, hidden inference, packaging,
# or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
proposal_dir="${root}/analysis/candidate_refiner_v2_training_20260822/epoch105_abd_g090_l050"
proposal_marker="${proposal_dir}/PROPOSALS_COMPLETE.json"
work="${root}/analysis/candidate_refiner_v2_training_20260822"
patches="${work}/patches_bbox24_min5_v5"
audit="${patches}/AUDIT.json"
oof="${work}/five_fold_oof_summary.json"
expected=210

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${work}" "${root}/runs"
exec 9>"${root}/runs/run_candidate_refiner_v2_bbox24_210_20260822.lock"
flock -n 9 || { echo "candidate-refiner v2 controller already active" >&2; exit 1; }
[[ -s "${work}/TRAINING_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${proposal_marker}" ]]; do
  printf '%s waiting for epoch-105 proposal cache\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" - "${manifest}" "${proposal_dir}" "${expected}" <<'PY'
import json,sys
from pathlib import Path
manifest=Path(sys.argv[1]); proposal=Path(sys.argv[2]); expected=int(sys.argv[3])
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
names={str(item["prediction_name"]) for item in items}
observed={path.name for path in proposal.glob("*.nii.gz")}
if len(items)!=expected or len(names)!=expected or observed!=names:
    raise SystemExit(
        f"proposal coverage mismatch items={len(items)} names={len(names)} "
        f"observed={len(observed)} missing={sorted(names-observed)[:3]} "
        f"extra={sorted(observed-names)[:3]}"
    )
PY

export MPLCONFIGDIR="${work}/matplotlib"
export TMPDIR="/tmp/gfref_v2_20260822"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
if [[ ! -s "${patches}/COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/prepare_candidate_refiner_patches.py" \
    --manifest "${manifest}" --proposal "epoch105=${proposal_dir}" \
    --output "${patches}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 \
    --max-negatives-per-case 64 --max-negative-tiles 1 --workers 6 \
    > "${work}/prepare_patches_bbox24_min5.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_patches.py" \
  --index "${patches}/index.csv" --summary "${patches}/summary.json" \
  --output "${audit}" --workers 4 > "${work}/audit_patches_bbox24_min5.log" 2>&1

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

train_fold() {
  local gpu="$1" fold="$2" output
  output="$(fold_dir "${fold}")"
  mkdir -p "${output}"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
    --index "${patches}/index.csv" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 32 --seed 20260822 \
    --wandb-project pancancer-ct --wandb-name "candidate_refiner_v2_bbox24_b32_v2det_fold${fold}" \
    --resume > "${output}/train.log" 2>&1
}
train_sequence() {
  local gpu="$1"; shift
  local fold
  for fold in "$@"; do train_fold "${gpu}" "${fold}"; done
}
train_sequence 0 0 2 4 & pid0=$!
train_sequence 1 1 3 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

selection_args=()
for strategy_spec in \
  'composite checkpoint_best.pt validation_predictions_best.npz' \
  'segmentation checkpoint_best_segmentation.pt validation_predictions_best_segmentation.npz' \
  'classification checkpoint_best_classification.pt validation_predictions_best_classification.npz'; do
  read -r strategy checkpoint_name prediction_name <<< "${strategy_spec}"
  strategy_summary="${work}/five_fold_oof_summary_${strategy}.json"
  oof_args=()
  for fold in 0 1 2 3 4; do
    output="$(fold_dir "${fold}")"
    oof_args+=(--checkpoint "${output}/${checkpoint_name}")
    oof_args+=(--prediction "${output}/${prediction_name}")
  done
  "${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
    --index "${patches}/index.csv" "${oof_args[@]}" --output "${strategy_summary}" \
    > "${work}/five_fold_oof_summary_${strategy}.log" 2>&1
  selection_args+=(--candidate "${strategy}=${strategy_summary}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_oof_strategy.py" \
  "${selection_args[@]}" --output "${work}/checkpoint_selection.json" \
  --selected-summary "${oof}" > "${work}/checkpoint_selection.log" 2>&1
date -Ins > "${work}/TRAINING_COMPLETE.txt"
cat "${oof}"

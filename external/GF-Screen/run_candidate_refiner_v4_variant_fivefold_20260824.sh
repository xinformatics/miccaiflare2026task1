#!/usr/bin/env bash
set -euo pipefail

# Confirm one prequalified exact-707 fold-0 ablation on folds 1--4. Each
# variant owns one GPU, has no public/hidden path, and must beat the primary
# five-fold patch OOF before full-volume evaluation is allowed.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
variant="${1:-}"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
primary_complete="${train}/TRAINING_COMPLETE.json"
primary_summary="${train}/five_fold_oof_summary.json"

case "${variant}" in
  prompt_pool)
    fold0_terminal="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_20260824/FOLD0_TERMINAL.json"
    fold0_decision="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_20260824/FOLD0_DECISION.json"
    pass_decision="EXPAND_PROMPT_MASKED_CLASSIFIER"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_promptpool_fold"
    work="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_fivefold_20260824"
    gpu="${PROMPT_POOL_FIVEFOLD_GPU:-1}"
    train_extra=(--prompt-masked-classifier)
    ;;
  source_condition)
    fold0_terminal="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_20260823/FOLD0_TERMINAL.json"
    fold0_decision="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_20260823/FOLD0_DECISION.json"
    pass_decision="EXPAND_SOURCE_CONDITIONING"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_srccond_fold"
    work="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_fivefold_20260824"
    gpu="${SOURCE_CONDITION_FIVEFOLD_GPU:-1}"
    train_extra=(--source-conditioning)
    ;;
  small_lesion)
    fold0_terminal="${root}/analysis/candidate_refiner_v4_cell707_small_loss_20260823/FOLD0_TERMINAL.json"
    fold0_decision="${root}/analysis/candidate_refiner_v4_cell707_small_loss_20260823/FOLD0_DECISION.json"
    pass_decision="EXPAND_SMALL_LESION_LOSS"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_smallloss30_fold"
    work="${root}/analysis/candidate_refiner_v4_cell707_small_loss_fivefold_20260824"
    gpu="${SMALL_LESION_FIVEFOLD_GPU:-0}"
    train_extra=(--small-lesion-loss --small-lesion-positive-weight-cap 30)
    ;;
  uncertain_negative)
    fold0_terminal="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_20260824/FOLD0_TERMINAL.json"
    fold0_decision="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_20260824/FOLD0_DECISION.json"
    pass_decision="EXPAND_UNCERTAIN_NEGATIVE_025"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_uneg025_fold"
    work="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_fivefold_20260824"
    gpu="${UNCERTAIN_NEGATIVE_FIVEFOLD_GPU:-0}"
    train_extra=(--classification-negative-weight 0.25 --negative-segmentation-weight 0.025)
    ;;
  *) echo "usage: $0 prompt_pool|source_condition|small_lesion|uncertain_negative" >&2; exit 2;;
esac
[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }

selection="${work}/checkpoint_selection.json"
summary="${work}/five_fold_oof_summary.json"
decision="${work}/DECISION.json"
terminal="${work}/FIVEFOLD_TERMINAL.json"
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }

mkdir -p "${work}"
exec 9>"${root}/runs/run_candidate_refiner_v4_${variant}_fivefold_20260824.lock"
flock -n 9 || { echo "${variant} five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${fold0_terminal}" ]]; do
  printf '%s waiting for %s fold-0 decision\n' "$(date -Ins)" "${variant}"
  sleep 30
done
observed=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${fold0_decision}")
if [[ "${observed}" != "${pass_decision}" ]]; then
  "${python_bin}" - "${variant}" "${fold0_terminal}" "${observed}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
variant=sys.argv[1]; source=Path(sys.argv[2]); decision=sys.argv[3]; output=Path(sys.argv[4])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","variant":variant,"decision":"KEEP_PRIMARY",
 "reason":f"fold-0 decision was {decision}","fold0_terminal":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
  exit 0
fi
while [[ ! -s "${primary_complete}" ]]; do
  printf '%s waiting for primary exact-707 five-fold summary\n' "$(date -Ins)"
  sleep 30
done
for required in "${patches}/index.csv" "${patches}/AUDIT.json" "${primary_summary}" \
  "$(fold_dir 0)/COMPLETE.txt" "$(fold_dir 0)/checkpoint_best.pt"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 3; }
done
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
for fold in 1 2 3 4; do
  output="$(fold_dir "${fold}")"; mkdir -p "${output}/matplotlib"
  if [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
      "${python_bin}" -u "${root}/train_candidate_refiner.py" \
      --index "${patches}/index.csv" --output "${output}" \
      --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
      --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
      --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260823 \
      --segmentation-threshold-max 0.95 --validation-interval 3 \
      --validation-batch-size 32 "${train_extra[@]}" \
      --wandb-project pancancer-ct --wandb-name "${run_prefix}${fold}" --resume \
      > "${output}/train.log" 2>&1
  fi
done

selection_args=()
for specification in \
  'composite checkpoint_best.pt validation_predictions_best.npz' \
  'segmentation checkpoint_best_segmentation.pt validation_predictions_best_segmentation.npz' \
  'cohort_segmentation checkpoint_best_cohort_segmentation.pt validation_predictions_best_cohort_segmentation.npz' \
  'classification checkpoint_best_classification.pt validation_predictions_best_classification.npz'; do
  read -r strategy checkpoint_name prediction_name <<< "${specification}"
  strategy_summary="${work}/five_fold_oof_summary_${strategy}.json"; oof_args=()
  for fold in 0 1 2 3 4; do
    output="$(fold_dir "${fold}")"
    oof_args+=(--checkpoint "${output}/${checkpoint_name}" --prediction "${output}/${prediction_name}")
  done
  "${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
    --index "${patches}/index.csv" "${oof_args[@]}" --output "${strategy_summary}" \
    --source-group lung_cells=lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7 \
    > "${work}/summarize_${strategy}.log" 2>&1
  selection_args+=(--candidate "${strategy}=${strategy_summary}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_oof_strategy.py" \
  "${selection_args[@]}" --output "${selection}" --selected-summary "${summary}" \
  > "${work}/select_strategy.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_variant_oof.py" \
  --baseline "${primary_summary}" --candidate "${summary}" --variant "${variant}" \
  --output "${decision}" > "${work}/select_variant.log" 2>&1
"${python_bin}" - "${variant}" "${fold0_terminal}" "${selection}" "${summary}" \
  "${decision}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
variant=sys.argv[1]; fold0,selection,summary,decision,output=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","variant":variant,"fold0_terminal":str(fold0.resolve()),
 "selection":str(selection.resolve()),"selection_sha256":sha(selection),
 "summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "decision":str(decision.resolve()),"decision_payload":json.loads(decision.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY

#!/usr/bin/env bash
set -euo pipefail

# Expand the discovery-only contour specialist only after its exact fold-0
# gate passes. Then compose primary classification + mature contours with the
# specialist weak/cell contours and compare against the best primary route.
# No public/hidden inference, packaging, submission, or upload is possible.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
primary_complete="${train}/TRAINING_COMPLETE.json"
primary_route="${train}/source_routed_oof"
primary_route_terminal="${primary_route}/TERMINAL.json"
primary_decoupled="${train}/decoupled_oof"
pilot="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_20260824"
pilot_terminal="${pilot}/FOLD0_TERMINAL.json"
pilot_decision="${pilot}/FOLD0_DECISION.json"
work="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_fivefold_20260824"
summary="${work}/five_fold_oof_summary.json"
gate="${work}/SOURCE_ROUTE_GATE.json"
terminal="${work}/TERMINAL.json"
gpu="${DISCOVERY_SPECIALIST_FIVEFOLD_GPU:-0}"

primary_fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' "${root}" "$1"; }
specialist_fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_discovery_fold%s_20260824' "${root}" "$1"; }

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || { echo "invalid GPU: ${gpu}" >&2; exit 2; }
mkdir -p "${work}"
exec 9>"${root}/runs/run_candidate_refiner_v4_discovery_specialist_fivefold_20260824.lock"
flock -n 9 || { echo "discovery-specialist five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${pilot_terminal}" || ! -s "${primary_complete}" \
    || ! -s "${primary_route_terminal}" ]]; do
  printf '%s waiting for fold-0 specialist and exact primary/source-route terminals\n' "$(date -Ins)"
  sleep 30
done
pilot_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${pilot_decision}")
primary_route_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${primary_route_terminal}")
if [[ "${pilot_result}" != EXPAND_DISCOVERY_SPECIALIST ]]; then
  "${python_bin}" - "${pilot_terminal}" "${pilot_result}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); decision=sys.argv[2]; output=Path(sys.argv[3])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_PRIMARY_SOURCE_ROUTE",
 "reason":f"fold-0 decision was {decision}","fold0_terminal":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
  exit 0
fi
if [[ "${primary_route_result}" == QUALIFY_SOURCE_ROUTED_FULL_VOLUME ]]; then
  comparison_anchor="${primary_route}/five_fold_oof_summary.json"
  comparison_anchor_name="primary_source_route"
else
  comparison_anchor="${primary_decoupled}/five_fold_oof_summary.json"
  comparison_anchor_name="primary_global_decoupled"
fi
for required in "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${comparison_anchor}"; do
  [[ -s "${required}" ]] || { echo "missing prerequisite: ${required}" >&2; exit 3; }
done

while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
for fold in 1 2 3 4; do
  output="$(specialist_fold_dir "${fold}")"; mkdir -p "${output}/matplotlib"
  if [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best_routed_segmentation.pt" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
      "${python_bin}" -u "${root}/train_candidate_refiner.py" \
      --index "${patches}/index.csv" --output "${output}" \
      --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
      --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
      --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260823 \
      --segmentation-threshold-max 0.95 --validation-interval 3 \
      --validation-batch-size 32 \
      --train-source-group autopet_weak --train-source-group lndb_weak \
      --train-source-group lung_cells \
      --segmentation-selection-source-group autopet_weak \
      --segmentation-selection-source-group lndb_weak \
      --segmentation-selection-source-group lung_cells \
      --wandb-project pancancer-ct \
      --wandb-name "candidate_refiner_v4_cell707_discovery_fold${fold}" \
      --resume > "${output}/train.log" 2>&1
  fi
done

"${python_bin}" - "${patches}/index.csv" \
  "$(specialist_fold_dir 0)/checkpoint_best_routed_segmentation.pt" \
  "$(specialist_fold_dir 1)/checkpoint_best_routed_segmentation.pt" \
  "$(specialist_fold_dir 2)/checkpoint_best_routed_segmentation.pt" \
  "$(specialist_fold_dir 3)/checkpoint_best_routed_segmentation.pt" \
  "$(specialist_fold_dir 4)/checkpoint_best_routed_segmentation.pt" <<'PY'
import hashlib,sys,torch
from pathlib import Path
index=Path(sys.argv[1]); digest=hashlib.sha256(index.read_bytes()).hexdigest()
expected=["autopet_weak","lndb_weak","lung_cells"]
for fold,raw in enumerate(sys.argv[2:]):
    path=Path(raw); checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    config=checkpoint.get("config",{})
    if checkpoint.get("index_sha256")!=digest or int(checkpoint.get("fold",-1))!=fold:
        raise SystemExit(f"specialist checkpoint identity mismatch: {path}")
    if config.get("train_source_groups")!=expected or config.get("segmentation_selection_source_groups")!=expected:
        raise SystemExit(f"specialist source recipe mismatch: {path}")
PY

source_route_args=(
  --source-route epoch105=epoch105 --source-route currentgf=currentgf
  --source-route autopet_weak=discovery --source-route lndb_weak=discovery
)
for cell in 0 1 2 3 4 5 6 7; do source_route_args+=(--source-route "lung_cell${cell}=discovery"); done
summary_args=()
for fold in 0 1 2 3 4; do
  primary="$(primary_fold_dir "${fold}")"; specialist="$(specialist_fold_dir "${fold}")"
  fold_output="${work}/fold${fold}"; mkdir -p "${fold_output}"
  classification_checkpoint="${primary}/checkpoint_best_classification.pt"
  current_checkpoint="${primary}/checkpoint_best_currentgf_segmentation.pt"
  epoch_checkpoint="${primary}/checkpoint_best_epoch105_segmentation.pt"
  discovery_checkpoint="${specialist}/checkpoint_best_routed_segmentation.pt"
  classification_prediction="${primary}/validation_predictions_best_classification.npz"
  current_prediction="${primary}/validation_predictions_best_currentgf_segmentation.npz"
  epoch_prediction="${primary}/validation_predictions_best_epoch105_segmentation.npz"
  discovery_prediction="${specialist}/validation_predictions_best_routed_segmentation.npz"
  composed="${fold_output}/validation_predictions_source_routed.npz"
  audit="${fold_output}/SOURCE_COMPOSITION_AUDIT.json"
  for required in "${classification_checkpoint}" "${current_checkpoint}" \
    "${epoch_checkpoint}" "${discovery_checkpoint}" "${classification_prediction}" \
    "${current_prediction}" "${epoch_prediction}" "${discovery_prediction}"; do
    [[ -s "${required}" ]] || { echo "missing composed-route input: ${required}" >&2; exit 3; }
  done
  "${python_bin}" -u "${root}/compose_candidate_refiner_source_routed_validation_predictions.py" \
    --index "${patches}/index.csv" \
    --classification "${classification_prediction}" \
    --classification-checkpoint "${classification_checkpoint}" \
    --contour "currentgf=${current_prediction}" --contour "epoch105=${epoch_prediction}" \
    --contour "discovery=${discovery_prediction}" \
    --contour-checkpoint "currentgf=${current_checkpoint}" \
    --contour-checkpoint "epoch105=${epoch_checkpoint}" \
    --contour-checkpoint "discovery=${discovery_checkpoint}" \
    "${source_route_args[@]}" --output "${composed}" --audit "${audit}" \
    > "${fold_output}/compose.log" 2>&1
  summary_args+=(--checkpoint "${classification_checkpoint}" --prediction "${composed}" --source-composition-audit "${audit}")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${summary_args[@]}" --output "${summary}" \
  --source-group mature=epoch105,currentgf \
  --source-group weak_sources=autopet_weak,lndb_weak \
  --source-group lung_cells=lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7 \
  > "${work}/summarize.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_source_routed_oof.py" \
  --baseline "${comparison_anchor}" --candidate "${summary}" \
  --minimum-dice-gain 0.002 --maximum-group-drop 0.002 \
  --output "${gate}" > "${work}/select.log" 2>&1
"${python_bin}" - "${pilot_terminal}" "${summary}" "${gate}" \
  "${comparison_anchor_name}" "${comparison_anchor}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
pilot,summary,gate=map(Path,sys.argv[1:4]); anchor_name=sys.argv[4]
anchor=Path(sys.argv[5]); output=Path(sys.argv[6]); decision=json.loads(gate.read_text())["decision"]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,
 "experiment":"five-fold discovery specialist source-routed around primary mature contours",
 "pilot_terminal":str(pilot.resolve()),"summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "comparison_anchor":anchor_name,"comparison_anchor_path":str(anchor.resolve()),
 "comparison_anchor_sha256":sha(anchor),
 "gate":str(gate.resolve()),"gate_payload":json.loads(gate.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY

#!/usr/bin/env bash
set -euo pipefail

# Scheduling accelerator for the conditional five-fold x+y TTA audit. It
# evaluates already complete folds 0/1 and reuses the exact fold-4 pilot
# artifacts while GPU 0 finishes folds 2/3. Final selection remains the exact
# five-fold summarizer in run_focus_routed_tta_fivefold_20260824.sh.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
pilot_root="${root}/analysis/focus_dualwindow_cell707_fold4_tta_pilot_20260824"
pilot_terminal="${pilot_root}/TERMINAL.json"
pilot_decision="${pilot_root}/DECISION.json"
pilot_routes="${pilot_root}/FOLD4_ROUTE_SELECTION.json"
output="${root}/analysis/focus_dualwindow_cell707_routed_tta_fivefold_20260824"
route_work="${output}/early_route_selection"
terminal="${output}/EARLY_ACCELERATOR_TERMINAL.json"
gpu="${FOCUS_ROUTED_TTA_EARLY_GPU:-1}"

fold_run() {
  printf '%s/runs/focus_dualwindow_cell707_release10_fold%s_20260823' \
    "${root}" "$1"
}

mkdir -p "${route_work}" "${output}/predictions" "${output}/matplotlib"
exec 9>"${root}/runs/run_focus_routed_tta_completed_folds_accelerator_20260824.lock"
flock -n 9 || { echo "early routed TTA accelerator already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${terminal}" "${status}" "${decision}" "${reason}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":sys.argv[2],"decision":sys.argv[3],"reason":sys.argv[4],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_NOTTA "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${pilot_terminal}" ]]; do
  printf '%s early TTA accelerator waiting for fold-4 pilot\n' "$(date -Ins)"
  sleep 30
done
pilot_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${pilot_terminal}")
if [[ "${pilot_result}" != PROMOTE_TTA_TO_FIVEFOLD ]]; then
  write_terminal SKIPPED KEEP_NOTTA "fold-4 TTA pilot did not promote"
  trap - EXIT
  exit 0
fi
for required in "${pilot_decision}" "${pilot_routes}" "$(fold_run 0)/COMPLETE.txt" \
  "$(fold_run 1)/COMPLETE.txt" "$(fold_run 4)/COMPLETE.txt"; do
  [[ -s "${required}" ]] || { echo "missing early TTA asset: ${required}" >&2; exit 2; }
done

mapfile -t eligible_groups < <("${python_bin}" - "${pilot_decision}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
for group in p.get("eligible_groups",[]): print(group)
PY
)
[[ "${#eligible_groups[@]}" -ge 1 && "${#eligible_groups[@]}" -le 2 ]] || exit 2

for fold in 0 1; do
  run="$(fold_run "${fold}")"
  directory="${route_work}/fold${fold}"
  mkdir -p "${directory}/notta"
  for epoch in $(seq 0 10); do
    checkpoint="${run}/checkpoint_epoch_$(printf '%04d' "${epoch}").pt"
    prediction="${run}/validation_predictions_epoch_$(printf '%04d' "${epoch}").npz"
    audit="${directory}/notta/epoch_$(printf '%04d' "${epoch}").json"
    [[ -s "${checkpoint}" && -s "${prediction}" ]] || exit 2
    if [[ ! -s "${audit}" ]]; then
      "${python_bin}" "${root}/audit_focus_fold_epoch_sources.py" \
        --index "${index}" --checkpoint "${checkpoint}" \
        --prediction "${prediction}" --folds 5 --fold "${fold}" \
        --seed 20260823 --epoch "${epoch}" --output "${audit}" \
        > "${directory}/notta/epoch_$(printf '%04d' "${epoch}").log" 2>&1
    fi
  done
  audit_args=()
  for epoch in $(seq 0 10); do
    audit_args+=(--audit "${directory}/notta/epoch_$(printf '%04d' "${epoch}").json")
  done
  "${python_bin}" "${root}/select_focus_single_fold_routes.py" \
    "${audit_args[@]}" --expected-epochs 11 \
    --output "${directory}/ROUTE_SELECTION.json" \
    > "${directory}/select.log" 2>&1
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
wait_for_gpu() {
  local used
  while :; do
    used=$(gpu_memory_mib "${gpu}")
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && return 0
    printf '%s early routed TTA waiting for GPU %s: %s MiB\n' \
      "$(date -Ins)" "${gpu}" "${used}"
    sleep 30
  done
}

for fold in 0 1 4; do
  selection="${route_work}/fold${fold}/ROUTE_SELECTION.json"
  [[ "${fold}" -ne 4 ]] || selection="${pilot_routes}"
  mapfile -t specifications < <("${python_bin}" - "${selection}" "${pilot_decision}" <<'PY'
import json,sys
route=json.load(open(sys.argv[1],encoding="utf-8")); pilot=json.load(open(sys.argv[2],encoding="utf-8"))
seen=set()
for group in pilot["eligible_groups"]:
 selected=route["selected"][group]; epoch=int(selected["epoch"])
 if epoch not in seen:
  print(f"{epoch}\t{selected['checkpoint']}"); seen.add(epoch)
PY
  )
  for specification in "${specifications[@]}"; do
    IFS=$'\t' read -r epoch checkpoint <<< "${specification}"
    stem="fold${fold}_epoch$(printf '%04d' "${epoch}")"
    destination="${output}/predictions/${stem}"
    mkdir -p "${destination}"
    if [[ "${fold}" -eq 4 ]]; then
      source_prediction="${pilot_root}/tta_xy/epoch_$(printf '%04d' "${epoch}").npz"
      source_summary="${pilot_root}/tta_xy/epoch_$(printf '%04d' "${epoch}").summary.json"
      [[ -s "${source_prediction}" && -s "${source_summary}" ]] || exit 2
      [[ -e "${destination}/prediction.npz" || -L "${destination}/prediction.npz" ]] \
        || ln -s "${source_prediction}" "${destination}/prediction.npz"
      [[ -e "${destination}/summary.json" || -L "${destination}/summary.json" ]] \
        || ln -s "${source_summary}" "${destination}/summary.json"
    elif [[ ! -s "${destination}/summary.json" || ! -s "${destination}/prediction.npz" ]]; then
      wait_for_gpu
      CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
        TMPDIR="/tmp/gf_focus_routed_tta_early_20260824" \
        "${python_bin}" -u "${root}/evaluate_focus_checkpoint_tta.py" \
        --index "${index}" --checkpoint "${checkpoint}" \
        --prediction "${destination}/prediction.npz" \
        --summary "${destination}/summary.json" \
        --folds 5 --fold "${fold}" --seed 20260823 --tta-axes xy \
        --batch-size 8 --workers 2 --segmentation-threshold-max 0.95 \
        > "${destination}/evaluate.log" 2>&1
    fi
  done
done

write_terminal COMPLETE ACCELERATED_COMPLETED_FOLDS \
  "folds 0/1 evaluated and fold 4 pilot artifacts reused"
trap - EXIT
cat "${terminal}"

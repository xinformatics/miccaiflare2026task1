#!/usr/bin/env bash
set -euo pipefail

# Conditional exact five-fold patch-OOF x+y TTA evaluation. It can only run
# for source groups promoted by the held-out fold-4 pilot and never reads
# public/hidden volumes or creates a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
training="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
route_summary="${training}/focus_epoch_routes.json"
route_decision="${training}/focus_epoch_routes_decision.json"
pilot_root="${root}/analysis/focus_dualwindow_cell707_fold4_tta_pilot_20260824"
pilot_terminal="${pilot_root}/TERMINAL.json"
pilot_decision="${pilot_root}/DECISION.json"
output="${root}/analysis/focus_dualwindow_cell707_routed_tta_fivefold_20260824"
early_accelerator_terminal="${output}/EARLY_ACCELERATOR_TERMINAL.json"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}/predictions" "${output}/matplotlib"
exec 9>"${root}/runs/run_focus_routed_tta_fivefold_20260824.lock"
flock -n 9 || { echo "routed Focus TTA five-fold audit already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" observed_decision="$2" reason="$3"
  "${python_bin}" - "${terminal}" "${status}" "${observed_decision}" "${reason}" <<'PY'
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
  printf '%s waiting for fold-4 x+y TTA pilot terminal\n' "$(date -Ins)"
  sleep 30
done
pilot_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${pilot_terminal}")
pilot_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${pilot_terminal}")
if [[ "${pilot_status}" != COMPLETE || "${pilot_result}" != PROMOTE_TTA_TO_FIVEFOLD ]]; then
  write_terminal SKIPPED KEEP_NOTTA "fold-4 x+y TTA pilot did not promote"
  trap - EXIT
  exit 0
fi
[[ -s "${pilot_decision}" ]] || exit 2

# The accelerator fills folds 0/1 and reuses fold 4 while folds 2/3 are still
# training. Wait for its terminal before touching the shared prediction paths;
# a failed/skipped accelerator is non-fatal because the main sequences below
# deterministically regenerate anything missing.
while [[ ! -s "${early_accelerator_terminal}" ]]; do
  printf '%s waiting for completed-fold routed TTA accelerator terminal\n' \
    "$(date -Ins)"
  sleep 30
done
early_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${early_accelerator_terminal}")
case "${early_status}" in COMPLETE|SKIPPED|FAILED) ;; *) exit 2 ;; esac

while [[ ! -s "${route_summary}" || ! -s "${route_decision}" ]]; do
  printf '%s waiting for exact five-fold Focus epoch routes\n' "$(date -Ins)"
  sleep 30
done
route_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${route_decision}")
# The global epoch-routing gate averages mature, weak, and lung families.  A
# rejection there only blocks deployment of epoch routing as a global model;
# it must not suppress the separately qualified mixed route, which assigns B32
# to mature sources and uses these Focus epochs only for weak/lung TTA deltas.
case "${route_result}" in
  QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME|KEEP_GLOBAL_FOCUS_EPOCHS) ;;
  *)
    write_terminal SKIPPED KEEP_NOTTA \
      "Focus epoch-route evidence was not a terminal local decision"
    trap - EXIT
    exit 0
    ;;
esac

mapfile -t specifications < <("${python_bin}" - "${route_summary}" "${pilot_decision}" <<'PY'
import json,sys
route=json.load(open(sys.argv[1],encoding="utf-8")); pilot=json.load(open(sys.argv[2],encoding="utf-8"))
eligible=set(pilot.get("eligible_groups",[]))
if not eligible or not eligible <= {"weak_sources","lung_cells"}: raise SystemExit("invalid pilot groups")
seen=set()
for row in sorted(route["fold_records"],key=lambda value:int(value["fold"])):
 fold=int(row["fold"])
 for group in sorted(eligible):
  selected=row["selected"][group]; epoch=int(selected["epoch"]); checkpoint=selected["checkpoint"]
  key=(fold,epoch)
  if key not in seen:
   print(f"{fold}\t{epoch}\t{checkpoint}"); seen.add(key)
PY
)
[[ "${#specifications[@]}" -ge 5 && "${#specifications[@]}" -le 10 ]] || exit 2

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
wait_for_gpu() {
  local gpu="$1" used
  while :; do
    used=$(gpu_memory_mib "${gpu}")
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && return 0
    printf '%s waiting for routed TTA GPU %s: %s MiB\n' \
      "$(date -Ins)" "${gpu}" "${used}"
    sleep 30
  done
}
run_sequence() {
  local gpu="$1"; shift
  local specification fold epoch checkpoint stem directory
  for specification in "$@"; do
    IFS=$'\t' read -r fold epoch checkpoint <<< "${specification}"
    stem="fold${fold}_epoch$(printf '%04d' "${epoch}")"
    directory="${output}/predictions/${stem}"
    mkdir -p "${directory}"
    if [[ ! -s "${directory}/summary.json" || ! -s "${directory}/prediction.npz" ]]; then
      wait_for_gpu "${gpu}"
      CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
        TMPDIR="/tmp/gf_focus_routed_tta_fivefold_20260824_gpu${gpu}" \
        "${python_bin}" -u "${root}/evaluate_focus_checkpoint_tta.py" \
        --index "${index}" --checkpoint "${checkpoint}" \
        --prediction "${directory}/prediction.npz" \
        --summary "${directory}/summary.json" \
        --folds 5 --fold "${fold}" --seed 20260823 --tta-axes xy \
        --batch-size 8 --workers 2 --segmentation-threshold-max 0.95 \
        > "${directory}/evaluate.log" 2>&1
    fi
  done
}

specifications0=(); specifications1=()
for specification in "${specifications[@]}"; do
  IFS=$'\t' read -r fold _ <<< "${specification}"
  if (( fold % 2 == 0 )); then
    specifications0+=("${specification}")
  else
    specifications1+=("${specification}")
  fi
done
run_sequence 0 "${specifications0[@]}" & pid0=$!
run_sequence 1 "${specifications1[@]}" & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

summary_args=()
for specification in "${specifications[@]}"; do
  IFS=$'\t' read -r fold epoch _ <<< "${specification}"
  summary_args+=(--tta-summary "${output}/predictions/fold${fold}_epoch$(printf '%04d' "${epoch}")/summary.json")
done
"${python_bin}" -u "${root}/summarize_focus_routed_tta.py" \
  --route-summary "${route_summary}" --pilot "${pilot_decision}" \
  "${summary_args[@]}" --output "${decision}" \
  > "${output}/summarize.log" 2>&1
final_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
write_terminal COMPLETE "${final_decision}" "exact five-fold routed x+y TTA audit complete"
trap - EXIT
cat "${decision}"

#!/usr/bin/env bash
set -euo pipefail

# Requalify x/y TTA on only the encoder-upgraded weak/lung routes that pass
# exact five-fold no-TTA patch OOF. This is training-data-only and cannot
# package, submit, or read challenge validation/test cases.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
upgrade_root="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824"
upgrade_terminal="${upgrade_root}/RESTRICTED_TERMINAL.json"
route_summary="${upgrade_root}/focus_epoch_routes.json"
upgrade_decision="${upgrade_root}/RESTRICTED_DECISION.json"
output="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_routed_tta_fivefold_20260824"
eligibility="${output}/TTA_ELIGIBILITY.json"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}/predictions" "${output}/matplotlib"
exec 9>"${root}/runs/run_focus_encoder_upgrade_routed_tta_20260824.lock"
flock -n 9 || { echo "encoder-upgrade routed TTA audit already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" observed="$2" reason="$3"
  "${python_bin}" - "${terminal}" "${status}" "${observed}" "${reason}" <<'PY'
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_NOTTA \
    "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${upgrade_terminal}" ]]; do
  printf '%s waiting for encoder-upgrade five-fold terminal\n' "$(date -Ins)"
  sleep 30
done
upgrade_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${upgrade_terminal}")
upgrade_result=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' \
  "${upgrade_terminal}")
if [[ "${upgrade_status}" != COMPLETE || \
      "${upgrade_result}" != PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF ]]; then
  write_terminal SKIPPED KEEP_NOTTA \
    "encoder upgrade did not qualify on exact five-fold no-TTA routes"
  trap - EXIT
  exit 0
fi
for required in "${index}" "${route_summary}" "${upgrade_decision}"; do
  [[ -s "${required}" ]] || { echo "missing encoder TTA input: ${required}" >&2; exit 2; }
done
"${python_bin}" -u "${root}/prepare_focus_encoder_upgrade_tta.py" \
  --route-summary "${route_summary}" --upgrade-decision "${upgrade_decision}" \
  --output "${eligibility}" > "${output}/eligibility.log" 2>&1

mapfile -t specifications < <("${python_bin}" - "${route_summary}" "${eligibility}" <<'PY'
import json,sys
route=json.load(open(sys.argv[1],encoding="utf-8")); eligibility=json.load(open(sys.argv[2],encoding="utf-8"))
groups=set(eligibility["eligible_groups"]); seen=set()
for row in sorted(route["fold_records"],key=lambda value:int(value["fold"])):
 fold=int(row["fold"])
 for group in sorted(groups):
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
    printf '%s encoder TTA waiting for GPU %s: %s MiB\n' \
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
        TMPDIR="/tmp/gf_focus_encoder_tta_20260824_gpu${gpu}" \
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
  --route-summary "${route_summary}" --pilot "${eligibility}" \
  "${summary_args[@]}" --output "${decision}" \
  > "${output}/summarize.log" 2>&1
final_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
write_terminal COMPLETE "${final_decision}" \
  "exact five-fold encoder-upgrade x/y TTA audit complete"
trap - EXIT
cat "${decision}"

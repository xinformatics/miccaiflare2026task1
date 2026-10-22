#!/usr/bin/env bash
set -euo pipefail

# Local-only fold-4 comparison of the complete four-view x/y flip group
# against the already frozen three-view x+y TTA recipe. The current five-fold
# TTA controller must finish first so this pilot never competes for its GPUs.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
run="${root}/runs/focus_dualwindow_cell707_release10_fold4_20260823"
xy_root="${root}/analysis/focus_dualwindow_cell707_fold4_tta_pilot_20260824"
xy_route="${xy_root}/FOLD4_ROUTE_SELECTION.json"
xy_decision="${xy_root}/DECISION.json"
fivefold_terminal="${root}/analysis/focus_dualwindow_cell707_routed_tta_fivefold_20260824/TERMINAL.json"
output="${root}/analysis/focus_dualwindow_cell707_xy4_fold4_pilot_20260824"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}/predictions" "${output}/matplotlib"
exec 9>"${root}/runs/run_focus_xy4_fold4_pilot_20260824.lock"
flock -n 9 || { echo "Focus xy4 fold-4 pilot already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" observed_decision="$2" reason="$3" source="${4:-}"
  "${python_bin}" - "${terminal}" "${status}" "${observed_decision}" \
    "${reason}" "${source}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); source=Path(sys.argv[5]) if sys.argv[5] else None
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":sys.argv[2],"decision":sys.argv[3],"reason":sys.argv[4],
 "source":str(source.resolve()) if source else None,
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_XY3 \
    "failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${fivefold_terminal}" ]]; do
  printf '%s waiting for three-view routed TTA five-fold terminal\n' "$(date -Ins)"
  sleep 30
done
fivefold_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${fivefold_terminal}")
if [[ "${fivefold_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_XY3 "three-view five-fold TTA did not complete"
  trap - EXIT
  exit 0
fi
for required in "${index}" "${xy_route}" "${xy_decision}" "${run}/COMPLETE.txt"; do
  [[ -s "${required}" ]] || { echo "missing xy4 pilot asset: ${required}" >&2; exit 2; }
done

mapfile -t specifications < <("${python_bin}" - "${xy_route}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); seen=set()
for group in ("weak_sources","lung_cells"):
 selected=p["selected"][group]; epoch=int(selected["epoch"])
 if epoch not in seen:
  print(f"{epoch}\t{selected['checkpoint']}"); seen.add(epoch)
PY
)
[[ "${#specifications[@]}" -ge 1 && "${#specifications[@]}" -le 2 ]] || exit 2

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
wait_for_gpu() {
  local gpu="$1" used
  while :; do
    used=$(gpu_memory_mib "${gpu}")
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && return 0
    printf '%s waiting for xy4 pilot GPU %s: %s MiB\n' \
      "$(date -Ins)" "${gpu}" "${used}"
    sleep 30
  done
}
run_one() {
  local gpu="$1" specification="$2" epoch checkpoint directory
  IFS=$'\t' read -r epoch checkpoint <<< "${specification}"
  [[ -s "${checkpoint}" ]] || exit 2
  directory="${output}/predictions/epoch_$(printf '%04d' "${epoch}")"
  mkdir -p "${directory}"
  if [[ ! -s "${directory}/summary.json" || ! -s "${directory}/prediction.npz" ]]; then
    wait_for_gpu "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
      TMPDIR="/tmp/gf_focus_xy4_fold4_20260824_gpu${gpu}" \
      "${python_bin}" -u "${root}/evaluate_focus_checkpoint_xy4.py" \
      --index "${index}" --checkpoint "${checkpoint}" \
      --prediction "${directory}/prediction.npz" \
      --summary "${directory}/summary.json" --folds 5 --fold 4 \
      --seed 20260823 --batch-size 8 --workers 2 \
      --segmentation-threshold-max 0.95 > "${directory}/evaluate.log" 2>&1
  fi
}

pids=(); gpu=0
for specification in "${specifications[@]}"; do
  run_one "${gpu}" "${specification}" & pids+=("$!")
  gpu=$((1-gpu))
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
[[ "${status}" -eq 0 ]] || exit "${status}"

audit_args=()
for specification in "${specifications[@]}"; do
  IFS=$'\t' read -r epoch checkpoint <<< "${specification}"
  directory="${output}/predictions/epoch_$(printf '%04d' "${epoch}")"
  audit="${directory}/source_audit.json"
  "${python_bin}" "${root}/audit_focus_fold_epoch_sources.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --prediction "${directory}/prediction.npz" --folds 5 --fold 4 \
    --seed 20260823 --epoch "${epoch}" --output "${audit}" \
    > "${directory}/audit.log" 2>&1
  audit_args+=(--xy4-audit "${audit}")
done

"${python_bin}" "${root}/select_focus_xy4_pilot.py" \
  --xy-decision "${xy_decision}" --route-selection "${xy_route}" \
  "${audit_args[@]}" --output "${decision}" > "${output}/select.log" 2>&1
final_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${decision}")
write_terminal COMPLETE "${final_decision}" \
  "fold-4 complete xy flip-group comparison finished" "${decision}"
trap - EXIT
cat "${decision}"

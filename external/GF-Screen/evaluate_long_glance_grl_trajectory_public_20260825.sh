#!/usr/bin/env bash
set -euo pipefail

# Public-only evaluation for one materially long, teacher-safe strong-GRL
# Glance trajectory. Checkpoints and thresholds are fixed before inference.
# The child evaluator applies the exact current deployment family and cannot
# read hidden images, create an archive, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run_dir="${LONG_GLANCE_RUN_DIR:-${root}/runs/gf_glance_grl_long30_20260825}"
source="${LONG_GLANCE_SOURCE:-${root}/public_sweeps/gf_glance_grl_long30_20260825/SOURCE.json}"
output="${LONG_GLANCE_TRAJECTORY_OUTPUT:-${root}/public_sweeps/gf_glance_grl_long30_trajectory_20260825}"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
lock="${root}/runs/evaluate_long_glance_grl_trajectory_public_20260825.lock"
thresholds=(0.10 0.20 0.35)

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long Glance public evaluator already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED)
      preserved_terminal="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved_terminal}"
      printf '%s preserved failed long-Glance trajectory terminal for exact resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing long-Glance trajectory status: ${existing_status}" >&2; exit 2 ;;
  esac
fi
[[ -s "${source}" ]] || { echo "missing long Glance source contract" >&2; exit 2; }

read -r source_epoch source_status < <("${python_bin}" - "${source}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
print(payload.get("source_epoch",-1),payload.get("status",""))
PY
)
[[ "${source_status}" == PASS && "${source_epoch}" =~ ^[0-9]+$ ]] || {
  echo "invalid long Glance source contract" >&2; exit 2;
}
epochs=($((source_epoch + 10)) $((source_epoch + 20)) $((source_epoch + 30)))
for epoch in "${epochs[@]}"; do
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  [[ -s "${checkpoint}" ]] || { echo "missing Glance checkpoint: ${checkpoint}" >&2; exit 2; }
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for both long-Glance evaluation GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

evaluate_state() {
  local epoch="$1" threshold="$2" gpu="$3"
  local checkpoint threshold_tag child child_lock
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  threshold_tag="g${threshold/./p}"
  child=$(printf '%s/epoch_%04d/%s' "${output}" "${epoch}" "${threshold_tag}")
  child_lock=$(printf '%s/runs/evaluate_long_glance_e%04d_%s_20260825.lock' \
    "${root}" "${epoch}" "${threshold_tag}")
  LONG_FOCUS_CHECKPOINT="${checkpoint}" \
  LONG_FOCUS_PUBLIC_OUTPUT="${child}" \
  LONG_FOCUS_PUBLIC_GPU="${gpu}" \
  LONG_FOCUS_GLANCE_THRESHOLD="${threshold}" \
  LONG_FOCUS_PUBLIC_LOCK="${child_lock}" \
  bash "${root}/evaluate_long_focus_pu_champion_public_20260825.sh" \
    > "${child}.controller.log" 2>&1
  [[ -s "${child}/TERMINAL.json" && -s "${child}/summary.tsv" ]]
}

states=()
for epoch in "${epochs[@]}"; do
  for threshold in "${thresholds[@]}"; do
    states+=("${epoch}:${threshold}")
  done
done
for ((index=0; index<${#states[@]}; index+=2)); do
  left="${states[index]}"
  IFS=: read -r left_epoch left_threshold <<< "${left}"
  evaluate_state "${left_epoch}" "${left_threshold}" 0 & left_pid=$!
  right_pid=""
  if (( index + 1 < ${#states[@]} )); then
    right="${states[index+1]}"
    IFS=: read -r right_epoch right_threshold <<< "${right}"
    evaluate_state "${right_epoch}" "${right_threshold}" 1 & right_pid=$!
  fi
  status=0
  wait "${left_pid}" || status=$?
  if [[ -n "${right_pid}" ]]; then wait "${right_pid}" || status=$?; fi
  [[ "${status}" -eq 0 ]]
done

# Bind one descriptive lesion-size audit to the highest-DSC state. It is not
# an input to the public gate or state selection.
read -r diagnostic_epoch diagnostic_threshold diagnostic_variant diagnostic_predictions < <(
  "${python_bin}" - "${output}" "${source_epoch}" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1]);source_epoch=int(sys.argv[2]);rows=[]
for epoch in (source_epoch+10,source_epoch+20,source_epoch+30):
    for threshold in (0.10,0.20,0.35):
        tag=f"g{threshold:.2f}".replace(".","p")
        with open(root/f"epoch_{epoch:04d}"/tag/"summary.tsv",encoding="utf-8") as handle:
            for row in csv.DictReader(handle,delimiter="\t"):
                rows.append((float(row["DSC"]),float(row["NSD"]),epoch,threshold,row))
rows.sort(key=lambda item:(item[0],item[1]),reverse=True)
_,_,epoch,threshold,row=rows[0]
print(epoch,f"{threshold:.2f}",row["variant"],row["predictions"],sep="\t")
PY
)
lesion_error="${output}/best_dsc_lesion_error"
if [[ ! -s "${lesion_error}/summary.json" ]]; then
  "${python_bin}" -u "${root}/analyze_lesion_errors.py" \
    --labels "$(cd "${root}/../.." && pwd)/validation/Validation-Public-Labels" \
    --predictions "${diagnostic_predictions}" --output "${lesion_error}" \
    > "${output}/best_dsc_lesion_error.log" 2>&1
fi
[[ -s "${lesion_error}/summary.json" ]]

"${python_bin}" -u "${root}/select_long_glance_grl_trajectory.py" \
  --run-dir "${run_dir}" --source "${source}" --trajectory-root "${output}" \
  --summary "${summary}" --terminal "${terminal}" \
  --lesion-error-summary "${lesion_error}/summary.json" \
  > "${output}/selection.log"
cat "${summary}"

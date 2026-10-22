#!/usr/bin/env bash
set -euo pipefail

# Low-priority dense internal validation.  The final Glance saver is the last
# GPU consumer in the controlled Focus -> public -> save -> Glance -> public ->
# save chain.  Do not claim either GPU until that consumer is terminal and both
# devices have released model memory.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream="${root}/public_sweeps/gf_glance_grl_long30_saved_20260825/TERMINAL.json"
output="${root}/public_sweeps/gf_focus_pu_long30_internal_grid_20260825"
terminal="${output}/TERMINAL.json"
service_log="${output}/service.log"
idle_memory_mib=4096

mkdir -p "${output}"
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${upstream}" "${service_log}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
reason=sys.argv[1];upstream,log,output=map(Path,sys.argv[2:])
def asset(path):
    return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else None
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"NO_INTERNAL_SHORTLIST","reason":reason,
 "upstream_terminal":asset(upstream),"service_log":asset(log),
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure_line=unknown
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ -s "${terminal}" ]]; then
    preserved="${output}/TERMINAL.unverified_$(date +%Y%m%dT%H%M%S%N).json"
    mv --no-clobber "${terminal}" "${preserved}"
  fi
  write_failure "internal-grid controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED)
      preserved="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved}"
      ;;
    *) echo "unexpected existing internal-grid terminal status: ${status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${upstream}" ]]; then
    printf '%s waiting for final Glance saver terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${upstream}")
  case "${status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed final Glance saver terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected final Glance saver status: ${status}" >&2; exit 2 ;;
  esac
done

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${idle_memory_mib}" \
      && "${used1}" -le "${idle_memory_mib}" ]]; then
    break
  fi
  printf '%s waiting for idle GPUs before internal grid: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

MPLCONFIGDIR="${output}/.matplotlib" \
  "${python_bin}" -u "${root}/evaluate_internal_checkpoint_grid.py" \
    --run-dir "${root}/runs/gf_focus_pu_long30_e283_to313_20260825" \
    --output "${output}" --start-epoch 284 --end-epoch 313 \
    --gpus 0 1 --workers 2 --idle-memory-mib "${idle_memory_mib}" \
    --parity-epoch 295 --parity-tolerance 1e-6 \
    >> "${service_log}" 2>&1

[[ -s "${terminal}" ]]
"${python_bin}" - "${terminal}" <<'PY'
import hashlib,json,sys
from pathlib import Path
payload=json.load(open(sys.argv[1],encoding="utf-8"))
if payload.get("status")!="COMPLETE" or payload.get("evaluated_checkpoints")!=30:
    raise SystemExit("dense internal grid did not complete all 30 checkpoints")
if payload.get("public_read") or payload.get("hidden_read") or payload.get("submission_created"):
    raise SystemExit("dense internal grid crossed its diagnostic-only boundary")
for key in ("learning_curve", "plot", "fixed_public_shortlist"):
    asset=payload.get(key,{})
    path=Path(asset.get("path",""))
    if not path.is_file() or asset.get("sha256")!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit(f"dense internal grid {key} identity mismatch")
curve_asset=payload["learning_curve"]
curve=json.load(open(curve_asset["path"],encoding="utf-8"))
if curve.get("recall_available_checkpoint_count")!=24:
    raise SystemExit("dense internal grid did not retain 24 recall-instrumented checkpoints")
for key in ("best_annotated_large_component_recall_ge3000", "best_annotated_voxel_recall"):
    point=curve.get(key)
    if not isinstance(point,dict) or point.get("epoch") is None:
        raise SystemExit(f"dense internal grid lacks {key}")
shortlist_asset=payload["fixed_public_shortlist"]
shortlist=json.load(open(shortlist_asset["path"],encoding="utf-8"))
if shortlist.get("status")!="COMPLETE" or not 1<=shortlist.get("candidate_count",0)<=3:
    raise SystemExit("dense internal fixed-public shortlist is invalid")
if (shortlist.get("selection_objective")
    !="case_balanced_dice_plus_ge3000mm3_and_voxel_recall"
    or shortlist.get("large_component_minimum_mm3")!=3000):
    raise SystemExit("dense internal shortlist objective mismatch")
allowed_roles={"best_internal_dice","best_large_component_recall_ge3000_within_dice_guard",
               "best_voxel_recall_within_dice_guard"}
for candidate in shortlist.get("candidates",[]):
    roles=set(candidate.get("selection_roles",[]))
    if not roles or not roles<=allowed_roles:
        raise SystemExit("dense internal shortlist contains a deprecated selection role")
if shortlist.get("promotion_authority")!="fixed_public_50_only":
    raise SystemExit("dense internal shortlist crossed its promotion boundary")
if shortlist.get("public_read") or shortlist.get("hidden_read") or shortlist.get("submission_created"):
    raise SystemExit("dense internal shortlist crossed its diagnostic-only boundary")
parity=payload.get("source_parity",{})
if parity.get("status")!="PASS" or parity.get("epoch")!=295:
    raise SystemExit("dense internal grid source-parity check missing")
asset=parity.get("standalone_audit",{})
path=Path(asset.get("path",""))
if not path.is_file() or asset.get("sha256")!=hashlib.sha256(path.read_bytes()).hexdigest():
    raise SystemExit("dense internal grid source-parity identity mismatch")
PY
trap - EXIT

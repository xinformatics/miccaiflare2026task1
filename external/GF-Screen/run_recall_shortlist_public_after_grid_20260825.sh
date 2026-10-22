#!/usr/bin/env bash
set -euo pipefail

# Public-only screen of checkpoints nominated by partial-label-safe internal
# recall diagnostics. The internal grid may nominate candidates, but only the
# immutable fixed public 50 can promote one. Hidden data, ZIP creation,
# networking, and upload are unreachable from this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run_dir="${root}/runs/gf_focus_pu_long30_e283_to313_20260825"
grid_root="${root}/public_sweeps/gf_focus_pu_long30_internal_grid_20260825"
grid_terminal="${grid_root}/TERMINAL.json"
shortlist="${grid_root}/fixed_public_shortlist.json"
output="${root}/public_sweeps/gf_focus_pu_recall_shortlist_public_20260825"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
lock="${root}/runs/run_recall_shortlist_public_after_grid_20260825.lock"
idle_memory_mib=4096

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "recall-shortlist public controller already active" >&2; exit 1; }

write_failure() {
  local status="$1" reason="$2"
  "${python_bin}" - "${status}" "${reason}" "${shortlist}" "${output}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,reason=sys.argv[1:3];shortlist,work,output=map(Path,sys.argv[3:])
def asset(path):
    return {"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else None
public_read=False
if work.is_dir():
    public_read=any(work.rglob("*.nii.gz")) or any(work.glob("epoch_*/*.metrics.csv"))
    for child in work.glob("epoch_*/TERMINAL.json"):
        try:
            public_read=public_read or bool(json.load(open(child,encoding="utf-8")).get("public_read"))
        except Exception:
            public_read=True
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":"KEEP_TOUCHCOMP","reason":reason,
 "internal_shortlist":asset(shortlist),"public_read":public_read,"hidden_read":False,
 "submission_created":False,"uploaded":False,
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
  [[ -s "${terminal}" ]] || write_failure FAILED \
    "recall-shortlist public controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED)
      preserved="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved}"
      ;;
    *) echo "unexpected recall-shortlist terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${grid_terminal}" ]]; then
    printf '%s waiting for recall-aware internal grid\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  grid_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${grid_terminal}")
  case "${grid_status}" in
    COMPLETE) break ;;
    FAILED)
      printf '%s waiting for replacement of failed internal-grid terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected internal-grid status: ${grid_status}" >&2; exit 2 ;;
  esac
done

mapfile -t epochs < <("${python_bin}" - "${grid_terminal}" "${shortlist}" "${run_dir}" <<'PY'
import hashlib,json,sys
from pathlib import Path
grid_path,shortlist_path,run_dir=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
grid=json.load(open(grid_path,encoding="utf-8"));shortlist=json.load(open(shortlist_path,encoding="utf-8"))
identity=grid.get("fixed_public_shortlist",{})
if Path(identity.get("path","")).resolve()!=shortlist_path.resolve() or identity.get("sha256")!=sha(shortlist_path):
    raise SystemExit("internal-grid/shortlist identity mismatch")
candidates=shortlist.get("candidates",[])
if (shortlist.get("status")!="COMPLETE" or shortlist.get("promotion_authority")!="fixed_public_50_only"
    or shortlist.get("selection_objective")!="case_balanced_dice_plus_ge3000mm3_and_voxel_recall"
    or shortlist.get("large_component_minimum_mm3")!=3000
    or shortlist.get("candidate_count")!=len(candidates) or not 1<=len(candidates)<=3
    or shortlist.get("public_read") or shortlist.get("hidden_read") or shortlist.get("submission_created")):
    raise SystemExit("invalid recall-shortlist contract")
allowed_roles={"best_internal_dice","best_large_component_recall_ge3000_within_dice_guard",
               "best_voxel_recall_within_dice_guard"}
seen=set()
for candidate in candidates:
    epoch=int(candidate["epoch"]);checkpoint=run_dir/f"checkpoint_epoch_{epoch:04d}.pt"
    roles=set(candidate.get("selection_roles",[]))
    if (epoch in seen or not roles or not roles<=allowed_roles or not checkpoint.is_file()
        or candidate.get("checkpoint_sha256")!=sha(checkpoint)):
        raise SystemExit(f"invalid recall-shortlist checkpoint {epoch}")
    seen.add(epoch);print(epoch)
PY
)
[[ "${#epochs[@]}" -ge 1 && "${#epochs[@]}" -le 3 ]]

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_for_idle_pair() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le "${idle_memory_mib}" \
        && "${used1}" -le "${idle_memory_mib}" ]]; then
      break
    fi
    printf '%s waiting for public-screen GPUs: gpu0=%s gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}
evaluate_epoch() {
  local epoch="$1" gpu="$2"
  local checkpoint child child_lock
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  child=$(printf '%s/epoch_%04d' "${output}" "${epoch}")
  child_lock=$(printf '%s/runs/evaluate_recall_shortlist_epoch_%04d_20260825.lock' "${root}" "${epoch}")
  LONG_FOCUS_CHECKPOINT="${checkpoint}" \
  LONG_FOCUS_PUBLIC_OUTPUT="${child}" \
  LONG_FOCUS_PUBLIC_GPU="${gpu}" \
  LONG_FOCUS_PUBLIC_LOCK="${child_lock}" \
  bash "${root}/evaluate_long_focus_pu_champion_public_20260825.sh" \
    > "${output}/epoch_${epoch}.log" 2>&1
  [[ -s "${child}/TERMINAL.json" && -s "${child}/summary.tsv" ]]
}

index=0
while [[ "${index}" -lt "${#epochs[@]}" ]]; do
  wait_for_idle_pair
  evaluate_epoch "${epochs[${index}]}" 0 & pid0=$!
  pid1=""
  if [[ $((index + 1)) -lt "${#epochs[@]}" ]]; then
    evaluate_epoch "${epochs[$((index + 1))]}" 1 & pid1=$!
  fi
  status=0
  wait "${pid0}" || status=$?
  if [[ -n "${pid1}" ]]; then
    wait "${pid1}" || status=$?
  fi
  [[ "${status}" -eq 0 ]]
  index=$((index + 2))
done

"${python_bin}" -u "${root}/select_recall_shortlist_public.py" \
  --run-dir "${run_dir}" --shortlist "${shortlist}" \
  --children-root "${output}" --summary "${summary}" --terminal "${terminal}" \
  > "${output}/selection.log"
trap - EXIT
cat "${summary}"

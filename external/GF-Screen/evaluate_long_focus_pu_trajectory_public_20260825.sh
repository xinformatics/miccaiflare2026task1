#!/usr/bin/env bash
set -euo pipefail

# Predeclared checkpoint selection along one materially long Focus trajectory.
# Epochs 293/303/313 represent 10/20/30 new optimizer epochs. All are scored
# with the identical fixed-public deployment family; exactly one checkpoint
# may qualify. Hidden data, packaging, networking, and upload are unreachable.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run_dir="${LONG_FOCUS_RUN_DIR:-${root}/runs/gf_focus_pu_long30_e283_to313_20260825}"
output="${LONG_FOCUS_TRAJECTORY_OUTPUT:-${root}/public_sweeps/gf_focus_pu_long30_trajectory_20260825}"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
lock="${root}/runs/evaluate_long_focus_pu_trajectory_public_20260825.lock"
epochs=(293 303 313)
reuse_validator="${root}/validate_long_focus_interim_reuse.py"

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long Focus public trajectory evaluator already active" >&2; exit 1; }

write_failure() {
  local status="$1" reason="$2"
  "${python_bin}" - "${status}" "${reason}" "${run_dir}" "${summary}" \
    "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,reason=sys.argv[1:3];run_dir,summary,output=map(Path,sys.argv[3:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
children=[]
for epoch in (293,303,313):
    path=output.parent / f"epoch_{epoch:04d}" / "TERMINAL.json"
    children.append({"epoch":epoch,"terminal":asset(path)})
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":"KEEP_TOUCHCOMP","reason":reason,
 "predeclared_epochs":[293,303,313],"selected_epoch":None,
 "selected_variant":None,"checkpoint":None,"selected_public_terminal":None,
 "summary":asset(summary),"children":children,
 "fixed_public_DSC_gate":0.6154316379500749,
 "fixed_public_NSD_floor":0.5104225107769019,
 "public_read":summary.is_file(),"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED)
      preserved_terminal="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved_terminal}"
      printf '%s preserved failed trajectory terminal for exact resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing trajectory terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_failure FAILED \
    "trajectory public evaluation failed at line ${line}, status ${status}"
  exit "${status}"
}
trap failure EXIT

# Epochs 293 and 303 were already screened with this exact evaluator while the
# long run was paused. Reuse is allowed only after a strict checkpoint,
# deployment, 50-case, metric, gate, and public-only provenance validation.
# Epoch 313 has no cache and is always inferred fresh.
for epoch in 293 303; do
  source="${root}/public_sweeps/gf_focus_pu_long30_e${epoch}_interim_20260825"
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  child=$(printf '%s/epoch_%04d' "${output}" "${epoch}")
  "${python_bin}" -u "${reuse_validator}" --source "${source}" \
    --checkpoint "${checkpoint}" --epoch "${epoch}" \
    > "${output}/epoch_${epoch}_reuse_validation.json"
  if [[ ! -e "${child}" && ! -L "${child}" ]]; then
    ln -s "${source}" "${child}"
  fi
  [[ -L "${child}" && "$(readlink -f "${child}")" == "$(readlink -f "${source}")" ]]
done

for epoch in "${epochs[@]}"; do
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  [[ -s "${checkpoint}" ]] || { echo "missing trajectory checkpoint: ${checkpoint}" >&2; exit 2; }
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for both trajectory-evaluation GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

evaluate_epoch() {
  local epoch="$1" gpu="$2"
  local checkpoint child child_lock
  checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${run_dir}" "${epoch}")
  child=$(printf '%s/epoch_%04d' "${output}" "${epoch}")
  child_lock=$(printf '%s/runs/evaluate_long_focus_pu_epoch_%04d_20260825.lock' "${root}" "${epoch}")
  LONG_FOCUS_CHECKPOINT="${checkpoint}" \
  LONG_FOCUS_PUBLIC_OUTPUT="${child}" \
  LONG_FOCUS_PUBLIC_GPU="${gpu}" \
  LONG_FOCUS_PUBLIC_LOCK="${child_lock}" \
  bash "${root}/evaluate_long_focus_pu_champion_public_20260825.sh" \
    > "${output}/epoch_${epoch}.log" 2>&1
  [[ -s "${child}/TERMINAL.json" && -s "${child}/summary.tsv" ]]
}

# Two checkpoints are inferred concurrently, one per GPU; the third follows
# after both complete. This changes wall time only, never predictions.
evaluate_epoch 293 0 & pid293=$!
evaluate_epoch 303 1 & pid303=$!
status=0
wait "${pid293}" || status=$?
wait "${pid303}" || status=$?
[[ "${status}" -eq 0 ]]
evaluate_epoch 313 0

# Diagnose the mechanism on exactly one output: the highest-DSC trajectory
# deployment. This audit is descriptive only and cannot alter the promotion
# gate or selected checkpoint.
read -r diagnostic_epoch diagnostic_variant diagnostic_predictions < <(
  "${python_bin}" - "${output}" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1]);rows=[]
for epoch in (293,303,313):
    with open(root/f"epoch_{epoch:04d}"/"summary.tsv",encoding="utf-8") as handle:
        for row in csv.DictReader(handle,delimiter="\t"):
            rows.append((float(row["DSC"]),float(row["NSD"]),epoch,row))
rows.sort(key=lambda item:(item[0],item[1]),reverse=True)
_,_,epoch,row=rows[0]
print(epoch,row["variant"],row["predictions"],sep="\t")
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

"${python_bin}" -u "${root}/select_long_focus_pu_trajectory.py" \
  --run-dir "${run_dir}" --trajectory-root "${output}" \
  --summary "${summary}" --terminal "${terminal}" \
  --lesion-error-summary "${lesion_error}/summary.json" \
  > "${output}/selection.log"
trap - EXIT
cat "${summary}"

#!/usr/bin/env bash
set -euo pipefail

# Run the public-only exhaustive pseudo-label teacher diagnostic only after the
# active unlabeled-Glance baseline has fully evaluated and saved/rejected.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${root}/public_sweeps/gf_glance_unlabeled_saved_20260822/SAVE_ONLY_COMPLETE.txt"
output="${root}/public_sweeps/exhaustive_teacher_discovery_20260822"

exec 9>"${root}/runs/run_exhaustive_teacher_audit_after_unlabeled_20260822.lock"
flock -n 9 || { echo "exhaustive teacher controller already active" >&2; exit 1; }
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for unlabeled-Glance save-only completion\n' "$(date -Ins)"
  sleep 30
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0)
  used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
bash "${root}/evaluate_exhaustive_teacher_discovery_public_20260822.sh"
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]]

#!/usr/bin/env bash
set -euo pipefail

# Give the cheap qualified-module composition/audit priority over the later,
# long unlabeled-data training branch. It never uploads to Codabench.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_marker="${root}/public_sweeps/gf_focus_allpartial_saved_20260822/SAVE_ONLY_COMPLETE.txt"
glance_marker="${root}/public_sweeps/gf_glance_boxloc_saved_20260822/SAVE_ONLY_COMPLETE.txt"
log="${root}/qualified_module_composite_controller_20260822.log"

exec 9>"${root}/runs/run_qualified_module_composite_after_focus_20260822.lock"
flock -n 9 || { echo "qualified module composite controller already active" >&2; exit 1; }
while [[ ! -s "${focus_marker}" || ! -s "${glance_marker}" ]]; do
  printf '%s waiting for qualified module save stages\n' "$(date -Ins)"
  sleep 5
done
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 5
done
bash "${root}/prepare_qualified_module_composite_20260822.sh"
CUDA_DEVICE=0 bash "${root}/evaluate_qualified_module_composite_public_20260822.sh"
bash "${root}/save_qualified_module_composite_20260822.sh"
date -Ins > "${root}/public_sweeps/qualified_focus_glance_queue_20260822_COMPLETE.txt"
printf '%s qualified composite queue complete\n' "$(date -Ins)" >> "${log}"

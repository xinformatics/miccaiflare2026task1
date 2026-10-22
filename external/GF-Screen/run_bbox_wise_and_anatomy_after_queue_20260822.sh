#!/usr/bin/env bash
set -euo pipefail

# Serialize new public-only work after both the component-background Glance
# saver and its qualified champion-intersection saver have finished hidden
# inference, ZIP validation, and atomic queue refresh.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${root}/public_sweeps/gf_cbg_intersection_saved_20260822/SAVE_ONLY_COMPLETE.txt"
wise_marker="${root}/public_sweeps/a055_a069_bbox_wise_20260822/PUBLIC_ONLY_COMPLETE.txt"
log="${root}/bbox_wise_and_anatomy_queue_20260822.log"

exec 9>"${root}/runs/run_bbox_wise_and_anatomy_after_queue_20260822.lock"
flock -n 9 || { echo "bbox-WiSE/anatomy queue already active" >&2; exit 1; }
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for CBG intersection save-only completion\n' "$(date -Ins)"
  sleep 30
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
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

bash "${root}/evaluate_a055_a069_bbox_interpolations_20260822.sh"
[[ -s "${wise_marker}" ]]
bash "${root}/save_wise_candidate_20260822.sh"
[[ -s "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/SAVE_ONLY_COMPLETE.txt" ]]

# TotalSegmentator uses only GPU 0; check again in case another user acquired
# the device while the interpolation audit was finishing.
while :; do
  used0=$(gpu_memory_mib 0)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU 0 before TotalSegmentator: %s MiB\n' "$(date -Ins)" "${used0}"
  sleep 30
done
bash "${root}/run_totalseg_anatomy_component_cv_20260822.sh"
date -Ins > "${root}/public_sweeps/bbox_wise_and_anatomy_queue_20260822_COMPLETE.txt"
printf '%s queue complete\n' "$(date -Ins)" >> "${log}"

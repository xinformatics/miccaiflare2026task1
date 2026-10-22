#!/usr/bin/env bash
set -euo pipefail

# Read-only early cohort-transition audit. It waits until 16 finalized MSD
# masks exist at every nested threshold, then validates only MSD source images.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
teacher="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825/teacher_nested"
output="${root}/analysis/train_unlabel_msd_teacher_transition_audit_16case_20260825.json"
lock="${root}/runs/watch_msd_teacher_snapshot_20260825.lock"
minimum=16

mkdir -p "${root}/analysis" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "MSD teacher-transition audit already active" >&2; exit 1; }
[[ -s "${output}" ]] && exit 0

while :; do
  ready=1
  counts=()
  for threshold in 0.85 0.90 0.95 0.97; do
    count=$(find "${teacher}/lesion_${threshold}" -maxdepth 1 \
      -type f -name '*.nii.gz' ! -name 'amos_*.nii.gz' \
      ! -name '*_trans.nii.gz' | wc -l)
    counts+=("${count}")
    [[ "${count}" -ge "${minimum}" ]] || ready=0
  done
  [[ "${ready}" -eq 1 ]] && break
  printf '%s waiting for first %s finalized MSD masks: %s/%s/%s/%s\n' \
    "$(date -Ins)" "${minimum}" \
    "${counts[0]}" "${counts[1]}" "${counts[2]}" "${counts[3]}"
  sleep 30
done

PYTHONPATH="${root}" "${python_bin}" -u \
  "${root}/audit_nested_teacher_masks.py" \
  --images "${project}/train_unlabel/MSD-506" \
  --threshold-dir "0.85=${teacher}/lesion_0.85" \
  --threshold-dir "0.90=${teacher}/lesion_0.90" \
  --threshold-dir "0.95=${teacher}/lesion_0.95" \
  --threshold-dir "0.97=${teacher}/lesion_0.97" \
  --output "${output}" --max-cases "${minimum}"

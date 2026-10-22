#!/usr/bin/env bash
set -euo pipefail

# Scheduling-only watcher: release downstream GPU evaluation after every raw
# teacher mask exists, without pretending CPU pseudo materialization is done.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
teacher="${work}/teacher_nested"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
audit="${work}/train_unlabel_audit.json"
terminal="${work}/TEACHER_INFERENCE_COMPLETE.json"
lock="${root}/runs/watch_teacher_inference_complete_20260825.lock"

mkdir -p "${work}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "teacher inference completion watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while :; do
  complete=1
  counts=()
  for threshold in 0.85 0.90 0.95 0.97; do
    # SaveImaged writes ``*_trans.nii.gz`` before the inference process
    # atomically renames it to the final prediction name.  Counting that live
    # intermediate can release the GPU consumer one case too early and make
    # the exact inventory verifier fail permanently at the finish line.
    count=$(find "${teacher}/lesion_${threshold}" -maxdepth 1 \
      -name '*.nii.gz' ! -name '*_trans.nii.gz' 2>/dev/null | wc -l)
    counts+=("${count}")
    [[ "${count}" -eq 2856 ]] || complete=0
  done
  [[ "${complete}" -eq 1 ]] && break
  printf '%s waiting for exact teacher inventories: %s/%s/%s/%s\n' \
    "$(date -Ins)" "${counts[0]}" "${counts[1]}" "${counts[2]}" "${counts[3]}"
  sleep 30
done
for required in "${checkpoint}" "${audit}"; do
  [[ -s "${required}" ]] || { echo "missing teacher-release asset: ${required}" >&2; exit 2; }
done
PYTHONPATH="${root}" "${python_bin}" -u "${root}/mark_teacher_inference_complete.py" \
  --images "${project}/train_unlabel" --teacher-root "${teacher}" \
  --checkpoint "${checkpoint}" --source-audit "${audit}" \
  --output "${terminal}" --expected-count 2856

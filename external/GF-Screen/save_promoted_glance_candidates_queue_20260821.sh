#!/usr/bin/env bash
set -euo pipefail

# Save-only queue for the teacher-safe Glance audit. It never uploads. Public
# paired robustness must say PROMOTE before any hidden case is inferred.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
audit="${root}/public_sweeps/gf_glance_teacher_safe_deployed_20260821"
marker="${audit}/PUBLIC_ONLY_COMPLETE.txt"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
queue_log="${audit}/saved_candidates.tsv"

exec 9>"${root}/runs/save_promoted_glance_candidates_queue_20260821.lock"
if ! flock -n 9; then
  echo "promoted Glance save queue is already active" >&2
  exit 1
fi
while [[ ! -s "${marker}" ]]; do
  sleep 30
done

summary="${audit}/summary.tsv"
[[ -s "${summary}" ]]
printf 'rank\ttag\tcheckpoint\tglance_threshold\tmetrics\trobust\tsubmission\n' \
  > "${queue_log}"

rank=0
while IFS=$'\t' read -r variant checkpoint_tag checkpoint glance dsc nsd \
    combined metrics robust; do
  [[ "${variant}" == "variant" ]] && continue
  [[ -s "${checkpoint}" && -s "${metrics}" && -s "${robust}" ]] || continue
  decision=$(awk -F'"' '/"decision":/ {print $4}' "${robust}")
  [[ "${decision}" == "PROMOTE" ]] || continue
  rank=$((rank + 1))
  [[ "${variant}" == *posonly* ]] && arm=po || arm=pn
  [[ "${checkpoint_tag}" == "model_final" ]] && state=fin || state=best
  glance_tag="${glance/./}"
  tag="gf_gsafe_${arm}_${state}_g${glance_tag}"
  CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
    ROBUST_JSON="${robust}" GLANCE_THRESHOLD="${glance}" \
    LESION_THRESHOLD=0.50 bash "${materializer}"
  submission="${root}/../../submissions/${tag}.zip"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${rank}" "${tag}" \
    "${checkpoint}" "${glance}" "${metrics}" "${robust}" "${submission}" \
    >> "${queue_log}"
  [[ "${rank}" -ge 3 ]] && break
done < <(sort -t $'\t' -k7,7gr -k5,5gr "${summary}")

if [[ "${rank}" -eq 0 ]]; then
  printf '%s\n' 'No Glance candidate passed the paired robustness gate; no ZIP saved.' \
    > "${audit}/NO_PROMOTED_SUBMISSION.txt"
fi

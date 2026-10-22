#!/usr/bin/env bash
set -euo pipefail

# Save-only watcher for the cohort-aware Focus, Glance, and merged checkpoint
# audits. It materializes at most three strictly PROMOTE candidates and never
# uploads them. Family diversity is preferred over correlated threshold copies.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus_audit="${root}/public_sweeps/gf_hybrid_complete04_08_deployed_20260822"
glance_audit="${root}/public_sweeps/gf_glance_hybrid04_lr_deployed_20260822"
cross_audit="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
plan="${cross_audit}/submission_plan.tsv"
saved="${cross_audit}/saved_submissions.tsv"

mkdir -p "${cross_audit}"
exec 9>"${root}/runs/save_hybrid_submission_queue_20260822.lock"
flock -n 9 || { echo "hybrid submission save queue is already active" >&2; exit 1; }
[[ -s "${cross_audit}/SAVED_SUBMISSIONS_COMPLETE.txt" ]] && exit 0

for marker in "${focus_audit}/PUBLIC_ONLY_COMPLETE.txt" \
    "${glance_audit}/PUBLIC_ONLY_COMPLETE.txt" \
    "${cross_audit}/PUBLIC_ONLY_COMPLETE.txt"; do
  while [[ ! -s "${marker}" ]]; do
    printf '%s waiting for %s\n' "$(date -Ins)" "${marker}"
    sleep 30
  done
done

"${python_bin}" "${root}/build_hybrid_submission_plan.py" \
  --focus-summary "${focus_audit}/summary.tsv" \
  --glance-summary "${glance_audit}/summary.tsv" \
  --cross-summary "${cross_audit}/summary.tsv" \
  --output "${plan}" --count 3 > "${cross_audit}/submission_plan.log"

candidate_count=$(tail -n +2 "${plan}" | sed '/^$/d' | wc -l)
if [[ "${candidate_count}" -eq 0 ]]; then
  printf '%s\n' 'No hybrid candidate passed the strict paired PROMOTE gate; no ZIP saved.' \
    > "${cross_audit}/NO_PROMOTED_SUBMISSION.txt"
  date -Ins > "${cross_audit}/SAVED_SUBMISSIONS_COMPLETE.txt"
  exit 0
fi

# Rewrite the saved manifest with the materialized archive and checksum fields.
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tsubmission\tsha256\n' \
  > "${saved}"
while IFS=$'\t' read -r rank tag family checkpoint glance lesion dsc nsd metrics robust; do
  [[ "${rank}" == "rank" ]] && continue
  CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
    ROBUST_JSON="${robust}" GLANCE_THRESHOLD="${glance}" \
    LESION_THRESHOLD="${lesion}" bash "${materializer}"
  submission="${project}/submissions/${tag}.zip"
  [[ -s "${submission}" ]]
  unzip -t "${submission}" > "${submission}.zip_test.log"
  [[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
  digest=$(sha256sum "${submission}" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${rank}" "${tag}" "${family}" "${checkpoint}" "${glance}" "${lesion}" \
    "${dsc}" "${nsd}" "${metrics}" "${robust}" "${submission}" "${digest}" \
    >> "${saved}"
done < "${plan}"
date -Ins > "${cross_audit}/SAVED_SUBMISSIONS_COMPLETE.txt"

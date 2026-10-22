#!/usr/bin/env bash
set -euo pipefail

# Fill unused capacity with diverse near-anchor backups after the strict
# save-only queue completes. This script never uploads to Codabench.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus_audit="${root}/public_sweeps/gf_hybrid_complete04_08_deployed_20260822"
glance_audit="${root}/public_sweeps/gf_glance_hybrid04_lr_deployed_20260822"
cross_audit="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822"
strict_marker="${cross_audit}/SAVED_SUBMISSIONS_COMPLETE.txt"
strict_saved="${cross_audit}/saved_submissions.tsv"
plan="${cross_audit}/backup_submission_plan.tsv"
saved="${cross_audit}/saved_backup_submissions.tsv"
marker="${cross_audit}/SAVED_BACKUPS_COMPLETE.txt"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"

mkdir -p "${cross_audit}"
exec 9>"${root}/runs/save_hybrid_backup_queue_20260822.lock"
flock -n 9 || { echo "hybrid backup save queue is already active" >&2; exit 1; }
[[ -s "${marker}" ]] && exit 0
while [[ ! -s "${strict_marker}" ]]; do
  printf '%s waiting for strict hybrid save completion\n' "$(date -Ins)"
  sleep 30
done

strict_count=0
if [[ -s "${strict_saved}" ]]; then
  strict_count=$(tail -n +2 "${strict_saved}" | sed '/^$/d' | wc -l)
fi
remaining=$((3 - strict_count))
if [[ "${remaining}" -le 0 ]]; then
  printf '%s\n' 'Strict promotions already occupy all three saved slots.' \
    > "${cross_audit}/NO_BACKUP_CAPACITY.txt"
  date -Ins > "${marker}"
  exit 0
fi

planner_args=(
  --focus-summary "${focus_audit}/summary.tsv"
  --glance-summary "${glance_audit}/summary.tsv"
  --cross-summary "${cross_audit}/summary.tsv"
  --output "${plan}" --count "${remaining}"
)
[[ -s "${strict_saved}" ]] && planner_args+=(--saved "${strict_saved}")
"${python_bin}" "${root}/build_hybrid_backup_plan.py" "${planner_args[@]}" \
  > "${cross_audit}/backup_submission_plan.log"

candidate_count=$(tail -n +2 "${plan}" | sed '/^$/d' | wc -l)
if [[ "${candidate_count}" -eq 0 ]]; then
  printf '%s\n' 'No near-anchor backup passed the bounded fallback gate.' \
    > "${cross_audit}/NO_ELIGIBLE_BACKUP.txt"
  date -Ins > "${marker}"
  exit 0
fi

printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' \
  > "${saved}"
while IFS=$'\t' read -r rank tag family checkpoint glance lesion dsc nsd metrics robust gate; do
  [[ "${rank}" == "rank" ]] && continue
  CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
    ROBUST_JSON="${robust}" GLANCE_THRESHOLD="${glance}" \
    LESION_THRESHOLD="${lesion}" ALLOW_FALLBACK=1 bash "${materializer}"
  submission="${project}/submissions/${tag}.zip"
  [[ -s "${submission}" ]]
  unzip -t "${submission}" > "${submission}.zip_test.log"
  [[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
  digest=$(sha256sum "${submission}" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${rank}" "${tag}" "${family}" "${checkpoint}" "${glance}" "${lesion}" \
    "${dsc}" "${nsd}" "${metrics}" "${robust}" "${gate}" \
    "${submission}" "${digest}" >> "${saved}"
done < "${plan}"
date -Ins > "${marker}"

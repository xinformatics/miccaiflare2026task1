#!/usr/bin/env bash
set -euo pipefail

# Save (never upload) the strongest alpha=0.69 candidate after the complete
# public trajectory/threshold audits.  Hidden inference is permitted only when
# the candidate either passes the strict robust gate or is no worse than the
# strongest already-saved backup on both public metrics.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
bracket="${root}/public_sweeps/gf_a069_cp276_threshold_bracket_20260822"
trajectory="${root}/public_sweeps/gf_a069_sparse_trajectory_20260822"
appearance="${root}/public_sweeps/gf_v7_appearance_deployed_20260822"
output="${root}/public_sweeps/gf_a069_saved_candidate_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
existing_backup="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv"
strict_saved="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"

mkdir -p "${output}"
exec 9>"${root}/runs/save_a069_candidate_after_audits_20260822.lock"
flock -n 9 || { echo "alpha=0.69 save-only controller already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0

for marker in "${bracket}/PUBLIC_ONLY_COMPLETE.txt" \
              "${trajectory}/PUBLIC_ONLY_COMPLETE.txt"; do
  while [[ ! -s "${marker}" ]]; do
    printf '%s waiting for %s\n' "$(date -Ins)" "${marker}"
    sleep 30
  done
done

checkpoint276="${root}/runs/gf_a069_bbox_pu_lr3e5_5ep_20260822/checkpoint_epoch_0276.pt"
candidates="${output}/all_public_candidates.tsv"
printf 'epoch\tcheckpoint\tlesion_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${candidates}"
while IFS=$'\t' read -r threshold dsc nsd combined metrics robust decision; do
  [[ "${threshold}" == "threshold" ]] && continue
  printf '276\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${checkpoint276}" "${threshold}" "${dsc}" "${nsd}" "${combined}" \
    "${metrics}" "${robust}" "${decision}" >> "${candidates}"
done < "${bracket}/summary.tsv"
while IFS=$'\t' read -r epoch checkpoint dsc nsd combined metrics robust decision; do
  [[ "${epoch}" == "epoch" || "${epoch}" == "276" ]] && continue
  printf '%s\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${epoch}" "${checkpoint}" "${dsc}" "${nsd}" "${combined}" \
    "${metrics}" "${robust}" "${decision}" >> "${candidates}"
done < "${trajectory}/summary.tsv"

{
  head -1 "${candidates}"
  tail -n +2 "${candidates}" | sort -t $'\t' -k6,6gr -k4,4gr -k5,5gr
} > "${output}/ranked_public_candidates.tsv"
best=$(sed -n '2p' "${output}/ranked_public_candidates.tsv")
[[ -n "${best}" ]] || { echo "no alpha candidate rows" >&2; exit 2; }
IFS=$'\t' read -r epoch checkpoint lesion dsc nsd combined metrics robust decision \
  <<< "${best}"

# The existing best unsent backup is 0.6072 DSC / 0.5120 NSD.  Do not spend
# another hidden inference on a dominated alpha state.  A formal PROMOTE
# decision always remains eligible.
eligible=$(awk -v d="${dsc}" -v n="${nsd}" -v gate="${decision}" \
  'BEGIN { print (gate == "PROMOTE" || (d >= 0.6072 && n >= 0.5120)) ? "yes" : "no" }')
if [[ "${eligible}" != "yes" ]]; then
  printf 'epoch\tlesion_threshold\tDSC\tNSD\tcombined\tdecision\treason\n' \
    > "${output}/NO_ELIGIBLE_ALPHA_CANDIDATE.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tdominated_by_saved_focus_backup\n' \
    "${epoch}" "${lesion}" "${dsc}" "${nsd}" "${combined}" "${decision}" \
    >> "${output}/NO_ELIGIBLE_ALPHA_CANDIDATE.tsv"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi

# The appearance experiment owns both GPUs immediately after alpha endpoint
# evaluation.  Waiting for its durable marker avoids a launch race and still
# leaves the archive ready many hours before the 20:00 submission renewal.
while [[ ! -s "${appearance}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s alpha candidate eligible; waiting for appearance GPU queue\n' "$(date -Ins)"
  sleep 30
done

lesion_tag=${lesion/./p}
tag="gf_bak_a069_e${epoch}_l${lesion_tag}"
ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
  ROBUST_JSON="${robust}" GLANCE_THRESHOLD=0.50 LESION_THRESHOLD="${lesion}" \
  bash "${materializer}"

submission="${project}/submissions/${tag}.zip"
[[ -s "${submission}" ]]
unzip -t "${submission}" > "${submission}.zip_test.log"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')

saved="${output}/saved_alpha_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' \
  > "${saved}"
printf '1\t%s\talpha069\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoint}" "${lesion}" "${dsc}" "${nsd}" "${metrics}" \
  "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

combined_backup="${output}/combined_backup_submissions.tsv"
head -1 "${existing_backup}" > "${combined_backup}"
tail -n +2 "${existing_backup}" >> "${combined_backup}"
tail -n +2 "${saved}" >> "${combined_backup}"

# Refresh the three-item local upload queue atomically after the new archive
# has passed filename, member-name, count, CRC, binary, and content checks.
ready_tmp="${project}/submissions/READY_20260822.alpha.tmp.tsv"
audit_tmp="${project}/submissions/READY_20260822.alpha.tmp.audit.json"
"${python_bin}" "${root}/finalize_saved_submission_queue.py" \
  --strict "${strict_saved}" --backup "${combined_backup}" \
  --submitted "${submitted}" --output "${ready_tmp}" \
  --audit-output "${audit_tmp}" --expected-count 100 --max-count 3 \
  > "${output}/refresh_ready_queue.log"
mv "${ready_tmp}" "${project}/submissions/READY_20260822.tsv"
mv "${audit_tmp}" "${project}/submissions/READY_20260822.audit.json"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
echo "saved alpha submission candidate: ${submission}"

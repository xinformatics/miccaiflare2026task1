#!/usr/bin/env bash
set -euo pipefail

# Save (never upload) the strongest verified7 appearance candidate after the
# complete public grid. Hidden inference is allowed only for a formal promotion
# or a tail-safe state whose balanced public evidence can enter the current
# three-item upload queue.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_v7_appearance_deployed_20260822"
output="${root}/public_sweeps/gf_v7_appearance_saved_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
existing_backup="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv"
strict_saved="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
ready_manifest="${project}/submissions/READY_20260822.tsv"

mkdir -p "${output}"
exec 9>"${root}/runs/save_v7_appearance_candidate_20260822.lock"
flock -n 9 || { echo "verified7 appearance save-only controller already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0

while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for complete appearance public grid\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${audit}/summary.tsv" ]] || { echo "missing appearance summary" >&2; exit 2; }

{
  head -1 "${audit}/summary.tsv"
  tail -n +2 "${audit}/summary.tsv" | sort -t $'\t' -k7,7gr -k5,5gr -k6,6gr
} > "${output}/ranked_public_candidates.tsv"

# Match the finalizer's evidence order. If the queue is already full, a
# fallback must exceed its weakest balanced DSC/NSD score; if it is not full,
# any tail-safe complete-grid winner may fill the open slot.
ready_count=0
queue_floor=-1
if [[ -s "${ready_manifest}" ]]; then
  ready_count=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' "${ready_manifest}")
  if [[ "${ready_count}" -ge 3 ]]; then
    queue_floor=$(awk -F '\t' '
      NR == 2 { floor = ($5 + $6) / 2 }
      NR > 2 { score = ($5 + $6) / 2; if (score < floor) floor = score }
      END { printf "%.9f", floor }
    ' "${ready_manifest}")
  fi
fi
printf 'ready_count=%s\tbalanced_queue_floor=%s\n' "${ready_count}" "${queue_floor}" \
  > "${output}/queue_floor.tsv"

selected=""
while IFS=$'\t' read -r arm checkpoint_tag checkpoint lesion dsc nsd combined metrics robust; do
  [[ "${arm}" == "variant" ]] && continue
  read -r decision tail_guard < <("${python_bin}" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(x["decision"], str(x.get("checks", {}).get("tail_regression_guard", False)).lower())' \
    "${robust}")
  eligible=$(awk -v c="${combined}" -v floor="${queue_floor}" \
    -v gate="${decision}" -v guard="${tail_guard}" \
    'BEGIN { print (gate == "PROMOTE" || (guard == "true" && c > floor + 0.000001)) ? "yes" : "no" }')
  if [[ "${eligible}" == "yes" ]]; then
    selected=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${arm}" "${checkpoint_tag}" "${checkpoint}" "${lesion}" "${dsc}" \
      "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}")
    break
  fi
done < "${output}/ranked_public_candidates.tsv"

if [[ -z "${selected}" ]]; then
  best=$(sed -n '2p' "${output}/ranked_public_candidates.tsv")
  printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\treason\n%s\tdoes_not_enter_current_queue_or_fails_tail_guard\n' "${best}" \
    > "${output}/NO_ELIGIBLE_APPEARANCE_CANDIDATE.tsv"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi

IFS=$'\t' read -r arm checkpoint_tag checkpoint lesion dsc nsd combined metrics robust decision \
  <<< "${selected}"
if [[ "${arm}" == *"charm"* ]]; then family="v7ch"; else family="v7a"; fi
epoch=${checkpoint_tag##*_}
epoch=${epoch#0}
epoch=${epoch#0}
lesion_tag=${lesion/./p}
tag="gf_${family}_e${epoch}_l${lesion_tag}"

# This controller and the following Glance experiment share this durable
# marker as their serialization boundary, so no separate GPU race is possible.
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
  printf '%s waiting for GPUs before appearance materialization: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
  ROBUST_JSON="${robust}" GLANCE_THRESHOLD=0.50 LESION_THRESHOLD="${lesion}" \
  bash "${materializer}"

submission="${project}/submissions/${tag}.zip"
[[ -s "${submission}" ]]
unzip -t "${submission}" > "${submission}.zip_test.log"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
saved="${output}/saved_appearance_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' \
  > "${saved}"
printf '1\t%s\t%s\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoint}" "${lesion}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

combined_strict="${output}/combined_strict_submissions.tsv"
combined_backup="${output}/combined_backup_submissions.tsv"
if [[ -s "${strict_saved}" ]]; then
  head -1 "${strict_saved}" > "${combined_strict}"
  tail -n +2 "${strict_saved}" >> "${combined_strict}"
else
  # No strict promotion exists yet. Preserve the common schema so the
  # finalizer can still consume an empty strict tier.
  head -1 "${existing_backup}" > "${combined_strict}"
fi
head -1 "${existing_backup}" > "${combined_backup}"
tail -n +2 "${existing_backup}" >> "${combined_backup}"
if [[ "${decision}" == "PROMOTE" ]]; then
  tail -n +2 "${saved}" >> "${combined_strict}"
else
  tail -n +2 "${saved}" >> "${combined_backup}"
fi
ready_tmp="${project}/submissions/READY_20260822.appearance.tmp.tsv"
audit_tmp="${project}/submissions/READY_20260822.appearance.tmp.audit.json"
"${python_bin}" "${root}/finalize_saved_submission_queue.py" \
  --strict "${combined_strict}" --backup "${combined_backup}" \
  --submitted "${submitted}" --output "${ready_tmp}" \
  --audit-output "${audit_tmp}" --expected-count 100 --max-count 3 \
  > "${output}/refresh_ready_queue.log"
mv "${ready_tmp}" "${project}/submissions/READY_20260822.tsv"
mv "${audit_tmp}" "${project}/submissions/READY_20260822.audit.json"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
echo "saved appearance submission candidate: ${submission}"

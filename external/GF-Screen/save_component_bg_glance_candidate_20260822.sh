#!/usr/bin/env bash
set -euo pipefail

# Save (never upload) the strongest component-positive/background Glance state
# after its complete public grid, preserving any earlier appearance candidate.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_glance_component_bg_deployed_20260822"
appearance_saved="${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv"
output="${root}/public_sweeps/gf_glance_component_bg_saved_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
existing_backup="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv"
existing_strict="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
ready_manifest="${project}/submissions/READY_20260822.tsv"

mkdir -p "${output}"
exec 9>"${root}/runs/save_component_bg_glance_candidate_20260822.lock"
flock -n 9 || { echo "component-background Glance saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for complete component-background Glance grid\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${audit}/summary.tsv" ]] || { echo "missing Glance summary" >&2; exit 2; }

{
  head -1 "${audit}/summary.tsv"
  tail -n +2 "${audit}/summary.tsv" | sort -t $'\t' -k7,7gr -k5,5gr -k6,6gr
} > "${output}/ranked_public_candidates.tsv"

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
while IFS=$'\t' read -r arm checkpoint_tag checkpoint glance dsc nsd combined metrics robust; do
  [[ "${arm}" == "variant" ]] && continue
  read -r decision tail_guard < <("${python_bin}" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(x["decision"], str(x.get("checks", {}).get("tail_regression_guard", False)).lower())' \
    "${robust}")
  eligible=$(awk -v c="${combined}" -v floor="${queue_floor}" \
    -v gate="${decision}" -v guard="${tail_guard}" \
    'BEGIN { print (gate == "PROMOTE" || (guard == "true" && c > floor + 0.000001)) ? "yes" : "no" }')
  if [[ "${eligible}" == "yes" ]]; then
    selected=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${arm}" "${checkpoint_tag}" "${checkpoint}" "${glance}" "${dsc}" \
      "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}")
    break
  fi
done < "${output}/ranked_public_candidates.tsv"

if [[ -z "${selected}" ]]; then
  best=$(sed -n '2p' "${output}/ranked_public_candidates.tsv")
  printf 'variant\tcheckpoint_tag\tcheckpoint\tglance_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\treason\n%s\tdoes_not_enter_current_queue_or_fails_tail_guard\n' "${best}" \
    > "${output}/NO_ELIGIBLE_GLANCE_CANDIDATE.tsv"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi

IFS=$'\t' read -r arm checkpoint_tag checkpoint glance dsc nsd combined metrics robust decision \
  <<< "${selected}"
if [[ "${arm}" == *"p4n1"* ]]; then family="gcb41"; else family="gcb11"; fi
epoch=${checkpoint_tag##*_}
epoch=${epoch#0}
glance_tag=${glance/./p}
tag="gf_${family}_e${epoch}_g${glance_tag}"

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
  printf '%s waiting for GPUs before Glance materialization: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
  ROBUST_JSON="${robust}" GLANCE_THRESHOLD="${glance}" LESION_THRESHOLD=0.50 \
  bash "${materializer}"

submission="${project}/submissions/${tag}.zip"
[[ -s "${submission}" ]]
unzip -t "${submission}" > "${submission}.zip_test.log"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
saved="${output}/saved_glance_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' \
  > "${saved}"
printf '1\t%s\t%s\t%s\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoint}" "${glance}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

combined_strict="${output}/combined_strict_submissions.tsv"
combined_backup="${output}/combined_backup_submissions.tsv"
if [[ -s "${existing_strict}" ]]; then
  head -1 "${existing_strict}" > "${combined_strict}"
  tail -n +2 "${existing_strict}" >> "${combined_strict}"
else
  head -1 "${existing_backup}" > "${combined_strict}"
fi
head -1 "${existing_backup}" > "${combined_backup}"
tail -n +2 "${existing_backup}" >> "${combined_backup}"
for candidate_file in "${appearance_saved}" "${saved}"; do
  [[ -s "${candidate_file}" ]] || continue
  while IFS=$'\t' read -r rank row_tag row_family row_checkpoint row_glance row_lesion row_dsc row_nsd row_metrics row_robust row_gate row_submission row_sha; do
    [[ "${rank}" == "rank" ]] && continue
    if [[ "${row_gate}" == "PROMOTE" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${rank}" "${row_tag}" "${row_family}" "${row_checkpoint}" "${row_glance}" \
        "${row_lesion}" "${row_dsc}" "${row_nsd}" "${row_metrics}" "${row_robust}" \
        "${row_gate}" "${row_submission}" "${row_sha}" >> "${combined_strict}"
    else
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${rank}" "${row_tag}" "${row_family}" "${row_checkpoint}" "${row_glance}" \
        "${row_lesion}" "${row_dsc}" "${row_nsd}" "${row_metrics}" "${row_robust}" \
        "${row_gate}" "${row_submission}" "${row_sha}" >> "${combined_backup}"
    fi
  done < "${candidate_file}"
done

ready_tmp="${project}/submissions/READY_20260822.glance.tmp.tsv"
audit_tmp="${project}/submissions/READY_20260822.glance.tmp.audit.json"
"${python_bin}" "${root}/finalize_saved_submission_queue.py" \
  --strict "${combined_strict}" --backup "${combined_backup}" \
  --submitted "${submitted}" --output "${ready_tmp}" \
  --audit-output "${audit_tmp}" --expected-count 100 --max-count 3 \
  > "${output}/refresh_ready_queue.log"
mv "${ready_tmp}" "${project}/submissions/READY_20260822.tsv"
mv "${audit_tmp}" "${project}/submissions/READY_20260822.audit.json"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" \
  --candidates "${project}/submissions/READY_20260822.tsv" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${output}/calibrated_rank.log"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
echo "saved component-background Glance candidate: ${submission}"

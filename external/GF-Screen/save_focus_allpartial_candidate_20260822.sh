#!/usr/bin/env bash
set -euo pipefail

# Save, never upload, the strongest qualified all-partial Focus state.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_focus_allpartial_deployed_20260822"
output="${root}/public_sweeps/gf_focus_allpartial_saved_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"

mkdir -p "${output}"
exec 9>"${root}/runs/save_focus_allpartial_candidate_20260822.lock"
flock -n 9 || { echo "all-partial Focus saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" ]]; do sleep 30; done
[[ -s "${audit}/summary.tsv" ]]

queue_floor=-1
if [[ -s "${ready}" && $(awk 'END {print NR-1}' "${ready}") -ge 3 ]]; then
  queue_floor=$(awk -F '\t' 'NR==2{f=($5+$6)/2} NR>2{s=($5+$6)/2;if(s<f)f=s} END{printf "%.9f",f}' "${ready}")
fi
selected=""
while IFS=$'\t' read -r arm checkpoint_tag checkpoint lesion dsc nsd combined metrics robust; do
  [[ "${arm}" == variant ]] && continue
  read -r decision tail_guard < <("${python_bin}" -c 'import json,sys;x=json.load(open(sys.argv[1]));print(x["decision"],str(x.get("checks",{}).get("tail_regression_guard",False)).lower())' "${robust}")
  eligible=$(awk -v c="${combined}" -v f="${queue_floor}" -v g="${decision}" -v t="${tail_guard}" \
    'BEGIN{print (g=="PROMOTE" || (t=="true" && c>f+0.000001)) ? "yes" : "no"}')
  if [[ "${eligible}" == yes ]]; then
    selected=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${arm}" "${checkpoint_tag}" "${checkpoint}" "${lesion}" "${dsc}" "${nsd}" \
      "${combined}" "${metrics}" "${robust}" "${decision}")
    break
  fi
done < <({ head -1 "${audit}/summary.tsv"; tail -n +2 "${audit}/summary.tsv" | sort -t $'\t' -k7,7gr -k5,5gr; })

if [[ -z "${selected}" ]]; then
  date -Ins > "${output}/NO_ELIGIBLE_FOCUS_CANDIDATE.txt"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
IFS=$'\t' read -r arm checkpoint_tag checkpoint lesion dsc nsd combined metrics robust decision <<< "${selected}"
[[ "${arm}" == *lr1e5* ]] && family=fap1 || family=fap3
epoch=${checkpoint_tag#checkpoint_epoch_}; epoch=${epoch%%_*}; epoch=$((10#${epoch}))
lesion_tag=${lesion/./p}
tag="gf_${family}_e${epoch}_l${lesion_tag}"
gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done

ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
  ROBUST_JSON="${robust}" GLANCE_THRESHOLD=0.50 LESION_THRESHOLD="${lesion}" \
  bash "${materializer}"
submission="${project}/submissions/${tag}.zip"
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
saved="${output}/saved_focus_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
printf '1\t%s\t%s\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoint}" "${lesion}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_component_bg_saved_20260822/saved_glance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_cbg_intersection_saved_20260822/saved_intersection_submission.tsv" \
  --manifest "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/saved_wise_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_boxloc_saved_20260822/saved_boxloc_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_cbg_box_saved_20260822/saved_cbg_box_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_cbg_center_saved_20260822/saved_cbg_center_submission.tsv" \
  --manifest "${root}/public_sweeps/candidate_refiner_v1_saved_20260822/saved_candidate_refiner_submission.tsv" \
  --manifest "${saved}" --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${output}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${output}/calibrated_rank.log"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"

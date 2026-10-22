#!/usr/bin/env bash
set -euo pipefail

# Preserve the public-qualified voxel intersection between the immutable hidden
# champion and the complementary component-background Glance state. Never upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream="${root}/public_sweeps/gf_glance_component_bg_saved_20260822/SAVE_ONLY_COMPLETE.txt"
public="${root}/public_sweeps/champion_cbg41_e283_g020_fusion_20260822"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
base_tag="gf_gcb41_e283_g0p20"
tag="gf_int_cbg41_e283_g020"
candidate_state="${root}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
output="${root}/hidden_predictions/cbg_intersection_20260822"
saved_root="${root}/public_sweeps/gf_cbg_intersection_saved_20260822"
submission="${project}/submissions/${tag}.zip"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"

mkdir -p "${output}"/{champion,candidate,intersection} "${saved_root}"
exec 9>"${root}/runs/save_cbg_intersection_candidate_20260822.lock"
flock -n 9 || { echo "CBG intersection saver already active" >&2; exit 1; }
[[ -s "${saved_root}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${upstream}" ]]; do sleep 30; done
for required in "${checkpoint}" "${champion_zip}" \
    "${public}/intersection.metrics.csv" "${public}/intersection.robust.json" \
    "${candidate_state}/g0p20.metrics.csv" "${candidate_state}/g0p20.robust.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

read -r decision tail_guard ci_positive < <("${python_bin}" -c \
  'import json,sys;x=json.load(open(sys.argv[1]));c=x.get("checks",{});print(x["decision"],str(c.get("tail_regression_guard",False)).lower(),str(c.get("combined_ci_positive",False)).lower())' \
  "${public}/intersection.robust.json")
if [[ "${tail_guard}" != true || "${ci_positive}" != true ]]; then
  date -Ins > "${saved_root}/NO_ELIGIBLE_INTERSECTION.txt"
  date -Ins > "${saved_root}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done

candidate_deployed="${root}/hidden_predictions/deployed_candidates_20260821/${base_tag}/deployed"
if [[ ! -d "${candidate_deployed}" || $(find "${candidate_deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${base_tag}" \
    METRICS="${candidate_state}/g0p20.metrics.csv" \
    ROBUST_JSON="${candidate_state}/g0p20.robust.json" \
    GLANCE_THRESHOLD=0.20 LESION_THRESHOLD=0.50 bash "${materializer}"
fi
[[ $(find "${candidate_deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
unzip -qo "${champion_zip}" -d "${output}/champion"
cp "${candidate_deployed}"/*.nii.gz "${output}/candidate/"
[[ $(find "${output}/champion" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${output}/candidate" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${output}/champion" --pred-b "${output}/candidate" \
  --fusion intersection --min-mm3 0 --workers 4 \
  --output "${output}/intersection" > "${output}/intersection.log" 2>&1
[[ $(find "${output}/intersection" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${project}/validation/Validation-Hidden-Images" \
  --pred-dir "${output}/intersection" --output "${submission}" \
  --expected-count 100 > "${output}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
dsc=$(awk '/DSC  mean/{print $3}' "${public}/intersection.score.log")
nsd=$(awk '/NSD  mean/{print $3}' "${public}/intersection.score.log")
saved="${saved_root}/saved_intersection_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
printf '1\t%s\tcbgint\t%s\t0.20\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoint}" "${dsc}" "${nsd}" \
  "${public}/intersection.metrics.csv" "${public}/intersection.robust.json" \
  "${decision}" "${submission}" "${digest}" >> "${saved}"

"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_component_bg_saved_20260822/saved_glance_submission.tsv" \
  --manifest "${saved}" \
  --manifest "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/saved_wise_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_boxloc_saved_20260822/saved_boxloc_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_focus_allpartial_saved_20260822/saved_focus_submission.tsv" \
  --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${saved_root}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${saved_root}/calibrated_rank.log"
date -Ins > "${saved_root}/SAVE_ONLY_COMPLETE.txt"

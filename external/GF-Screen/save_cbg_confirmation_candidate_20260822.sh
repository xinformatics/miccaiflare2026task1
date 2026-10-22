#!/usr/bin/env bash
set -euo pipefail

# Preserve (never upload) the component-level confirmation complement to the
# voxel-intersection candidate.  This keeps an entire champion component when
# it has any overlap with the component-background Glance support.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream="${root}/public_sweeps/gf_cbg_intersection_saved_20260822/SAVE_ONLY_COMPLETE.txt"
public="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822"
hidden="${root}/hidden_predictions/cbg_intersection_20260822"
config="${public}/configs/cbg_confirm_distance_mm_0_rescue_min_mm3_inf.json"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
saved_root="${root}/public_sweeps/gf_cbg_intersection_saved_20260822"
saved="${saved_root}/saved_intersection_submission.tsv"
tag="gf_conf_cbg41_e283"
submission="${project}/submissions/${tag}.zip"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
ready="${project}/submissions/READY_20260822.tsv"

mkdir -p "${hidden}/confirm_exact" "${saved_root}"
exec 9>"${root}/runs/save_cbg_confirmation_candidate_20260822.lock"
flock -n 9 || { echo "CBG confirmation saver already active" >&2; exit 1; }
[[ -s "${saved_root}/CONFIRM_SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${upstream}" ]]; do sleep 30; done

for required in "${config}" "${public}/exact0.metrics.csv" \
    "${public}/exact0.robust.json" "${hidden}/champion" "${hidden}/candidate" \
    "${saved}"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${hidden}/champion" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${hidden}/candidate" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${project}/external/BiomedParse/biomedparse_fusion.py" materialize \
  --base "${hidden}/champion" --support "${hidden}/candidate" \
  --config "${config}" --output "${hidden}/confirm_exact" \
  > "${hidden}/confirm_exact.log" 2>&1
[[ $(find "${hidden}/confirm_exact" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${project}/validation/Validation-Hidden-Images" \
  --pred-dir "${hidden}/confirm_exact" --output "${submission}" \
  --expected-count 100 > "${hidden}/confirm_exact.package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
read -r dsc nsd < <("${python_bin}" -c \
  'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(sum(float(x["Lesion_DSC"]) for x in r)/len(r), sum(float(x["Lesion_NSD"]) for x in r)/len(r))' \
  "${public}/exact0.metrics.csv")
decision=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "${public}/exact0.robust.json")

manifest_tmp="${saved}.confirm.tmp"
awk -F '\t' -v tag="${tag}" 'NR == 1 || $2 != tag' "${saved}" > "${manifest_tmp}"
printf '2\t%s\tcbgconf\t%s\t0.20\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoint}" "${dsc}" "${nsd}" \
  "${public}/exact0.metrics.csv" "${public}/exact0.robust.json" \
  "${decision}" "${submission}" "${digest}" >> "${manifest_tmp}"
mv "${manifest_tmp}" "${saved}"

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
  > "${saved_root}/refresh_ready_after_confirm.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${saved_root}/calibrated_rank_after_confirm.log"
date -Ins > "${saved_root}/CONFIRM_SAVE_ONLY_COMPLETE.txt"
echo "saved component-confirmation candidate: ${submission}"

#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden inference for a publicly qualified standalone refiner.
# This script never uploads and never reads hidden labels.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/candidate_refiner_standalone_v1_20260822"
output="${root}/public_sweeps/candidate_refiner_standalone_v1_saved_20260822"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_proposals="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
hidden_root="${root}/hidden_predictions/candidate_refiner_standalone_v1_20260822"
hidden_base="${hidden_root}/champion_base"
disc_work="${hidden_root}/work_disconnected_iso15"
conn_work="${hidden_root}/work_connected_iso15"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
packager="${project}/nnunet/package_codabench.py"
folds=()
for fold in 0 1 2 3 4; do
  folds+=("${root}/runs/candidate_refiner_v1_fold${fold}_20260822/checkpoint_best.pt")
done

mkdir -p "${output}" "${hidden_root}" "${hidden_base}"
exec 9>"${root}/runs/save_candidate_refiner_standalone_candidate_20260822.lock"
flock -n 9 || { echo "standalone refiner saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for standalone refiner public evaluation\n' "$(date -Ins)"
  sleep 30
done
for required in "${audit}/summary.tsv" "${champion_zip}" "${folds[@]}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

queue_floor=-1
if [[ -s "${ready}" && $(awk 'END {print NR-1}' "${ready}") -ge 3 ]]; then
  queue_floor=$(awk -F '\t' 'NR==2{f=($5+$6)/2} NR>2{s=($5+$6)/2;if(s<f)f=s} END{printf "%.9f",f}' "${ready}")
fi
selected=$("${python_bin}" - "${audit}/summary.tsv" "${queue_floor}" <<'PY'
import csv,json,sys
path,floor=sys.argv[1],float(sys.argv[2])
with open(path,newline='',encoding='utf-8') as stream:
    rows=list(csv.DictReader(stream,delimiter='\t'))
eligible=[]
for row in rows:
    with open(row['robust'],encoding='utf-8') as stream:
        robust=json.load(stream)
    decision=robust['decision'];tail=bool(robust.get('checks',{}).get('tail_regression_guard',False))
    combined=float(row['combined'])
    if decision=='PROMOTE' or (tail and combined>floor+1e-6):
        row['decision']=decision;eligible.append(row)
if eligible:
    row=max(eligible,key=lambda x:(float(x['combined']),float(x['DSC'])))
    fields=('variant','classification_threshold','segmentation_threshold','gate','strategy','DSC','NSD','combined','metrics','robust','decision')
    print('\t'.join(str(row[field]) for field in fields))
PY
)
if [[ -z "${selected}" ]]; then
  date -Ins > "${output}/NO_ELIGIBLE_STANDALONE_REFINER.txt"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
IFS=$'\t' read -r variant classification segmentation gate strategy dsc nsd combined metrics robust decision <<< "${selected}"
case "${gate}" in
  strict) seed_coverage=0.05; refined_overlap=0.05; min_ratio=0.25; max_ratio=16.0 ;;
  expand) seed_coverage=0.05; refined_overlap=0.001; min_ratio=0.10; max_ratio=64.0 ;;
  *) echo "unknown gate: ${gate}" >&2; exit 2 ;;
esac

if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  unzip -nq "${champion_zip}" -d "${hidden_base}"
fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
prepare_hidden() {
  local work="$1" selection="$2" log="$3"
  [[ -s "${work}/PREPARE_COMPLETE.txt" ]] && return
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" --proposal "gf_l050=${hidden_proposals}" \
    --base "${hidden_base}" "${selection}" --work "${work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 20000 \
    > "${log}" 2>&1
}
prepare_hidden "${disc_work}" --only-disconnected "${output}/prepare_hidden_disc.log" & p0=$!
prepare_hidden "${conn_work}" --only-connected "${output}/prepare_hidden_conn.log" & p1=$!
status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]
for work in "${disc_work}" "${conn_work}"; do
  tag=$(basename "${work}")
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 4 \
    > "${output}/audit_${tag}.log" 2>&1
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done
checkpoint_args=()
for checkpoint in "${folds[@]}"; do checkpoint_args+=(--checkpoint "${checkpoint}"); done
inference_matches() {
  local work="$1"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${folds[@]}" <<'PY'
import json,sys
from pathlib import Path
marker=Path(sys.argv[1]);expected=[str(Path(p).resolve()) for p in sys.argv[2:]]
if not marker.is_file():raise SystemExit(1)
raise SystemExit(0 if json.loads(marker.read_text()).get('checkpoints',[])==expected else 1)
PY
}
infer_hidden() {
  local gpu="$1" work="$2" log="$3"
  inference_matches "${work}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 > "${log}" 2>&1
}
infer_hidden 0 "${disc_work}" "${output}/infer_hidden_disc.log" & p0=$!
infer_hidden 1 "${conn_work}" "${output}/infer_hidden_conn.log" & p1=$!
status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]

ctag=${classification/./p}; stag=${segmentation/./p}
if [[ "${strategy}" == approved ]]; then strategy_tag=ap; else strategy_tag=rf; fi
tag="gf_fprst_c${ctag}_s${stag}_${gate}_${strategy_tag}"
disc="${hidden_root}/${tag}_disc"; conn="${hidden_root}/${tag}_conn"
prediction="${hidden_root}/${tag}"
for item in "${disc_work}:${disc}:refined" "${conn_work}:${conn}:placeholder"; do
  work=${item%%:*}; partial=${item#*:}
  partial=${partial%%:*}
  if [[ "${work}" == "${conn_work}" && "${strategy}" == approved ]]; then mode=approved_replace; else mode=refined; fi
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${hidden_base}" --empty-base --output "${partial}" \
    --classification-threshold "${classification}" --segmentation-threshold "${segmentation}" \
    --mode "${mode}" --min-seed-coverage "${seed_coverage}" \
    --min-refined-overlap "${refined_overlap}" --min-volume-ratio "${min_ratio}" \
    --max-volume-ratio "${max_ratio}" > "${partial}.materialize.log" 2>&1
done
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${disc}" --pred-b "${conn}" --fusion union --min-mm3 0 \
  --output "${prediction}" --workers 4 > "${prediction}.union.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" \
  --pred-dir "${prediction}" --output "${submission}" --expected-count 100 \
  > "${output}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"
saved="${output}/saved_standalone_refiner_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
checkpoints=$(IFS=,; echo "${folds[*]}")
printf '1\t%s\tcandidate_refiner_standalone_v1\t%s\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoints}" "${segmentation}" "${dsc}" "${nsd}" "${metrics}" \
  "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

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

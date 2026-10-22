#!/usr/bin/env bash
set -euo pipefail

# Save, verify, and rank a publicly qualified candidate-refiner archive. This
# script never uploads to Codabench and never reads hidden labels.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/candidate_refiner_v1_20260822"
output="${root}/public_sweeps/candidate_refiner_v1_saved_20260822"
fold0="${root}/runs/candidate_refiner_v1_fold0_20260822/checkpoint_best.pt"
fold1="${root}/runs/candidate_refiner_v1_fold1_20260822/checkpoint_best.pt"
fold2="${root}/runs/candidate_refiner_v1_fold2_20260822/checkpoint_best.pt"
fold3="${root}/runs/candidate_refiner_v1_fold3_20260822/checkpoint_best.pt"
fold4="${root}/runs/candidate_refiner_v1_fold4_20260822/checkpoint_best.pt"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_proposals="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
hidden_root="${root}/hidden_predictions/candidate_refiner_v1_20260822"
hidden_base="${hidden_root}/champion_base"
hidden_work_disconnected="${hidden_root}/work_disconnected_iso15"
hidden_work_connected="${hidden_root}/work_connected_iso15"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
packager="${project}/nnunet/package_codabench.py"

mkdir -p "${output}" "${hidden_root}" "${hidden_base}"
exec 9>"${root}/runs/save_candidate_refiner_candidate_20260822.lock"
flock -n 9 || { echo "candidate-refiner saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
for required in "${audit}/PUBLIC_ONLY_COMPLETE.txt" "${audit}/summary.tsv" \
  "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}" \
  "${champion_zip}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

queue_floor=-1
if [[ -s "${ready}" && $(awk 'END {print NR-1}' "${ready}") -ge 3 ]]; then
  queue_floor=$(awk -F '\t' 'NR==2{f=($5+$6)/2} NR>2{s=($5+$6)/2;if(s<f)f=s} END{printf "%.9f",f}' "${ready}")
fi
selected=$("${python_bin}" - "${audit}/summary.tsv" "${queue_floor}" <<'PY'
import csv
import json
import sys

path, queue_floor = sys.argv[1], float(sys.argv[2])
with open(path, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
eligible = []
for row in rows:
    with open(row["robust"], encoding="utf-8") as stream:
        robust = json.load(stream)
    decision = robust["decision"]
    tail_guard = bool(robust.get("checks", {}).get("tail_regression_guard", False))
    combined = float(row["combined"])
    if decision == "PROMOTE" or (tail_guard and combined > queue_floor + 1e-6):
        row["decision"] = decision
        eligible.append(row)
if eligible:
    row = max(eligible, key=lambda item: (float(item["combined"]), float(item["DSC"])))
    fields = (
        "variant", "classification_threshold", "segmentation_threshold",
        "mode", "DSC", "NSD", "combined", "metrics", "robust", "decision",
    )
    print("\t".join(str(row[field]) for field in fields))
PY
)
if [[ -z "${selected}" ]]; then
  date -Ins > "${output}/NO_ELIGIBLE_CANDIDATE_REFINER.txt"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
IFS=$'\t' read -r variant classification segmentation mode dsc nsd combined metrics robust decision <<< "${selected}"
if [[ "${mode}" == replace ]]; then
  hidden_work="${hidden_work_connected}"
  selection_arg=--only-connected
else
  hidden_work="${hidden_work_disconnected}"
  selection_arg=--only-disconnected
fi

if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  unzip -nq "${champion_zip}" -d "${hidden_base}"
fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" --proposal "gf_l050=${hidden_proposals}" \
    --base "${hidden_base}" "${selection_arg}" --work "${hidden_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 20000 \
    > "${output}/prepare_hidden.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT.json" --workers 4 \
  > "${output}/audit_hidden_cache.log" 2>&1
[[ -s "${hidden_work}/AUDIT.json" ]]

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done
inference_matches() {
  "${python_bin}" - "${hidden_work}/INFERENCE_COMPLETE.json" \
    "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}" <<'PY'
import json,sys
from pathlib import Path
marker=Path(sys.argv[1])
expected=[str(Path(path).resolve()) for path in sys.argv[2:]]
if not marker.is_file(): raise SystemExit(1)
actual=json.loads(marker.read_text()).get("checkpoints",[])
raise SystemExit(0 if actual==expected else 1)
PY
}
if ! inference_matches; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" --checkpoint "${fold0}" --checkpoint "${fold1}" \
    --checkpoint "${fold2}" --checkpoint "${fold3}" --checkpoint "${fold4}" \
    --batch-size 8 --workers 2 > "${output}/infer_hidden.log" 2>&1
fi

classification_tag=${classification/./p}
segmentation_tag=${segmentation/./p}
case "${mode}" in
  seed) mode_tag=sd ;;
  refined) mode_tag=rf ;;
  seed_union) mode_tag=su ;;
  replace) mode_tag=rp ;;
esac
tag="gf_fpr1_c${classification_tag}_s${segmentation_tag}_${mode_tag}"
prediction="${hidden_root}/${tag}"
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
  --work "${hidden_work}" --base "${hidden_base}" --output "${prediction}" \
  --classification-threshold "${classification}" \
  --segmentation-threshold "${segmentation}" --mode "${mode}" \
  --min-seed-coverage 0.05 --min-refined-overlap 0.05 \
  --min-volume-ratio 0.25 --max-volume-ratio 16.0 --min-base-dice 0.25 \
  --min-base-seed-coverage 0.25 \
  > "${output}/materialize_hidden.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" \
  --pred-dir "${prediction}" --output "${submission}" --expected-count 100 \
  > "${output}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"
saved="${output}/saved_candidate_refiner_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
printf '1\t%s\tcandidate_refiner_v1\t%s,%s,%s,%s,%s\t\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}" \
  "${segmentation}" "${dsc}" "${nsd}" \
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
  --manifest "${saved}" --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${output}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${output}/calibrated_rank.log"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"

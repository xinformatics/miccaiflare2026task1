#!/usr/bin/env bash
set -euo pipefail

# Public-only diagnostic for the unlabeled pseudo-label teacher. Compare the
# current high-recall Glance gate (0.20) with exhaustive Focus crops (0.0) at
# the high Focus thresholds used for pseudo labels. No hidden inputs, ZIP, or
# upload path exists in this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/exhaustive_teacher_discovery_20260822"
thresholds="0.85,0.90,0.95,0.97"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "exhaustive teacher audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
[[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

infer_mode() {
  local gpu="$1" mode="$2" glance="$3" mode_output="${output}/${mode}/inference"
  mkdir -p "${mode_output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${mode_output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${glance}" --lesion_thresholds "${thresholds}" \
    --focus_batch_size 2 --workers 2 --skip_existing \
    > "${output}/${mode}/inference.log" 2>&1
}
infer_mode 0 exhaustive 0.0 & pid0=$!
infer_mode 1 gated 0.20 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

printf 'mode\tglance_threshold\tlesion_threshold\tDSC\tNSD\tdetection_recall\tsmall_recall_100to1000\tcomponent_precision\tgt_detected\tsmall_detected_100to1000\tpred_components\tpredictions\tcomponent_audit\n' \
  > "${output}/summary.tsv"
for mode in exhaustive gated; do
  [[ "${mode}" == exhaustive ]] && glance=0.0 || glance=0.20
  for threshold in 0.85 0.90 0.95 0.97; do
    threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
    raw="${output}/${mode}/inference/${threshold_dir}"
    filtered="${output}/${mode}/${threshold_dir}_mm100"
    audit="${output}/${mode}/${threshold_dir}_mm100_component_audit"
    metrics="${output}/${mode}/${threshold_dir}_mm100.metrics.csv"
    score="${output}/${mode}/${threshold_dir}_mm100.score.log"
    [[ $(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
    mkdir -p "${filtered}"
    "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
      --pred-a "${raw}" --fusion a --min-mm3 100 --workers 4 \
      --output "${filtered}" > "${filtered}.log" 2>&1
    "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
      --pred "${filtered}" --gt "${labels}" --csv "${metrics}" --workers 4 \
      > "${score}" 2>&1
    "${python_bin}" -u "${root}/analyze_lesion_errors.py" \
      --labels "${labels}" --predictions "${filtered}" --output "${audit}" \
      > "${audit}.log" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score}")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score}")
    read -r detection small_recall precision detected small_detected components < <(
      "${python_bin}" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); c=x["components"]; b=x["gt_by_size_mm3"]; sd=b["100to500"]["detected"]+b["500to1000"]["detected"]; st=b["100to500"]["components"]+b["500to1000"]["components"]; print(c["detection_recall"],sd/st if st else 0,c["precision"],c["gt_detected"],sd,c["pred_total"])' \
        "${audit}/summary.json"
    )
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${mode}" "${glance}" "${threshold}" "${dsc}" "${nsd}" \
      "${detection}" "${small_recall}" "${precision}" "${detected}" \
      "${small_detected}" "${components}" \
      "${filtered}" "${audit}/summary.json" >> "${output}/summary.tsv"
  done
done
"${python_bin}" "${root}/select_exhaustive_teacher_decision.py" \
  --summary "${output}/summary.tsv" --output "${output}/decision.json" \
  > "${output}/decision.log"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary.tsv"

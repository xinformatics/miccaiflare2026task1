#!/usr/bin/env bash
set -euo pipefail

# Public-only WiSE audit between the alpha=0.55 bbox champion and the strongest
# alpha=0.69 bbox state (epoch 276). No hidden inference, ZIP, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
interpolations="${root}/checkpoint_interpolations/a055_bbox_to_a069_bbox_ep276_20260822"
output="${root}/public_sweeps/a055_a069_bbox_wise_20260822"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "bbox WiSE audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
for alpha in 0p250 0p500 0p750; do
  [[ -s "${interpolations}/wise_focus_alpha${alpha}.pt" ]] || {
    echo "missing interpolation alpha ${alpha}" >&2; exit 2;
  }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done
[[ -s "${prompt_config}" && -s "${anchor_metrics}" ]]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

evaluate_one() {
  local gpu="$1" alpha="$2"
  local checkpoint="${interpolations}/wise_focus_alpha${alpha}.pt"
  local state="${output}/alpha_${alpha}"
  local raw="${state}/raw_l0p50"
  local filtered="${state}/filtered_l0p50_mm1000"
  local prompted="${state}/prompted_l0p50_mm1000"
  local deployed="${state}/deployed_l0p50_mm1000_new2k_s25"
  mkdir -p "${raw}" "${filtered}" "${prompted}" "${deployed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${raw}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_batch_size 16 --workers 2 --skip_existing \
    > "${state}/inference.log" 2>&1
  [[ $(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
    --pred-a "${raw}" --fusion a --min-mm3 1000 --workers 4 \
    --output "${filtered}" > "${state}/filter.log" 2>&1
  "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
    --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
    --output "${prompted}" > "${state}/prompted.log" 2>&1
  "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
    --base "${prompted}" --candidate "${guided_candidate}" \
    --support "${prompt_support}" --output "${deployed}" \
    --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
    --min-support-fraction 0.25 --min-support-mm3 100 \
    --support-dilation-mm inf --workers 4 > "${state}/deployed.log" 2>&1
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${deployed}" --gt "${labels}" --csv "${state}/metrics.csv" \
    --workers 4 > "${state}/score.log" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${state}/metrics.csv" \
    --anchor-name champion_0p7006 --candidate-name "bbox_wise_${alpha}" \
    --output "${state}/robust.json" > "${state}/robust.log"
}

evaluate_one 0 0p250 & pid0=$!
evaluate_one 1 0p500 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
evaluate_one 0 0p750

printf 'alpha\tcheckpoint\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"
for alpha in 0p250 0p500 0p750; do
  state="${output}/alpha_${alpha}"
  dsc=$(awk '/DSC  mean/ {print $3}' "${state}/score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${state}/score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${state}/robust.json")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${alpha}" "${interpolations}/wise_focus_alpha${alpha}.pt" \
    "${dsc}" "${nsd}" "${combined}" "${state}/metrics.csv" \
    "${state}/robust.json" "${decision}" >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k3,3gr -k4,4gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

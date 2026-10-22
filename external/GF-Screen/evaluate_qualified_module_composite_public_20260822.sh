#!/usr/bin/env bash
set -euo pipefail

# Public-only evaluation of independently save-qualified Focus and Glance
# modules. The exact champion deployment transform is held fixed.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
composition="${root}/checkpoint_compositions/qualified_focus_glance_20260822"
selection="${composition}/selection.json"
checkpoint="${composition}/model.pt"
output="${root}/public_sweeps/qualified_focus_glance_deployed_20260822"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "qualified composite evaluation already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
for required in "${composition}/COMPOSITION_COMPLETE.txt" "${selection}" "${checkpoint}" \
    "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done

read -r focus_status glance_status glance lesion < <("${python_bin}" -c \
  'import json,sys;x=json.load(open(sys.argv[1]));print(x["focus"]["status"],x["glance"]["status"],x["glance"]["glance_threshold"],x["focus"]["lesion_threshold"])' \
  "${selection}")
if [[ "${focus_status}" != qualified_saved_source || \
      "${glance_status}" != qualified_saved_source ]]; then
  printf 'focus=%s\nglance=%s\n' "${focus_status}" "${glance_status}" \
    > "${output}/NO_DUAL_QUALIFIED_MODULES.txt"
  date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
  exit 0
fi

glance_tag="${glance/./p}"
lesion_tag="${lesion/./p}"
state="${output}/g${glance_tag}_l${lesion_tag}"
inference="${state}/inference"
filtered="${state}/filtered_mm1000"
prompted="${state}/prompted_mm1000"
deployed="${state}/deployed_mm1000_new2k_s25"
mkdir -p "${inference}" "${filtered}" "${prompted}" "${deployed}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
  --test_data_path "${images}" --save_prediction_path "${inference}" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold "${glance}" --lesion_threshold "${lesion}" \
  --focus_batch_size 16 --workers 2 --skip_existing > "${state}/inference.log" 2>&1
[[ $(find "${inference}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${inference}" --fusion a --min-mm3 1000 --workers 4 \
  --output "${filtered}" > "${filtered}.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
  --output "${prompted}" > "${prompted}.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${prompted}" --candidate "${guided_candidate}" --support "${prompt_support}" \
  --output "${deployed}" --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 --support-dilation-mm inf \
  --workers 4 > "${deployed}.log" 2>&1

metrics="${state}/metrics.csv"
score_log="${state}/score.log"
robust="${state}/robust.json"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
  > "${score_log}" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name champion_0p7006 --candidate-name qualified_focus_glance \
  --output "${robust}" > "${state}/robust.log"
dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
printf 'variant\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\n' \
  > "${output}/summary.tsv"
printf 'qualified_focus_glance\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${checkpoint}" "${glance}" "${lesion}" "${dsc}" "${nsd}" "${combined}" \
  "${metrics}" "${robust}" >> "${output}/summary.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary.tsv"

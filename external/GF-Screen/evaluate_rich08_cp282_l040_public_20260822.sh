#!/usr/bin/env bash
set -euo pipefail

# Single prespecified follow-up to the completed 4x/8x grid. This is public
# only and cannot access hidden images or create a submission archive.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_rich08_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0282.pt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
rich16_metrics="${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/l0p40.metrics.csv"
output="${root}/public_sweeps/gf_rich08_cp282_l040_deployed_20260822"

for required in "${checkpoint}" "${prompt_config}" "${anchor_metrics}" "${rich16_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done
mkdir -p "${output}"/{shard0,shard1,raw,filtered,prompted,deployed}
exec 9>"${output}.lock"
flock -n 9 || { echo "rich08 checkpoint-282 threshold-0.40 audit active" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${output}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.40 \
    --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${output}/shard${shard}.log" 2>&1
}
infer_shard 0 0 & p0=$!
infer_shard 1 1 & p1=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]]
cp "${output}/shard0"/*.nii.gz "${output}/raw/"
cp "${output}/shard1"/*.nii.gz "${output}/raw/"
[[ $(find "${output}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${output}/raw" --fusion a --min-mm3 1000 --workers 8 \
  --output "${output}/filtered" > "${output}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${output}/filtered" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${output}/prompted" \
  > "${output}/prompted.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${output}/prompted" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${output}/deployed" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 8 > "${output}/deployed.log" 2>&1
[[ $(find "${output}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

metrics="${output}/metrics.csv"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${output}/deployed" --gt "${labels}" --csv "${metrics}" --workers 8 \
  > "${output}/score.log" 2>&1
"${python_bin}" "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name champion_0p7006 --candidate-name rich08_cp282_l040 \
  --output "${output}/vs_champion.robust.json" > "${output}/vs_champion.robust.log"
"${python_bin}" "${root}/select_checkpoint_robust.py" \
  --anchor "${rich16_metrics}" --candidate "${metrics}" \
  --anchor-name rich16_cp281_l040 --candidate-name rich08_cp282_l040 \
  --output "${output}/vs_rich16_l040.robust.json" > "${output}/vs_rich16_l040.robust.log"
dsc=$(awk '/DSC  mean/ {print $3}' "${output}/score.log")
nsd=$(awk '/NSD  mean/ {print $3}' "${output}/score.log")
printf 'model\tcheckpoint\tfocus_threshold\tDSC\tNSD\tmetrics\n' > "${output}/summary.tsv"
printf 'gf_rich08\tcheckpoint_epoch_0282\t0.40\t%s\t%s\t%s\n' \
  "${dsc}" "${nsd}" "${metrics}" >> "${output}/summary.tsv"
cat "${output}/summary.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"

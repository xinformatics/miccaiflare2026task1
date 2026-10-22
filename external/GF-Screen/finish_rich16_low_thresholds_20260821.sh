#!/usr/bin/env bash
set -euo pipefail

# Recover only shard 0 of the public low-threshold sweep after it collided with
# 4x training. Shard 1 is already complete. This script is public-only and
# cannot read hidden images or create submission archives.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281.pt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
leader_metrics="${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281/l0p50.metrics.csv"
output="${root}/public_sweeps/gf_rich16_comp281_low_thresholds_20260821"
thresholds="0.20 0.25 0.30 0.35 0.38"

for required in "${checkpoint}" "${prompt_config}" "${anchor_metrics}" "${leader_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 2; }
done
[[ $(find "${output}/shard1" -name '*.nii.gz' | wc -l) -eq 125 ]] || {
  echo "completed shard1 must contain 125 threshold outputs" >&2
  exit 2
}
mkdir -p "${output}"/{shard0,inference}
exec 9>"${output}.recovery.lock"
flock -n 9 || { echo "low-threshold recovery already active" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ $(find "${output}/shard0" -name '*.nii.gz' | wc -l) -ne 125 ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${output}/shard0" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.20,0.25,0.30,0.35,0.38 \
    --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id 0 \
    --skip_existing >> "${output}/shard0.log" 2>&1
fi
[[ $(find "${output}/shard0" -name '*.nii.gz' | wc -l) -eq 125 ]]

printf 'threshold\tDSC\tNSD\tcombined\tmetrics\trobust_anchor\trobust_leader\n' \
  > "${output}/summary.tsv"
for threshold in ${thresholds}; do
  threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  tag="${threshold/./p}"
  raw="${output}/inference/${threshold_dir}"
  filtered="${output}/l${tag}/filtered_mm1000"
  prompted="${output}/l${tag}/prompted_mm1000"
  deployed="${output}/l${tag}/deployed_mm1000_new2k_s25"
  mkdir -p "${raw}" "${filtered}" "${prompted}" "${deployed}"
  cp "${output}/shard0/${threshold_dir}"/*.nii.gz "${raw}/"
  cp "${output}/shard1/${threshold_dir}"/*.nii.gz "${raw}/"
  [[ $(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
    --pred-a "${raw}" --fusion a --min-mm3 1000 --workers 4 \
    --output "${filtered}" > "${output}/l${tag}/filter.log" 2>&1
  "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
    --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
    --output "${prompted}" > "${output}/l${tag}/prompted.log" 2>&1
  "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
    --base "${prompted}" --candidate "${guided_candidate}" --support "${prompt_support}" \
    --output "${deployed}" --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
    --min-support-fraction 0.25 --min-support-mm3 100 --support-dilation-mm inf \
    --workers 4 > "${output}/l${tag}/deployed.log" 2>&1
  [[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

  metrics="${output}/l${tag}.metrics.csv"
  score_log="${output}/l${tag}.score.log"
  robust_anchor="${output}/l${tag}.vs_anchor.robust.json"
  robust_leader="${output}/l${tag}.vs_leader.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name champion_0p7006 --candidate-name "rich16_comp281_l${tag}" \
    --output "${robust_anchor}" > "${robust_anchor%.json}.log"
  "${python_bin}" "${root}/select_checkpoint_robust.py" \
    --anchor "${leader_metrics}" --candidate "${metrics}" \
    --anchor-name rich16_comp281_l0p50 --candidate-name "rich16_comp281_l${tag}" \
    --output "${robust_leader}" > "${robust_leader%.json}.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${threshold}" "${dsc}" "${nsd}" "${combined}" "${metrics}" \
    "${robust_anchor}" "${robust_leader}" >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k2,2gr -k3,3gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"

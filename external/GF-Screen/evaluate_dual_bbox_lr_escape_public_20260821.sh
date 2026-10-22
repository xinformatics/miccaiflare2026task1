#!/usr/bin/env bash
set -euo pipefail

# Public-only deployed-space evaluation for the two LR-escape training arms.
# Every checkpoint receives exactly the champion inference, 1,000 mm3 filter,
# frozen prompt confirmation, and fixed disconnected-component addition. This
# script cannot infer hidden cases or create a submission ZIP.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${DEPLOYED_OUTPUT:-${root}/public_sweeps/gf_bbox_lr_escape_deployed_20260821}"
arm_a="${ARM_A:-gf_bbox_escape_lr1e4_ep280_5ep_20260821}"
arm_b="${ARM_B:-gf_bbox_escape_lr3e4_ep280_5ep_20260821}"
eval_epochs="${EVAL_EPOCHS:-281 282 283 284 285}"
focus_thresholds="${FOCUS_THRESHOLDS:-0.50 0.60}"
focus_threshold_csv=$(tr ' ' ',' <<< "${focus_thresholds}")

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "dual bbox LR deployed evaluation is already active" >&2
  exit 1
fi
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 1
  }
done
[[ -s "${prompt_config}" && -s "${anchor_metrics}" ]]

evaluate_arm() {
  local gpu="$1" arm="$2"
  local arm_output="${output}/${arm}"
  mkdir -p "${arm_output}"
  printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\n' \
    > "${arm_output}/summary.tsv"
  declare -A seen=()
  local candidate_checkpoints=() epoch checkpoint
  for epoch in ${eval_epochs}; do
    candidate_checkpoints+=("$(printf '%s/runs/%s/checkpoint_epoch_%04d.pt' "${root}" "${arm}" "${epoch}")")
  done
  candidate_checkpoints+=("${root}/runs/${arm}/model.pt")
  for checkpoint in "${candidate_checkpoints[@]}"; do
    [[ -s "${checkpoint}" ]] || continue
    local digest tag inference_root
    digest=$("${python_bin}" "${root}/checkpoint_state_digest.py" "${checkpoint}")
    [[ -n "${seen[${digest}]:-}" ]] && continue
    seen["${digest}"]=1
    tag=$(basename "${checkpoint}" .pt)
    inference_root="${arm_output}/${tag}/inference"
    mkdir -p "${inference_root}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${inference_root}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_thresholds "${focus_threshold_csv}" \
      --focus_batch_size 16 --workers 2 --skip_existing \
      > "${arm_output}/${tag}/inference.log" 2>&1

    for focus in ${focus_thresholds}; do
      local focus_dir focus_tag filtered prompted deployed score_log metrics robust
      focus_dir="${inference_root}/lesion_$(printf '%.2f' "${focus}")"
      focus_tag="${focus/./p}"
      [[ $(find "${focus_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
      filtered="${arm_output}/${tag}/filtered_l${focus_tag}_mm1000"
      prompted="${arm_output}/${tag}/prompted_l${focus_tag}_mm1000"
      deployed="${arm_output}/${tag}/deployed_l${focus_tag}_mm1000_new2k_s25"
      mkdir -p "${filtered}" "${prompted}" "${deployed}"
      "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
        --pred-a "${focus_dir}" --fusion a --min-mm3 1000 --workers 4 \
        --output "${filtered}" > "${filtered}.log" 2>&1
      "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
        --base "${filtered}" --support "${prompt_support}" \
        --config "${prompt_config}" --output "${prompted}" \
        > "${prompted}.log" 2>&1
      "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
        --base "${prompted}" --candidate "${guided_candidate}" \
        --support "${prompt_support}" --output "${deployed}" \
        --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
        --min-support-fraction 0.25 --min-support-mm3 100 \
        --support-dilation-mm inf --workers 4 > "${deployed}.log" 2>&1
      [[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
      metrics="${arm_output}/${tag}/l${focus_tag}.metrics.csv"
      score_log="${arm_output}/${tag}/l${focus_tag}.score.log"
      robust="${arm_output}/${tag}/l${focus_tag}.robust.json"
      "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
        --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
        > "${score_log}" 2>&1
      "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
        --anchor "${anchor_metrics}" --candidate "${metrics}" \
        --anchor-name champion_0p7006 --candidate-name "${arm}_${tag}_l${focus_tag}" \
        --output "${robust}" > "${robust%.json}.log"
      local dsc nsd combined
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${tag}" "${checkpoint}" "${focus}" "${dsc}" "${nsd}" \
        "${combined}" "${metrics}" "${robust}" >> "${arm_output}/summary.tsv"
    done
  done
}

evaluate_arm 0 "${arm_a}" & eval_a=$!
evaluate_arm 1 "${arm_b}" & eval_b=$!
status=0
wait "${eval_a}" || status=$?
wait "${eval_b}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

head -1 "${output}/${arm_a}/summary.tsv" > "${output}/summary.tsv"
tail -n +2 "${output}/${arm_a}/summary.tsv" >> "${output}/summary.tsv"
tail -n +2 "${output}/${arm_b}/summary.tsv" >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"

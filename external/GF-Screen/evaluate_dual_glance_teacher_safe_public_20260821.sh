#!/usr/bin/env bash
set -euo pipefail

# Exact deployed-space public evaluation for the two Glance-only arms.  Only
# each arm's internal best and final state are considered, with checkpoint
# digest deduplication.  The champion Focus, filters, prompts, and guided rescue
# remain unchanged.  This script cannot infer hidden cases or create a ZIP.
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
output="${DEPLOYED_OUTPUT:-${root}/public_sweeps/gf_glance_teacher_safe_deployed_20260821}"
arm_a="${ARM_A:-gf_glance_safe_posonly_ep280_5ep_20260821}"
arm_b="${ARM_B:-gf_glance_safe_posneg_ep280_5ep_20260821}"
eval_epochs="${EVAL_EPOCHS:-281}"
glance_thresholds="${GLANCE_THRESHOLDS:-0.35 0.50}"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "teacher-safe Glance deployed evaluation is already active" >&2
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
  printf 'variant\tcheckpoint_tag\tcheckpoint\tglance_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\n' \
    > "${arm_output}/summary.tsv"
  declare -A seen=()
  local candidate_checkpoints=() epoch checkpoint
  for epoch in ${eval_epochs}; do
    candidate_checkpoints+=("$(printf '%s/runs/%s/checkpoint_epoch_%04d.pt' "${root}" "${arm}" "${epoch}")")
  done
  candidate_checkpoints+=("${root}/runs/${arm}/model.pt" "${root}/runs/${arm}/model_final.pt")
  for checkpoint in "${candidate_checkpoints[@]}"; do
    [[ -s "${checkpoint}" ]] || continue
    local digest tag
    digest=$("${python_bin}" "${root}/checkpoint_state_digest.py" "${checkpoint}")
    [[ -n "${seen[${digest}]:-}" ]] && continue
    seen["${digest}"]=1
    tag=$(basename "${checkpoint}" .pt)
    for glance in ${glance_thresholds}; do
      local glance_tag inference filtered prompted deployed metrics score_log robust
      glance_tag="${glance/./p}"
      inference="${arm_output}/${tag}/g${glance_tag}/inference"
      mkdir -p "${inference}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${inference}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold "${glance}" --lesion_threshold 0.50 \
        --focus_batch_size 16 --workers 2 --skip_existing \
        > "${arm_output}/${tag}/g${glance_tag}/inference.log" 2>&1
      [[ $(find "${inference}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
      filtered="${arm_output}/${tag}/g${glance_tag}/filtered_mm1000"
      prompted="${arm_output}/${tag}/g${glance_tag}/prompted_mm1000"
      deployed="${arm_output}/${tag}/g${glance_tag}/deployed_mm1000_new2k_s25"
      mkdir -p "${filtered}" "${prompted}" "${deployed}"
      "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
        --pred-a "${inference}" --fusion a --min-mm3 1000 --workers 4 \
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
      metrics="${arm_output}/${tag}/g${glance_tag}.metrics.csv"
      score_log="${arm_output}/${tag}/g${glance_tag}.score.log"
      robust="${arm_output}/${tag}/g${glance_tag}.robust.json"
      "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
        --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
        > "${score_log}" 2>&1
      "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
        --anchor "${anchor_metrics}" --candidate "${metrics}" \
        --anchor-name champion_0p7006 \
        --candidate-name "${arm}_${tag}_g${glance_tag}" \
        --output "${robust}" > "${robust%.json}.log"
      local dsc nsd combined
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${tag}" "${checkpoint}" "${glance}" "${dsc}" "${nsd}" \
        "${combined}" "${metrics}" "${robust}" >> "${arm_output}/summary.tsv"
    done
  done
}

evaluate_arm 0 "${arm_a}" & a=$!
evaluate_arm 1 "${arm_b}" & b=$!
status=0
wait "${a}" || status=$?
wait "${b}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
head -1 "${output}/${arm_a}/summary.tsv" > "${output}/summary.tsv"
tail -n +2 "${output}/${arm_a}/summary.tsv" >> "${output}/summary.tsv"
tail -n +2 "${output}/${arm_b}/summary.tsv" >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
printf '%s\n' 'PUBLIC-ONLY EVALUATION COMPLETE: no hidden masks or ZIP generated.' \
  > "${output}/PUBLIC_ONLY_COMPLETE.txt"

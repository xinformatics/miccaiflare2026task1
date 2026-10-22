#!/usr/bin/env bash
set -euo pipefail

# Exact public-only evaluation for Glance models carrying the auxiliary box
# head. The head is instantiated for checkpoint compatibility but deployment
# still uses only the original binary crop gate. No hidden inference or ZIP.
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
arm_a="${ARM_A:-gf_glance_boxloc_w005_allpartial_ep280_3ep_20260822}"
arm_b="${ARM_B:-gf_glance_boxloc_w015_allpartial_ep280_3ep_20260822}"
output="${DEPLOYED_OUTPUT:-${root}/public_sweeps/gf_glance_boxloc_deployed_20260822}"
eval_epochs="${EVAL_EPOCHS:-281 283 285}"
# The default queue screens the final checkpoint first. All earlier checkpoint
# files remain immutable and can be expanded explicitly after a positive final
# result by setting EVALUATE_ALL_EPOCHS=1.
if [[ "${EVALUATE_ALL_EPOCHS:-0}" != 1 ]]; then
  eval_epochs=$(printf '%s\n' ${eval_epochs} | sort -n | tail -1)
fi
glance_thresholds="${GLANCE_THRESHOLDS:-0.20 0.35 0.50}"
glance_box_localization="${GLANCE_BOX_LOCALIZATION:-1}"
glance_center_localization="${GLANCE_CENTER_LOCALIZATION:-0}"
model_args=()
[[ "${glance_box_localization}" == 1 ]] && model_args+=(--glance_box_localization)
[[ "${glance_center_localization}" == 1 ]] && model_args+=(--glance_center_localization)

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "Glance box-localization evaluation already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done
[[ -s "${prompt_config}" && -s "${anchor_metrics}" ]]

evaluate_arm() {
  local gpu="$1" arm="$2" arm_output="${output}/$2"
  mkdir -p "${arm_output}"
  printf 'variant\tcheckpoint_tag\tcheckpoint\tglance_threshold\tDSC\tNSD\tcombined\tmetrics\trobust\n' \
    > "${arm_output}/summary.tsv"
  local epoch checkpoint digest tag glance glance_tag inference filtered prompted deployed metrics score_log robust
  for epoch in ${eval_epochs}; do
    checkpoint=$(printf '%s/runs/%s/checkpoint_epoch_%04d.pt' "${root}" "${arm}" "${epoch}")
    [[ -s "${checkpoint}" ]] || continue
    digest=$("${python_bin}" "${root}/checkpoint_state_digest.py" "${checkpoint}")
    tag="checkpoint_epoch_$(printf '%04d' "${epoch}")_${digest:0:8}"
    for glance in ${glance_thresholds}; do
      glance_tag="${glance/./p}"
      inference="${arm_output}/${tag}/g${glance_tag}/inference"
      mkdir -p "${inference}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${inference}" \
        --trained_pth "${checkpoint}" "${model_args[@]}" \
        --a_min -175 --a_max 250 --glance_threshold "${glance}" \
        --lesion_threshold 0.50 --focus_batch_size 16 --workers 2 --skip_existing \
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
        --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
        --output "${prompted}" > "${prompted}.log" 2>&1
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
        --anchor-name champion_0p7006 --candidate-name "${arm}_${tag}_g${glance_tag}" \
        --output "${robust}" > "${robust%.json}.log"
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${tag}" "${checkpoint}" "${glance}" "${dsc}" "${nsd}" \
        "${combined}" "${metrics}" "${robust}" >> "${arm_output}/summary.tsv"
    done
  done
}

evaluate_arm 0 "${arm_a}" & pid_a=$!
evaluate_arm 1 "${arm_b}" & pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
head -1 "${output}/${arm_a}/summary.tsv" > "${output}/summary.tsv"
tail -n +2 "${output}/${arm_a}/summary.tsv" >> "${output}/summary.tsv"
tail -n +2 "${output}/${arm_b}/summary.tsv" >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

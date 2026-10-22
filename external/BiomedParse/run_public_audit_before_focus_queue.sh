#!/usr/bin/env bash
set -euo pipefail

# Compare two automatic text prompts, then use each mask only as an additional
# crop proposal for the immutable GF champion. Labels are read only by the
# scorer after all predictions exist. The public paired and independent
# healthy-case gates determine promotion status; the strongest candidate is
# also packaged as an explicitly labeled fallback for the three-slot shortlist.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
hidden_images="${project}/validation/Validation-Hidden-Images"
healthy_images="${project}/validation/HealthyImages-noLesion"
healthy_subset="${root}/healthy_proxy_v1/images"
inputs="${root}/public_a160_240_inputs"
output="${root}/public_sweeps/automatic_text_proposals_20260820"
checkpoint="${gf}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
anchor="${gf}/public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
anchor_predictions="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_mm1000"
hidden_anchor_predictions="${gf}/predictions/gf_a055_bbox_pu_best_ep280/hidden_mm1000"
prompt_config="${root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
combined_support_root="${root}/public_sweeps/combined_prompt_confirmation_20260820/supports"
public_prompt_support="${combined_support_root}/union_e050"
hidden_prompt_support="${root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
filtered_proposal_minima=(100 500)

if [[ -s "${output}/AUDIT_COMPLETE.json" ]]; then
  echo "BiomedParse automatic-proposal audit already complete: ${output}"
  cat "${output}/AUDIT_COMPLETE.json"
  exit 0
fi

[[ $(find "${inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]] || {
  echo "BiomedParse public inputs are incomplete" >&2
  exit 1
}
[[ -f "${checkpoint}" && -f "${anchor}" ]] || {
  echo "GF champion or anchor metrics are missing" >&2
  exit 1
}
for required in "${prompt_config}" "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing prompt-confirmation asset: ${required}" >&2; exit 1; }
done
[[ $(find "${public_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
for proposal_minimum in "${filtered_proposal_minima[@]}"; do
  filtered_support="${combined_support_root}/union_e050_mm${proposal_minimum}"
  mkdir -p "${filtered_support}"
  if [[ $(find "${filtered_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
    "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
      --pred-a "${public_prompt_support}" --fusion a \
      --min-mm3 "${proposal_minimum}" --workers 4 --output "${filtered_support}" \
      > "${filtered_support}.filter.log" 2>&1
  fi
  [[ $(find "${filtered_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 prompt-OR masks filtered at ${proposal_minimum} mm3" >&2
    exit 1
  }
done
[[ $(find "${hidden_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}
[[ $(find "${anchor_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "GF public anchor predictions are incomplete" >&2
  exit 1
}
[[ $(find "${hidden_anchor_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "GF hidden anchor predictions are incomplete" >&2
  exit 1
}
[[ $(find "${healthy_images}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 172 ]] || {
  echo "Expected exactly 172 held-out healthy scans" >&2
  exit 1
}
"${python_bin}" "${root}/select_healthy_proxy.py" \
  --images "${healthy_images}" --output-dir "${healthy_subset}" \
  --manifest "${root}/healthy_proxy_v1/manifest.json" --total 32 \
  > "${root}/healthy_proxy_v1.select.log" 2>&1
[[ $(find "${healthy_subset}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 32 ]] || {
  echo "expected 32 predeclared healthy proxy images" >&2
  exit 1
}
mkdir -p "${output}/scored"
existence_thresholds=(0.30 0.50 0.70)
existence_tags=(e030 e050 e070)
prompt_variants=(whole_body_lesion tumor)
score_workers="${SCORE_WORKERS:-8}"
[[ "${score_workers}" =~ ^[0-9]+$ && "${score_workers}" -ge 1 ]] || {
  echo "SCORE_WORKERS must be a positive integer" >&2
  exit 1
}

wait_pair() {
  local left="$1" right="$2" status=0
  wait "${left}" || status=$?
  wait "${right}" || status=$?
  return "${status}"
}

infer_variant() {
  local gpu="$1" variant="$2" prompt="$3"
  local npz_output="${output}/${variant}_npz"
  local inference_log="${output}/${variant}.inference.log"
  mkdir -p "${npz_output}"
  local complete=1
  [[ $(find "${npz_output}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]] || complete=0
  for cached_tag in "${existence_tags[@]}"; do
    local cached_dir="${output}/${variant}_${cached_tag}_nifti"
    [[ -d "${cached_dir}" ]] || { complete=0; continue; }
    [[ $(find "${cached_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || complete=0
  done
  if [[ "${complete}" -eq 1 ]]; then
    echo "${variant}: all 50 NPZ files and three NIfTI thresholds cached"
    return 0
  fi
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz_output}" \
      --prompt "${prompt}" --device cuda --slice-batch-size 8 \
      --existence-threshold 0.30 --existence-threshold 0.50 \
      --existence-threshold 0.70 --continue-prediction > "${inference_log}" 2>&1; then
    echo "batch 8 failed; resuming complete outputs with official batch 4" \
      >> "${inference_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz_output}" \
      --prompt "${prompt}" --device cuda --slice-batch-size 4 \
      --existence-threshold 0.30 --existence-threshold 0.50 \
      --existence-threshold 0.70 --continue-prediction >> "${inference_log}" 2>&1
  fi
  for index in "${!existence_tags[@]}"; do
    local tag="${existence_tags[${index}]}"
    local nifti_output="${output}/${variant}_${tag}_nifti"
    mkdir -p "${nifti_output}"
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${images}" --predictions "${npz_output}" --output "${nifti_output}" \
      --segmentation-key "segs_${tag}" --continue-prediction \
      > "${output}/${variant}_${tag}.convert.log" 2>&1
    [[ $(find "${nifti_output}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
      echo "${variant}_${tag}: expected 50 converted masks" >&2
      return 1
    }
  done
}

infer_variant 0 whole_body_lesion "whole-body lesion" & pid0=$!
infer_variant 1 tumor "Tumor" & pid1=$!
wait_pair "${pid0}" "${pid1}"

# Standalone scores establish whether the external model contributes genuine
# lesion recall rather than merely a denser crop grid.
printf 'mode\tvariant\tproposal_min_voxels\tfocus_threshold\tmin_mm3\tDSC\tNSD\tcombined\tmetrics\tpredictions\n' \
  > "${output}/summary.tsv"
for prompt_variant in "${prompt_variants[@]}"; do
 for index in "${!existence_tags[@]}"; do
  existence="${existence_thresholds[${index}]}"
  tag="${existence_tags[${index}]}"
  variant="${prompt_variant}_${tag}"
  prediction="${output}/${variant}_nifti"
  metrics="${output}/scored/standalone_${variant}.metrics.csv"
  score_log="${output}/scored/standalone_${variant}.score.log"
  if [[ ! -s "${metrics}" ]]; then
    "${python_bin}" "${gf}/evaluate_public_parallel.py" --pred "${prediction}" \
      --gt "${labels}" --csv "${metrics}" --workers "${score_workers}" \
      > "${score_log}" 2>&1
  fi
  # The per-case CSV is authoritative and remains valid even if an earlier
  # recovery restart interrupted only the human-readable score log.
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.4f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.4f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf 'standalone\t%s\t-\t-\t0\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${prediction}" \
    >> "${output}/summary.tsv"
  "${python_bin}" "${gf}/analyze_lesion_errors.py" --labels "${labels}" \
    --predictions "${prediction}" --output "${output}/component_audit_${variant}" \
    > "${output}/component_audit_${variant}.log" 2>&1
 done
done

# The current Focus oracle is only +0.0525 DSC above the anchor, so also test
# the independent model directly. The finite, predeclared family covers
# standalone support, support-component addition, and GF-component
# confirmation. Its Python screen computes DSC for all configurations but
# materializes only two winners per family (plus any distinct global winners)
# for the exact official NSD scorer.
fusion_root="${output}/direct_fusion"
fusion_args=()
for prompt_variant in "${prompt_variants[@]}"; do
  for tag in "${existence_tags[@]}"; do
    fusion_args+=(--support "${prompt_variant}_${tag}=${output}/${prompt_variant}_${tag}_nifti")
  done
done
direct_screen_complete=0
if [[ -s "${fusion_root}/screen.json" && -s "${fusion_root}/shortlist.tsv" ]]; then
  direct_screen_complete=1
  # Accept older csv-module manifests whose last field ends in CRLF.
  while IFS=$'\t\r' read -r _ _ _ _ _ _ _ _ _ _ _ _ _ _ predictions; do
    [[ -d "${predictions}" \
       && $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] \
      || direct_screen_complete=0
  done < <(tail -n +2 "${fusion_root}/shortlist.tsv")
fi
if [[ "${direct_screen_complete}" -eq 0 ]]; then
  if [[ -s "${fusion_root}/screen.json" ]]; then
    "${python_bin}" "${root}/biomedparse_fusion.py" resume-screen \
      --screen-json "${fusion_root}/screen.json" --output "${fusion_root}" --workers 3 \
      > "${fusion_root}.resume.log" 2>&1
  else
    "${python_bin}" "${root}/biomedparse_fusion.py" screen \
      --base "${anchor_predictions}" "${fusion_args[@]}" --labels "${labels}" \
      --output "${fusion_root}" --workers 4 > "${fusion_root}.screen.log" 2>&1
  fi
else
  echo "direct BiomedParse fusion screen and shortlist predictions cached"
fi
printf 'tag\tsupport\tkind\tconfig\tscreen_DSC\tDSC\tNSD\tcombined\tmetrics\tpredictions\n' \
  > "${fusion_root}/official_summary.tsv"
direct_candidate_args=()
while IFS=$'\t\r' read -r _ _ _ _ _ _ tag _ _ _ _ _ _ _ predictions; do
  direct_candidate_args+=(--candidate "${tag}=${predictions}")
done < <(tail -n +2 "${fusion_root}/shortlist.tsv")
"${python_bin}" "${gf}/evaluate_public_grid.py" --gt "${labels}" \
  --output-dir "${fusion_root}" --workers "${score_workers}" \
  "${direct_candidate_args[@]}" > "${fusion_root}/official_grid.score.log" 2>&1
while IFS=$'\t\r' read -r support kind mode support_min distance rescue tag \
    screen_dsc screen_gain screen_median screen_win screen_loss screen_catastrophic \
    config predictions; do
  metrics="${fusion_root}/${tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.4f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.4f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${support}" "${kind}" "${config}" "${screen_dsc}" \
    "${dsc}" "${nsd}" "${combined}" "${metrics}" "${predictions}" \
    >> "${fusion_root}/official_summary.tsv"
done < <(tail -n +2 "${fusion_root}/shortlist.tsv")
{ head -1 "${fusion_root}/official_summary.tsv"; \
  tail -n +2 "${fusion_root}/official_summary.tsv" | sort -t $'\t' -k8,8gr; } \
  > "${fusion_root}/official_summary_by_combined.tsv"

guided_variant() {
  local gpu="$1" variant="$2" proposal_min_voxels="$3"
  local proposal_path="${4:-${output}/${variant}_nifti}"
  local destination="${output}/gf_guided_${variant}_v${proposal_min_voxels}"
  local inference_log="${output}/gf_guided_${variant}_v${proposal_min_voxels}.inference.log"
  mkdir -p "${destination}"
  # A resumed audit should not make MONAI reload every public CT merely to
  # discover that both threshold outputs already exist. Require the exact
  # expected count in each directory before taking this cache fast path.
  if [[ $(find "${destination}/lesion_0.50" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 50 ]] && \
     [[ $(find "${destination}/lesion_0.60" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 50 ]]; then
    printf 'already complete; skipping\n' > "${inference_log}"
    return 0
  fi
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${destination}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
      --proposal_mask_path "${proposal_path}" \
      --proposal_min_voxels "${proposal_min_voxels}" \
      --focus_batch_size 16 --workers 2 --skip_existing \
      > "${inference_log}" 2>&1; then
    echo "Focus batch 16 failed; resuming guided cases at batch 8" \
      >> "${inference_log}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --proposal_mask_path "${proposal_path}" \
        --proposal_min_voxels "${proposal_min_voxels}" \
        --focus_batch_size 8 --workers 2 --skip_existing \
        >> "${inference_log}" 2>&1; then
      echo "Focus batch 8 failed; resuming guided cases at batch 2" \
        >> "${inference_log}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --proposal_mask_path "${proposal_path}" \
        --proposal_min_voxels "${proposal_min_voxels}" \
        --focus_batch_size 2 --workers 2 --skip_existing \
        >> "${inference_log}" 2>&1
    fi
  fi
}

guided_prompt() {
  local gpu="$1" prompt_variant="$2"
  local tag
  for tag in "${existence_tags[@]}"; do
    if [[ "${tag}" == "e030" ]]; then
      guided_variant "${gpu}" "${prompt_variant}_${tag}" 500
    elif [[ "${tag}" == "e050" ]]; then
      guided_variant "${gpu}" "${prompt_variant}_${tag}" 1
      guided_variant "${gpu}" "${prompt_variant}_${tag}" 500
    else
      guided_variant "${gpu}" "${prompt_variant}_${tag}" 1
    fi
  done
}

if [[ -n "${WAIT_BEFORE_GUIDED_PID:-}" ]]; then
  [[ "${WAIT_BEFORE_GUIDED_PID}" =~ ^[0-9]+$ ]] || {
    echo "WAIT_BEFORE_GUIDED_PID must be numeric" >&2
    exit 1
  }
  echo "waiting for GPU factorial PID ${WAIT_BEFORE_GUIDED_PID} before guided inference"
  while kill -0 "${WAIT_BEFORE_GUIDED_PID}" 2>/dev/null; do sleep 15; done
fi
guided_prompt 0 whole_body_lesion & guided0=$!
guided_prompt 1 tumor & guided1=$!
wait_pair "${guided0}" "${guided1}"

# Prompt OR adds complementary lesion locations but is too noisy for direct
# union. Use the two predeclared component filters only as additional crop
# proposals; GF Focus still determines every output boundary.
guided_variant 0 prompt_or_e050_mm100 1 \
  "${combined_support_root}/union_e050_mm100" & guided_filtered0=$!
guided_variant 1 prompt_or_e050_mm500 1 \
  "${combined_support_root}/union_e050_mm500" & guided_filtered1=$!
wait_pair "${guided_filtered0}" "${guided_filtered1}"

guided_manifest="${output}/guided_candidates.tsv"
printf 'variant\tproposal_min_voxels\tfocus_threshold\tmin_mm3\ttag\tpredictions\n' \
  > "${guided_manifest}"
append_guided_manifest() {
  local variant="$1" proposal_min_voxels="$2"
  local focus source minimum score_tag predictions
  for focus in 0.50 0.60; do
    source="${output}/gf_guided_${variant}_v${proposal_min_voxels}/lesion_${focus}"
    [[ $(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
      echo "${source}: expected 50 masks" >&2
      return 1
    }
    for minimum in 0 500 1000; do
      score_tag="guided_${variant}_v${proposal_min_voxels}_l${focus}_mm${minimum}"
      predictions="${source}"
      if [[ "${minimum}" -gt 0 ]]; then
        predictions="${output}/scored/${score_tag}"
        mkdir -p "${predictions}"
        if [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
          "${python_bin}" "${gf}/fuse_and_filter_masks.py" --pred-a "${source}" \
            --fusion a --min-mm3 "${minimum}" --output "${predictions}" \
            > "${output}/scored/${score_tag}.filter.log" 2>&1
        fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${variant}" "${proposal_min_voxels}" "${focus}" "${minimum}" \
        "${score_tag}" "${predictions}" >> "${guided_manifest}"
    done
  done
}
for prompt_variant in "${prompt_variants[@]}"; do
  for existence_tag in "${existence_tags[@]}"; do
    variant="${prompt_variant}_${existence_tag}"
    proposal_minima=(1)
    [[ "${existence_tag}" == "e030" ]] && proposal_minima=(500)
    [[ "${existence_tag}" == "e050" ]] && proposal_minima=(1 500)
    for proposal_min_voxels in "${proposal_minima[@]}"; do
      append_guided_manifest "${variant}" "${proposal_min_voxels}"
    done
  done
done
for proposal_minimum in "${filtered_proposal_minima[@]}"; do
  append_guided_manifest "prompt_or_e050_mm${proposal_minimum}" 1
done

guided_candidate_args=()
while IFS=$'\t' read -r _ _ _ _ score_tag predictions; do
  guided_candidate_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${guided_manifest}")
"${python_bin}" "${gf}/evaluate_public_grid.py" --gt "${labels}" \
  --output-dir "${output}/scored" --workers "${score_workers}" \
  "${guided_candidate_args[@]}" > "${output}/guided_grid.score.log" 2>&1

while IFS=$'\t' read -r variant proposal_min_voxels focus minimum score_tag predictions; do
  metrics="${output}/scored/${score_tag}.metrics.csv"
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.4f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.4f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf 'guided\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${proposal_min_voxels}" "${focus}" "${minimum}" \
    "${dsc}" "${nsd}" "${combined}" "${metrics}" "${predictions}" \
    >> "${output}/summary.tsv"
done < <(tail -n +2 "${guided_manifest}")

{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k8,8gr; } \
  > "${output}/summary_by_combined.tsv"
deployment_manifest="${output}/deployment_candidates.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\tselected_mode\tsupport\tproposal_min_voxels\tconfig\n' \
  > "${deployment_manifest}"
while IFS=$'\t' read -r variant proposal_min_voxels focus minimum score_tag predictions; do
  printf 'guided_%s\tanchor\t%s\t%s\t%s\t%s\t%s\t%s/scored/%s.metrics.csv\tguided\t%s\t%s\t-\n' \
    "${variant}" "${checkpoint}" "${focus}" "${minimum}" "${score_tag}" \
    "${predictions}" "${output}" "${score_tag}" "${variant}" \
    "${proposal_min_voxels}" >> "${deployment_manifest}"
done < <(tail -n +2 "${guided_manifest}")
while IFS=$'\t' read -r direct_tag direct_support direct_kind direct_config _ _ _ _ \
    direct_metrics direct_predictions; do
  printf 'direct_%s_%s\tanchor\t%s\t0.50\t0\tdirect_%s\t%s\t%s\tdirect\t%s\t0\t%s\n' \
    "${direct_support}" "${direct_kind}" "${checkpoint}" "${direct_tag}" \
    "${direct_predictions}" "${direct_metrics}" "${direct_support}" \
    "${direct_config}" >> "${deployment_manifest}"
done < <(tail -n +2 "${fusion_root}/official_summary.tsv")

prompt_rank_root="${output}/prompted_ranking"
"${python_bin}" "${gf}/rank_prompted_manifest.py" \
  --manifest "${deployment_manifest}" --output "${prompt_rank_root}" \
  --labels "${labels}" --prompt-support "${public_prompt_support}" \
  --prompt-config "${prompt_config}" --prompt-anchor "${prompt_anchor}" \
  --fusion-script "${root}/biomedparse_fusion.py" \
  --evaluator "${gf}/evaluate_prompted_grid.py" \
  --selector "${gf}/select_checkpoint_robust.py" --workers "${score_workers}" \
  > "${output}/prompted_ranking.log" 2>&1
selected_line=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r _ _ selected_checkpoint guided_focus guided_min selected_score_tag \
  raw_selected_predictions raw_selected_metrics selected_mode selected_support \
  guided_proposal_min selected_config selected_predictions selected_metrics \
  selected_prompt_robust prompt_decision selected_dsc selected_nsd \
  selected_combined deployment_tier <<< "${selected_line}"
candidate_tag="biomedparse_${selected_score_tag}"
guided_variant="${selected_support}"
[[ "${selected_config}" == "-" ]] && selected_config=""
cp "${selected_prompt_robust}" "${output}/robust_selection_prompt_confirmed.json"
cp "${selected_prompt_robust}" "${output}/robust_selection.json"
"${python_bin}" "${root}/select_materialization.py" \
  --robust-selection "${output}/robust_selection_prompt_confirmed.json" \
  --output "${output}/public_materialization_decision_prompt_confirmed.json" \
  > "${output}/public_materialization_decision_prompt_confirmed.log"
cp "${output}/public_materialization_decision_prompt_confirmed.json" \
  "${output}/public_materialization_decision.json"
printf 'branch\ttag\tsupport\tcombined\tDSC\tNSD\tmetrics\tpredictions\tconfig\tdeployment_tier\n' \
  > "${output}/selected_public_configuration.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${selected_mode}" "${candidate_tag}" "${selected_support}" \
  "${selected_combined}" "${selected_dsc}" "${selected_nsd}" \
  "${selected_metrics}" "${selected_predictions}" "${selected_config:--}" \
  "${deployment_tier}" >> "${output}/selected_public_configuration.tsv"
prompt_gate_passed=0
if [[ "${prompt_decision}" == "PROMOTE" ]]; then
  prompt_gate_passed=1
  public_decision="PROMOTE"
else
  public_decision="KEEP"
  printf '%s\n' "${candidate_tag}" > "${output}/not_promoted_after_prompt_confirmation.txt"
fi
"${python_bin}" "${gf}/analyze_lesion_errors.py" --labels "${labels}" \
  --predictions "${selected_predictions}" --output "${output}/component_audit_selected" \
  > "${output}/component_audit_selected.log" 2>&1
decision="KEEP"
# The user requested three novel submission-ready candidates. Run the healthy
# safety audit and materialize the strongest guided-family fallback even when
# the conservative paired confidence gate says KEEP; its fallback status is
# retained in materialization_decision.json and the final shortlist.
if true; then
  proposal_source="single_prompt"
  selected_proposal_filter_mm3=0
  case "${selected_support}" in
    whole_body_lesion_e*) best_prompt="whole-body lesion" ;;
    tumor_e*) best_prompt="Tumor" ;;
    prompt_or_e050_mm100)
      proposal_source="prompt_or_cached"
      selected_proposal_filter_mm3=100
      best_prompt="whole-body lesion"
      ;;
    prompt_or_e050_mm500)
      proposal_source="prompt_or_cached"
      selected_proposal_filter_mm3=500
      best_prompt="whole-body lesion"
      ;;
    *) echo "unknown selected BiomedParse variant: ${selected_support}" >&2; exit 1 ;;
  esac
  if [[ "${proposal_source}" == "prompt_or_cached" ]]; then
    best_existence_tag="e050"
  else
    best_existence_tag="${selected_support##*_}"
  fi
  case "${best_existence_tag}" in
    e030) best_existence=0.30 ;;
    e050) best_existence=0.50 ;;
    e070) best_existence=0.70 ;;
    *) echo "unknown existence tag: ${best_existence_tag}" >&2; exit 1 ;;
  esac

  # Public validation contains lesion-bearing cases, so independently guard
  # against new false positives on a prediction-blind, stratified sample of
  # 32 held-out lesion-free CTs (8 CHAOS, 9 train, 15 PANCREAS).
  healthy_root="${output}/healthy_safety_${candidate_tag}"
  healthy_inputs="${healthy_root}/inputs"
  healthy_npz="${healthy_root}/biomedparse_npz"
  healthy_proposals="${healthy_root}/proposals"
  mkdir -p "${healthy_inputs}" "${healthy_npz}" \
    "${healthy_proposals}" "${healthy_root}/anchor_raw" \
    "${healthy_root}/candidate_raw" "${healthy_root}/anchor_filtered" \
    "${healthy_root}/candidate_filtered"
  prepare_prompts=(--prompt "${best_prompt}")
  if [[ "${proposal_source}" == "prompt_or_cached" ]]; then
    prepare_prompts=(--prompt "whole-body lesion" --prompt "Tumor")
  fi
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py prepare \
    --images "${healthy_subset}" --output "${healthy_inputs}" \
    "${prepare_prompts[@]}" --a-min -160 --a-max 240 --continue-prediction \
    > "${healthy_root}/prepare.log" 2>&1
  [[ $(find "${healthy_inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 32 ]] || {
    echo "expected 32 prepared healthy inputs" >&2
    exit 1
  }

  infer_healthy_shard() {
    local gpu="$1" shard="$2" prompt="$3" destination="$4" tag="$5"
    local log="${healthy_root}/biomedparse_${tag}_shard${shard}.log"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
        ../miccai_inference.py --input-dir "${healthy_inputs}" --output-dir "${destination}" \
        --prompt "${prompt}" --device cuda --slice-batch-size 8 \
        --existence-threshold "${best_existence}" --continue-prediction \
        --num-shards 2 --shard-id "${shard}" > "${log}" 2>&1; then
      echo "batch 8 failed; resuming healthy shard with official batch 4" >> "${log}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
        ../miccai_inference.py --input-dir "${healthy_inputs}" --output-dir "${destination}" \
        --prompt "${prompt}" --device cuda --slice-batch-size 4 \
        --existence-threshold "${best_existence}" --continue-prediction \
        --num-shards 2 --shard-id "${shard}" >> "${log}" 2>&1
    fi
  }
  if [[ "${proposal_source}" == "single_prompt" ]]; then
    infer_healthy_shard 0 0 "${best_prompt}" "${healthy_npz}" single & biomed_healthy0=$!
    infer_healthy_shard 1 1 "${best_prompt}" "${healthy_npz}" single & biomed_healthy1=$!
    wait_pair "${biomed_healthy0}" "${biomed_healthy1}"
    [[ $(find "${healthy_npz}" -maxdepth 1 -name '*.npz' | wc -l) -eq 32 ]] || {
      echo "expected 32 healthy BiomedParse predictions" >&2
      exit 1
    }
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${healthy_subset}" --predictions "${healthy_npz}" \
      --output "${healthy_proposals}" --continue-prediction \
      > "${healthy_root}/convert.log" 2>&1
  else
    healthy_whole_npz="${healthy_root}/whole_npz"
    healthy_tumor_npz="${healthy_root}/tumor_npz"
    healthy_whole_masks="${healthy_root}/whole_masks"
    healthy_tumor_masks="${healthy_root}/tumor_masks"
    healthy_union_raw="${healthy_root}/prompt_or_raw"
    mkdir -p "${healthy_whole_npz}" "${healthy_tumor_npz}" \
      "${healthy_whole_masks}" "${healthy_tumor_masks}" "${healthy_union_raw}"
    infer_healthy_shard 0 0 "whole-body lesion" "${healthy_whole_npz}" whole & biomed_healthy0=$!
    infer_healthy_shard 1 1 "whole-body lesion" "${healthy_whole_npz}" whole & biomed_healthy1=$!
    wait_pair "${biomed_healthy0}" "${biomed_healthy1}"
    infer_healthy_shard 0 0 "Tumor" "${healthy_tumor_npz}" tumor & biomed_healthy0=$!
    infer_healthy_shard 1 1 "Tumor" "${healthy_tumor_npz}" tumor & biomed_healthy1=$!
    wait_pair "${biomed_healthy0}" "${biomed_healthy1}"
    for healthy_prediction_dir in "${healthy_whole_npz}" "${healthy_tumor_npz}"; do
      [[ $(find "${healthy_prediction_dir}" -maxdepth 1 -name '*.npz' | wc -l) -eq 32 ]] || {
        echo "expected 32 healthy prompt predictions in ${healthy_prediction_dir}" >&2
        exit 1
      }
    done
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${healthy_subset}" --predictions "${healthy_whole_npz}" \
      --segmentation-key segs_e050 --output "${healthy_whole_masks}" \
      --continue-prediction > "${healthy_root}/whole.convert.log" 2>&1
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${healthy_subset}" --predictions "${healthy_tumor_npz}" \
      --segmentation-key segs_e050 --output "${healthy_tumor_masks}" \
      --continue-prediction > "${healthy_root}/tumor.convert.log" 2>&1
    "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
      --pred-a "${healthy_whole_masks}" --pred-b "${healthy_tumor_masks}" \
      --fusion union --min-mm3 0 --output "${healthy_union_raw}" \
      > "${healthy_root}/prompt_or.log" 2>&1
    "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
      --pred-a "${healthy_union_raw}" --fusion a \
      --min-mm3 "${selected_proposal_filter_mm3}" --output "${healthy_proposals}" \
      > "${healthy_root}/proposal_filter.log" 2>&1
  fi
  [[ $(find "${healthy_proposals}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 32 ]] || {
    echo "expected 32 healthy proposal masks" >&2
    exit 1
  }

  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
    --test_data_path "${healthy_subset}" \
    --save_prediction_path "${healthy_root}/anchor_raw" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_batch_size 2 --workers 2 --skip_existing \
    > "${healthy_root}/anchor.inference.log" 2>&1 & healthy_anchor_pid=$!
  if [[ "${selected_mode}" == "guided" ]]; then
    CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
      --test_data_path "${healthy_subset}" \
      --save_prediction_path "${healthy_root}/candidate_raw" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_threshold "${guided_focus}" \
      --proposal_mask_path "${healthy_proposals}" \
      --proposal_min_voxels "${guided_proposal_min}" \
      --focus_batch_size 2 --workers 2 --skip_existing \
      > "${healthy_root}/candidate.inference.log" 2>&1 & healthy_candidate_pid=$!
    wait_pair "${healthy_anchor_pid}" "${healthy_candidate_pid}"
  else
    wait "${healthy_anchor_pid}"
  fi
  [[ $(find "${healthy_root}/anchor_raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 32 ]] || {
    echo "expected 32 anchor healthy predictions" >&2
    exit 1
  }
  "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
    --pred-a "${healthy_root}/anchor_raw" --fusion a --min-mm3 1000 \
    --output "${healthy_root}/anchor_filtered" > "${healthy_root}/anchor.filter.log" 2>&1
  if [[ "${selected_mode}" == "direct" ]]; then
    "${python_bin}" "${root}/biomedparse_fusion.py" materialize \
      --base "${healthy_root}/anchor_filtered" --support "${healthy_proposals}" \
      --config "${selected_config}" --output "${healthy_root}/candidate_filtered" \
      > "${healthy_root}/candidate.fusion.log" 2>&1
  elif [[ "${guided_min}" -gt 0 ]]; then
    "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
      --pred-a "${healthy_root}/candidate_raw" --fusion a --min-mm3 "${guided_min}" \
      --output "${healthy_root}/candidate_filtered" > "${healthy_root}/candidate.filter.log" 2>&1
  else
    cp "${healthy_root}/candidate_raw"/*.nii.gz "${healthy_root}/candidate_filtered/"
  fi
  [[ $(find "${healthy_root}/candidate_filtered" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 32 ]] || {
    echo "expected 32 candidate healthy predictions" >&2
    exit 1
  }
  "${python_bin}" "${gf}/evaluate_healthy_predictions.py" \
    --pred "${healthy_root}/anchor_filtered" \
    --output "${healthy_root}/anchor.metrics.csv" --expected-count 32 \
    > "${healthy_root}/anchor.score.log"
  "${python_bin}" "${gf}/evaluate_healthy_predictions.py" \
    --pred "${healthy_root}/candidate_filtered" \
    --output "${healthy_root}/candidate.metrics.csv" --expected-count 32 \
    > "${healthy_root}/candidate.score.log"
  decision=$("${python_bin}" "${root}/select_materialization.py" \
    --robust-selection "${output}/robust_selection.json" \
    --healthy-anchor "${healthy_root}/anchor.metrics.summary.json" \
    --healthy-candidate "${healthy_root}/candidate.metrics.summary.json" \
    --output "${output}/materialization_decision.json")
else
  if [[ "${prompt_gate_passed}" -eq 0 ]]; then
    cp "${output}/public_materialization_decision_prompt_confirmed.json" \
      "${output}/materialization_decision.json"
  else
    cp "${output}/public_materialization_decision.json" \
      "${output}/materialization_decision.json"
  fi
fi

if true; then

  hidden_root="${output}/hidden_${candidate_tag}"
  hidden_inputs="${hidden_root}/inputs"
  hidden_npz="${hidden_root}/biomedparse_npz"
  hidden_proposals="${hidden_root}/proposals"
  mkdir -p "${hidden_inputs}" "${hidden_npz}" "${hidden_proposals}" \
    "${hidden_root}/shard0" "${hidden_root}/shard1" \
    "${hidden_root}/raw" "${hidden_root}/filtered" \
    "${hidden_root}/prompt_confirmed" "${project}/submissions"
  if [[ "${proposal_source}" == "single_prompt" ]]; then
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py prepare \
      --images "${hidden_images}" --output "${hidden_inputs}" \
      --prompt "${best_prompt}" --a-min -160 --a-max 240 --continue-prediction \
      > "${hidden_root}/prepare.log" 2>&1
    [[ $(find "${hidden_inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]] || {
      echo "expected 100 prepared hidden inputs" >&2
      exit 1
    }

    infer_hidden_shard() {
      local gpu="$1" shard="$2"
      local log="${hidden_root}/biomedparse_shard${shard}.log"
      if ! CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
          ../miccai_inference.py --input-dir "${hidden_inputs}" --output-dir "${hidden_npz}" \
          --prompt "${best_prompt}" --device cuda --slice-batch-size 8 \
          --existence-threshold "${best_existence}" --continue-prediction \
          --num-shards 2 --shard-id "${shard}" > "${log}" 2>&1; then
        echo "batch 8 failed; resuming shard with official batch 4" >> "${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
          ../miccai_inference.py --input-dir "${hidden_inputs}" --output-dir "${hidden_npz}" \
          --prompt "${best_prompt}" --device cuda --slice-batch-size 4 \
          --existence-threshold "${best_existence}" --continue-prediction \
          --num-shards 2 --shard-id "${shard}" >> "${log}" 2>&1
      fi
    }
    infer_hidden_shard 0 0 & biomed_hidden0=$!
    infer_hidden_shard 1 1 & biomed_hidden1=$!
    wait_pair "${biomed_hidden0}" "${biomed_hidden1}"
    [[ $(find "${hidden_npz}" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]] || {
      echo "expected 100 hidden BiomedParse predictions" >&2
      exit 1
    }
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${hidden_images}" --predictions "${hidden_npz}" \
      --output "${hidden_proposals}" --continue-prediction \
      > "${hidden_root}/convert.log" 2>&1
  else
    # The exact hidden prompt-OR e050 masks are already cached and validated.
    # Reproduce the public component filter without re-running either prompt.
    "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
      --pred-a "${hidden_prompt_support}" --fusion a \
      --min-mm3 "${selected_proposal_filter_mm3}" --workers 4 \
      --output "${hidden_proposals}" > "${hidden_root}/proposal_filter.log" 2>&1
  fi
  [[ $(find "${hidden_proposals}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 hidden BiomedParse proposal masks" >&2
    exit 1
  }

  if [[ "${selected_mode}" == "direct" ]]; then
    "${python_bin}" "${root}/biomedparse_fusion.py" materialize \
      --base "${hidden_anchor_predictions}" --support "${hidden_proposals}" \
      --config "${selected_config}" --output "${hidden_root}/filtered" \
      > "${hidden_root}/fusion.log" 2>&1
  else
    for shard in 0 1; do
      CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
        --test_data_path "${hidden_images}" \
        --save_prediction_path "${hidden_root}/shard${shard}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_threshold "${guided_focus}" \
        --proposal_mask_path "${hidden_proposals}" \
        --proposal_min_voxels "${guided_proposal_min}" \
        --focus_batch_size 2 --workers 2 --skip_existing \
        --num_shards 2 --shard_id "${shard}" \
        > "${hidden_root}/gf_shard${shard}.log" 2>&1 &
      [[ "${shard}" -eq 0 ]] && gf_hidden0=$! || gf_hidden1=$!
    done
    wait_pair "${gf_hidden0}" "${gf_hidden1}"
    cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
    cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
    [[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
      echo "expected 100 hidden guided predictions" >&2
      exit 1
    }
    if [[ "${guided_min}" -gt 0 ]]; then
      "${python_bin}" "${gf}/fuse_and_filter_masks.py" \
        --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${guided_min}" \
        --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
    else
      cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
    fi
  fi
  [[ $(find "${hidden_root}/filtered" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 selected hidden predictions" >&2
    exit 1
  }
  "${python_bin}" "${root}/biomedparse_fusion.py" materialize \
    --base "${hidden_root}/filtered" --support "${hidden_prompt_support}" \
    --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
    > "${hidden_root}/prompt_materialize.log" 2>&1
  [[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 prompt-confirmed selected hidden predictions" >&2
    exit 1
  }
  submission="${project}/submissions/gf_${candidate_tag}_prompt_or_e050.zip"
  "${python_bin}" "${project}/nnunet/package_codabench.py" \
    --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
    --output "${submission}" --expected-count 100 \
    > "${hidden_root}/package.log" 2>&1
  unzip -t "${submission}" > "${hidden_root}/zip_test.log"
  sha256sum "${submission}" > "${submission}.sha256"
  printf '%s\n' "${submission}" > "${output}/MATERIALIZED_SUBMISSION.txt"
fi
"${python_bin}" - "${output}" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
submission_file = root / "MATERIALIZED_SUBMISSION.txt"
materialization = json.loads((root / "materialization_decision.json").read_text())
with (root / "selected_public_configuration.tsv").open(newline="") as stream:
    selected = next(csv.DictReader(stream, delimiter="\t"))
(root / "selection_candidates.json").write_text(
    (root / "prompted_ranking" / "ranking.json").read_text()
)
(root / "AUDIT_COMPLETE.json").write_text(json.dumps({
    "status": "complete",
    "materialization": materialization,
    "submission": submission_file.read_text().strip() if submission_file.exists() else None,
    "selected_public_configuration": selected,
}, indent=2) + "\n")
PY
echo "BiomedParse automatic-proposal public audit complete: ${output}"

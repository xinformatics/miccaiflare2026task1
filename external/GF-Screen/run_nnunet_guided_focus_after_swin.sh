#!/usr/bin/env bash
set -euo pipefail

# Prevent two resumptions from writing the same manifest and score folders.
lock_file=".run_nnunet_guided_focus_after_swin.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another nnU-Net-guided Focus launcher is already active" >&2
  exit 1
fi

# Use an independent full-volume nnU-Net mask only as a crop proposal map.
# The GF Focus model, not nnU-Net, decides the final lesion boundary. This is
# structurally different from the rejected binary mask fusion experiments.
wait_pid="${WAIT_PID:-871016}"
skip_public_inference="${SKIP_PUBLIC_INFERENCE:-0}"
public_only="${PUBLIC_ONLY:-0}"
package_fallback="${PACKAGE_FALLBACK:-0}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
project_root="../.."
public_images="${project_root}/validation/Validation-Public-Images"
public_labels="${project_root}/validation/Validation-Public-Labels"
hidden_images="${project_root}/validation/Validation-Hidden-Images"
late_root="${project_root}/predictions/flare25_finetune_late_checkpoints"
epoch50_root="${project_root}/predictions/flare25_finetune_by_epoch/epoch_0050"
root="public_sweeps/nnunet_guided_focus_20260820"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
nnunet_model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/nnUNetTrainer__nnUNetPlans__3d_fullres_MONAI_SwinUNETR_f48_pretrained/fold_all"
biomed_root="${project_root}/external/BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
for required in "${prompt_config}" "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing prompt-confirmation asset: ${required}" >&2; exit 1; }
done
[[ $(find "${public_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
[[ $(find "${hidden_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}
mkdir -p "${root}" "${project_root}/submissions"

support_tag="epoch_0050"
support_public="${epoch50_root}/predictions"
support_checkpoint="checkpoint_epoch_0050.pth"
if [[ -s "${late_root}/summary_by_combined.tsv" ]]; then
  candidate_tag=$(sed -n '2p' "${late_root}/summary_by_combined.tsv" | cut -f1)
  candidate_combined=$(sed -n '2p' "${late_root}/summary_by_combined.tsv" | cut -f4)
  anchor_combined=$("${python_bin}" - "${epoch50_root}/metrics_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print((d["mean_dsc"]+d["mean_nsd"])/2)
PY
)
  if awk -v c="${candidate_combined}" -v a="${anchor_combined}" 'BEGIN {exit !(c>a)}'; then
    support_tag="${candidate_tag}"
    support_public="${late_root}/${candidate_tag}/predictions"
    support_checkpoint="checkpoint_${candidate_tag}.pth"
  fi
fi
if [[ $(find "${support_public}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
  echo "Selected proposal set is incomplete: ${support_public}" >&2
  exit 1
fi
printf 'support_tag\tsupport_public\tsupport_checkpoint\n%s\t%s\t%s\n' \
  "${support_tag}" "${support_public}" "${support_checkpoint}" > "${root}/support.tsv"

infer_variant() {
  local gpu="$1"
  local min_voxels="$2"
  local output="${root}/proposal_v${min_voxels}"
  mkdir -p "${output}"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
      --test_data_path "${public_images}" --save_prediction_path "${output}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
      --proposal_mask_path "${support_public}" --proposal_min_voxels "${min_voxels}" \
      --focus_batch_size 16 --workers 2 --skip_existing \
      > "${output}.inference.log" 2>&1; then
    echo "Focus batch 16 failed; resuming guided cases at batch 8" \
      >> "${output}.inference.log"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
        --test_data_path "${public_images}" --save_prediction_path "${output}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --proposal_mask_path "${support_public}" --proposal_min_voxels "${min_voxels}" \
        --focus_batch_size 8 --workers 2 --skip_existing \
        >> "${output}.inference.log" 2>&1; then
      echo "Focus batch 8 failed; resuming guided cases at batch 2" \
        >> "${output}.inference.log"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
        --test_data_path "${public_images}" --save_prediction_path "${output}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
        --proposal_mask_path "${support_public}" --proposal_min_voxels "${min_voxels}" \
        --focus_batch_size 2 --workers 2 --skip_existing \
        >> "${output}.inference.log" 2>&1
    fi
  fi
}

if [[ "${skip_public_inference}" == "1" ]]; then
  for min_voxels in 1 500; do
    for focus in 0.50 0.60; do
      completed=$(find "${root}/proposal_v${min_voxels}/lesion_${focus}" \
        -maxdepth 1 -name '*.nii.gz' | wc -l)
      [[ "${completed}" -eq 50 ]] || {
        echo "cannot skip public inference: v${min_voxels} l${focus} has ${completed}/50 masks" >&2
        exit 1
      }
    done
  done
else
  infer_variant 0 1 &
  pid1=$!
  infer_variant 1 500 &
  pid500=$!
  wait "${pid1}"
  wait "${pid500}"
fi

manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\tproposal_min_voxels\n' \
  > "${manifest}"
for min_voxels in 1 500; do
  for focus in 0.50 0.60; do
    source="${root}/proposal_v${min_voxels}/lesion_${focus}"
    for min_mm3 in 0 500 1000 3000; do
      score_stem="${root}/scored/v${min_voxels}_l${focus}_mm${min_mm3}"
      candidate="${source}"
      mkdir -p "${root}/scored"
      if [[ "${min_mm3}" -gt 0 ]]; then
        candidate="${score_stem}"
        mkdir -p "${candidate}"
        completed=$(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l)
        if [[ "${completed}" -eq 50 ]]; then
          printf 'reusing complete filtered set (%s masks)\n' "${completed}" \
            > "${score_stem}.filter.log"
        else
          "${python_bin}" -u fuse_and_filter_masks.py \
            --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
            --output "${candidate}" > "${score_stem}.filter.log" 2>&1
          completed=$(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l)
          [[ "${completed}" -eq 50 ]] || {
            echo "incomplete filtered set ${candidate}: ${completed}/50" >&2
            exit 1
          }
        fi
      fi
      score_tag="v${min_voxels}_l${focus}_mm${min_mm3}"
      printf 'nnunet_%s_v%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s.metrics.csv\t%s\n' \
        "${support_tag}" "${min_voxels}" "${support_tag}" "${checkpoint}" \
        "${focus}" "${min_mm3}" "${score_tag}" "${candidate}" "${score_stem}" \
        "${min_voxels}" >> "${manifest}"
    done
  done
done
grid_args=()
while IFS=$'\t' read -r _ _ _ _ _ score_tag predictions _ _; do
  grid_args+=(--candidate "${score_tag}=${predictions}")
done < <(tail -n +2 "${manifest}")
"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1

printf 'proposal_min_voxels\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
while IFS=$'\t' read -r _ _ _ focus min_mm3 _ _ metrics min_voxels; do
  dsc=$(awk -F, 'NR > 1 {sum += $2; n++} END {printf "%.6f", sum/n}' "${metrics}")
  nsd=$(awk -F, 'NR > 1 {sum += $3; n++} END {printf "%.6f", sum/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${min_voxels}" "${focus}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done < <(tail -n +2 "${manifest}")
{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

prompt_rank_root="${root}/prompted_ranking"
"${python_bin}" -u rank_prompted_manifest.py --manifest "${manifest}" \
  --output "${prompt_rank_root}" --labels "${public_labels}" \
  --prompt-support "${public_prompt_support}" --prompt-config "${prompt_config}" \
  --prompt-anchor "${prompt_anchor}" --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator evaluate_prompted_grid.py --selector select_checkpoint_robust.py \
  --workers 8 > "${root}/prompted_ranking.log" 2>&1
best=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r _ _ best_checkpoint best_focus best_mm3 best_score_tag \
  selected_predictions metrics best_voxels public_prompt prompt_metrics \
  prompt_robust prompt_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name gf_champion --candidate-name "nnunet_guided_v${best_voxels}_l${best_focus}_mm${best_mm3}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best}" > "${root}/not_promoted.tsv"
  echo "nnU-Net-guided Focus did not clear the raw paired gate; retaining fallback status."
fi
cp "${prompt_robust}" "${root}/robust_selection_prompt_confirmed.json"
if [[ "${prompt_decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best}" > "${root}/not_promoted_after_prompt_confirmation.tsv"
  echo "nnU-Net-guided Focus did not clear the deployed paired gate."
  if [[ "${package_fallback}" != "1" ]]; then
    echo "Stopping before hidden inference; set PACKAGE_FALLBACK=1 to build a ranked fallback explicitly."
    exit 0
  fi
fi
if [[ "${public_only}" == "1" ]]; then
  echo "PUBLIC_ONLY=1: public scoring complete; stopping before hidden inference."
  exit 0
fi

# Generate a proposal set for the identical support checkpoint on hidden data.
source "${project_root}/nnunet/env.sh"
support_hidden="${root}/hidden_support_${support_tag}"
mkdir -p "${support_hidden}/part0" "${support_hidden}/part1"
nn_common=(
  -i "${hidden_images}" -d 501 -tr nnUNetTrainer -p nnUNetPlans
  -c 3d_fullres_MONAI_SwinUNETR_f48_pretrained -f all
  -chk "${support_checkpoint}" -npp 2 -nps 2 -num_parts 2 --disable_tta
)
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict "${nn_common[@]}" -part_id 0 \
  -o "${support_hidden}/part0" > "${support_hidden}/part0.log" 2>&1 &
nnpid0=$!
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict "${nn_common[@]}" -part_id 1 \
  -o "${support_hidden}/part1" > "${support_hidden}/part1.log" 2>&1 &
nnpid1=$!
wait "${nnpid0}"
wait "${nnpid1}"
mkdir -p "${support_hidden}/combined"
cp "${support_hidden}/part0"/*.nii.gz "${support_hidden}/combined/"
cp "${support_hidden}/part1"/*.nii.gz "${support_hidden}/combined/"

hidden_root="${root}/hidden_guided_v${best_voxels}_l${best_focus}_mm${best_mm3}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --proposal_mask_path "${support_hidden}/combined" --proposal_min_voxels "${best_voxels}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then gfpid0=$!; else gfpid1=$!; fi
done
wait "${gfpid0}"
wait "${gfpid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
if [[ "${best_mm3}" -gt 0 ]]; then
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_mm3}" \
    --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
else
  cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
fi
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${hidden_prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden masks" >&2
  exit 1
}
submission="${project_root}/submissions/gf_nnunet_guided_${support_tag}_v${best_voxels}_l${best_focus}_mm${best_mm3}_prompt_or_e050.zip"
"${python_bin}" -u "${project_root}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100
unzip -t "${submission}" > "${submission}.zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best}" > "${hidden_root}/selected_public_configuration.tsv"
echo "nnU-Net-guided GF submission ready: ${submission}"

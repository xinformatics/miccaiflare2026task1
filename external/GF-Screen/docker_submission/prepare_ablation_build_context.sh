#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
context="${project}/external/GF-Screen/docker_submission/deployment_context"

if [[ -e "${context}" ]]; then
  echo "refusing to overwrite existing generated context: ${context}" >&2
  exit 2
fi

mkdir -p \
  "${context}/assets" \
  "${context}/external/GF-Screen/docker_submission" \
  "${context}/external/BiomedParse/container_workspace/model_weights" \
  "${context}/external/STU-Net/pancancer" \
  "${context}/external/STU-Net/nnUNet-2.2/nnunetv2/training/nnUNetTrainer" \
  "${context}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all"

# Hard links keep the generated build context space-neutral on this filesystem.
for asset in gf_bbox_full_state.pt gf_cbg_glance_state.pt biomedparse_3d_state_only.pt; do
  ln "${project}/external/GF-Screen/docker_submission/assets/${asset}" "${context}/assets/${asset}"
done

cp -al "${project}/external/GF-Screen/models" "${context}/external/GF-Screen/models"
cp -al "${project}/external/GF-Screen/utils" "${context}/external/GF-Screen/utils"
cp -al "${project}/external/GF-Screen/docker_submission/vendor" "${context}/external/GF-Screen/docker_submission/vendor"
for source in \
  inference_naming.py \
  docker_submission/predict_dual_glance_shared_focus.py \
  docker_submission/shared_focus_dual_glance.py \
  docker_submission/winner_mask_recipe.py \
  docker_submission/runtime_winner_mask_recipe.py \
  docker_submission/compose_deployment_masks.py; do
  destination="${context}/external/GF-Screen/${source}"
  mkdir -p "$(dirname "${destination}")"
  cp -al "${project}/external/GF-Screen/${source}" "${destination}"
done

for source in miccai_nifti_adapter.py miccai_dual_prompt_inference.py miccai_inference.py; do
  cp -al "${project}/external/BiomedParse/${source}" "${context}/external/BiomedParse/${source}"
done
cp -al "${project}/external/BiomedParse/container_workspace/configs" "${context}/external/BiomedParse/container_workspace/configs"
cp -al "${project}/external/BiomedParse/container_workspace/src" "${context}/external/BiomedParse/container_workspace/src"
cp -al "${project}/external/BiomedParse/container_workspace/inference.py" "${context}/external/BiomedParse/container_workspace/inference.py"
cp -al "${project}/external/BiomedParse/container_workspace/utils.py" "${context}/external/BiomedParse/container_workspace/utils.py"
cp -al "${project}/external/BiomedParse/container_workspace/model_weights/local_tokenizer" "${context}/external/BiomedParse/container_workspace/model_weights/local_tokenizer"

for source in \
  STUNetTrainerPanCancer.py \
  STUNetTrainerPanCancerCompleteHybrid.py \
  STUNetTrainerPanCancerSourceBalanced.py \
  compact_probability_export.py \
  predict_stunet_safe.py \
  predict_stunet_anchor_sparse.py \
  predict_stunet_proxy_sparse.py \
  runtime_confirmation_policy.py; do
  cp -al "${project}/external/STU-Net/pancancer/${source}" "${context}/external/STU-Net/pancancer/${source}"
done
cp -al \
  "${project}/external/STU-Net/nnUNet-2.2/nnunetv2/training/nnUNetTrainer/STUNetTrainer.py" \
  "${context}/external/STU-Net/nnUNet-2.2/nnunetv2/training/nnUNetTrainer/STUNetTrainer.py"

model_source="${project}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
model_target="${context}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
cp -al "${model_source}/dataset.json" "${model_target}/dataset.json"
cp -al "${model_source}/plans.json" "${model_target}/plans.json"
ln "${model_source}/fold_all/checkpoint_inference_epoch_0160.pth" "${model_target}/fold_all/checkpoint.pth"

cp -al "${project}/external/GF-Screen/docker_submission/deployment_inference.py" "${context}/deployment_inference.py"
cp -al "${project}/external/GF-Screen/docker_submission/predict_ablation.sh" "${context}/predict.sh"

du -sh "${context}"

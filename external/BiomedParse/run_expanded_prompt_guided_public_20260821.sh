#!/usr/bin/env bash
set -euo pipefail

# Public-only, predeclared open-vocabulary proposal audit. BiomedParse masks
# select extra GF Focus crops; they never replace the champion segmentation.
# The final screen can only add disconnected, independently supported small
# components to the immutable best public candidate. Hidden inference is
# intentionally absent from this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
inputs="${root}/public_a160_240_inputs"
checkpoint="${gf}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
base="${gf}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
output="${root}/public_sweeps/expanded_prompt_guided_20260821"

exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "expanded-prompt public audit is already active" >&2
  exit 1
fi
mkdir -p "${output}"

[[ $(find "${inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]] || {
  echo "expected 50 prepared public inputs" >&2
  exit 1
}
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 immutable base masks" >&2
  exit 1
}

infer_bank() {
  local gpu="$1" bank="$2" first_prompt="$3" second_prompt="$4"
  local npz="${output}/${bank}_npz"
  local nifti="${output}/${bank}_e050_nifti"
  mkdir -p "${npz}" "${nifti}"
  if [[ $(find "${npz}" -maxdepth 1 -name '*.npz' | wc -l) -ne 50 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz}" \
      --prompt "${first_prompt}" --prompt "${second_prompt}" --device cuda \
      --slice-batch-size 8 --existence-threshold 0.50 --continue-prediction \
      > "${output}/${bank}.inference.log" 2>&1
  fi
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
    --images "${images}" --predictions "${npz}" --output "${nifti}" \
    --segmentation-key segs_e050 --continue-prediction \
    > "${output}/${bank}.convert.log" 2>&1
  [[ $(find "${nifti}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
}

# The two banks are fixed before labels are read. They target malignant disease
# identity and morphology respectively and are evaluated as complete banks,
# avoiding a post-hoc search across individual prompt synonyms.
infer_bank 0 malignant_metastasis "malignant lesion" "metastasis" &
bank0=$!
infer_bank 1 cancer_mass "cancer" "abnormal mass" &
bank1=$!
wait "${bank0}"
wait "${bank1}"

guided_bank() {
  local gpu="$1" bank="$2"
  local proposal="${output}/${bank}_e050_nifti"
  local destination="${output}/gf_guided_${bank}_v500"
  local log="${output}/gf_guided_${bank}_v500.inference.log"
  mkdir -p "${destination}"
  if [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 50 ]]; then
    return 0
  fi
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${destination}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_threshold 0.50 \
      --proposal_mask_path "${proposal}" --proposal_min_voxels 500 \
      --focus_batch_size 16 --workers 2 --skip_existing > "${log}" 2>&1; then
    echo "Focus batch 16 failed; resuming at batch 8" >> "${log}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_threshold 0.50 \
        --proposal_mask_path "${proposal}" --proposal_min_voxels 500 \
        --focus_batch_size 8 --workers 2 --skip_existing >> "${log}" 2>&1; then
      echo "Focus batch 8 failed; resuming at batch 2" >> "${log}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
        --test_data_path "${images}" --save_prediction_path "${destination}" \
        --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
        --glance_threshold 0.50 --lesion_threshold 0.50 \
        --proposal_mask_path "${proposal}" --proposal_min_voxels 500 \
        --focus_batch_size 2 --workers 2 --skip_existing >> "${log}" 2>&1
    fi
  fi
  [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
}

guided_bank 0 malignant_metastasis & guided0=$!
guided_bank 1 cancer_mass & guided1=$!
wait "${guided0}"
wait "${guided1}"

screen_bank() {
  local bank="$1"
  "${python_bin}" -u "${gf}/screen_disconnected_guided_additions.py" screen \
    --base "${base}" \
    --candidate "${output}/gf_guided_${bank}_v500" \
    --support "${output}/${bank}_e050_nifti" --gt "${labels}" \
    --output "${output}/screen_${bank}" --workers 8 \
    > "${output}/screen_${bank}.log" 2>&1
}

screen_bank malignant_metastasis & screen0=$!
screen_bank cancer_mass & screen1=$!
wait "${screen0}"
wait "${screen1}"

for bank in malignant_metastasis cancer_mass; do
  echo "${bank}"
  sed -n '1,4p' "${output}/screen_${bank}/summary.tsv"
done
printf '%s\n' \
  'PUBLIC-ONLY AUDIT COMPLETE: no hidden masks or submission archive were generated.' \
  > "${output}/PUBLIC_ONLY_COMPLETE.txt"

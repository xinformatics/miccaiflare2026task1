#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
checkpoint_root="checkpoint_interpolations/a055_to_bbox_ep280"
root="bbox_beta_search/a055_to_bbox_ep280"
evaluator="../../handoff_bundle/code/evaluate.py"
packager="../../nnunet/package_codabench.py"

run_inference() {
  local gpu="$1"
  local beta_tag="$2"
  local split="$3"
  local images="$4"
  local checkpoint_tag
  if [[ "${beta_tag}" == "0875" ]]; then
    checkpoint_tag="0p875"
  elif [[ "${beta_tag}" == "1125" ]]; then
    checkpoint_tag="1p125"
  else
    echo "Unsupported beta tag: ${beta_tag}" >&2
    return 2
  fi
  local checkpoint="${checkpoint_root}/wise_focus_alpha${checkpoint_tag}.pt"
  local output="${root}/b${beta_tag}/${split}_raw"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 \
    > "${root}/b${beta_tag}/${split}_inference.log" 2>&1
}

filter_split() {
  local beta_tag="$1"
  local split="$2"
  mkdir -p "${root}/b${beta_tag}/${split}_mm1000"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${root}/b${beta_tag}/${split}_raw" --fusion a --min-mm3 1000 \
    --output "${root}/b${beta_tag}/${split}_mm1000" \
    > "${root}/b${beta_tag}/${split}_filter.log" 2>&1
}

mkdir -p "${root}/b0875" "${root}/b1125" ../../submissions
test -f "${checkpoint_root}/wise_focus_alpha0p875.pt"
test -f "${checkpoint_root}/wise_focus_alpha1p125.pt"

# Hidden inference first so both Codabench candidates become available at the
# same time. Public inference follows on the same fixed protocol.
run_inference 0 0875 hidden "${hidden_images}" & pid0=$!
run_inference 1 1125 hidden "${hidden_images}" & pid1=$!
wait "${pid0}"
wait "${pid1}"

run_inference 0 0875 public "${public_images}" & pid0=$!
run_inference 1 1125 public "${public_images}" & pid1=$!
wait "${pid0}"
wait "${pid1}"

for beta_tag in 0875 1125; do
  public_count=$(find "${root}/b${beta_tag}/public_raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
  hidden_count=$(find "${root}/b${beta_tag}/hidden_raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${public_count}" -eq 50 ]]
  [[ "${hidden_count}" -eq 100 ]]
done

for beta_tag in 0875 1125; do
  filter_split "${beta_tag}" public &
  filter_split "${beta_tag}" hidden &
done
wait

for beta_tag in 0875 1125; do
  "${python_bin}" -u "${evaluator}" \
    --pred "${root}/b${beta_tag}/public_mm1000" --gt "${public_labels}" \
    --csv "${root}/b${beta_tag}/public_metrics.csv" \
    > "${root}/b${beta_tag}/public_score.log" 2>&1 &
  submission="../../submissions/gf_a055_bbox_ep280_beta${beta_tag}_mm1000.zip"
  "${python_bin}" -u "${packager}" \
    --images "${hidden_images}" \
    --pred-dir "${root}/b${beta_tag}/hidden_mm1000" \
    --output "${submission}" --expected-count 100 \
    > "${root}/b${beta_tag}/package.log" 2>&1 &
done
wait

for beta_tag in 0875 1125; do
  submission="../../submissions/gf_a055_bbox_ep280_beta${beta_tag}_mm1000.zip"
  unzip -t "${submission}" > "${root}/b${beta_tag}/zip_test.log"
  sha256sum "${submission}" > "${submission}.sha256"
  count=$(unzip -Z1 "${submission}" | awk '/\.nii\.gz$/ {n++} END {print n+0}')
  nested=$(unzip -Z1 "${submission}" | awk '/\.nii\.gz$/ && /\// {n++} END {print n+0}')
  [[ "${count}" -eq 100 ]]
  [[ "${nested}" -eq 0 ]]
  echo "beta=${beta_tag} nifti=${count} nested=${nested} submission=${submission}"
done

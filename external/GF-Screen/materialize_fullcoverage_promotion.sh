#!/usr/bin/env bash
set -euo pipefail

# Run only at a serialized GPU boundary. Most all-crop configurations will not
# clear promotion; in that common case this exits without touching the GPUs.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
root="${OUTPUT_ROOT:-public_sweeps/full_coverage_focus_20260820}"
pending="${root}/promotion_pending.tsv"
[[ -s "${pending}" ]] || { echo "No full-coverage promotion pending."; exit 0; }
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
IFS=$'\t' read -r best_gate best_focus best_min best_dsc best_nsd best_combined < "${pending}"
candidate_tag="fullcoverage_l${best_focus/./p}_mm${best_min}"
hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" ../../submissions
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.0 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  [[ "${shard}" -eq 0 ]] && pid0=$! || pid1=$!
done
wait "${pid0}"
wait "${pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 100 ]] || { echo "expected 100 hidden masks, found ${count}" >&2; exit 1; }
if [[ "${best_min}" -gt 0 ]]; then
  "${python_bin}" -u fuse_and_filter_masks.py --pred-a "${hidden_root}/raw" \
    --fusion a --min-mm3 "${best_min}" --output "${hidden_root}/filtered" \
    > "${hidden_root}/filter.log" 2>&1
else
  cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
fi
submission="../../submissions/gf_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py --images "${hidden_images}" \
  --pred-dir "${hidden_root}/filtered" --output "${submission}" --expected-count 100 \
  > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line:-$(cat "${pending}")}" > "${hidden_root}/selected_public_configuration.tsv"
date --iso-8601=seconds > "${root}/PROMOTION_MATERIALIZED"
echo "Full-coverage submission ready: ${submission}"

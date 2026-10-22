#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
root="${workspace}/external/BiomedParse"
gf="${workspace}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
hidden_images="${workspace}/validation/Validation-Hidden-Images"
checkpoint="${gf}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
anchor="${gf}/predictions/gf_a055_bbox_pu_best_ep280/hidden_mm1000"
biomed="${root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation"
audit="${root}/public_sweeps/component_confidence_rescue_20260820"
config="${audit}/focus075_tumor_mean_e010_mm1000_10000.json"
selection="${audit}/selection.json"
output="${audit}/hidden_focus075_tumor_mean_e010_mm1000_10000"
submission="${workspace}/submissions/gf_prompt_or_e050_focus075_tumor_mean_e010_mm1000_10000.zip"

[[ $(awk -F'"' '/"decision":/ {print $4}' "${selection}") == "MATERIALIZE_SECONDARY" ]] || {
  echo "confidence-rescue selection is not eligible" >&2
  exit 1
}
[[ $(find "${anchor}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden anchor masks" >&2
  exit 1
}
[[ $(find "${biomed}/prompt_or" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-OR masks" >&2
  exit 1
}
[[ $(find "${biomed}/tumor_npz" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden Tumor NPZ files" >&2
  exit 1
}

mkdir -p "${output}"/{shard0,shard1,raw075,filtered075,predictions}
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${output}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.75 \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${output}/shard${shard}.log" 2>&1 &
  [[ "${shard}" -eq 0 ]] && pid0=$! || pid1=$!
done
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
for shard in 0 1; do
  cp "${output}/shard${shard}/lesion_0.75/"*.nii.gz "${output}/raw075/"
done
[[ $(find "${output}/raw075" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden Focus-0.75 masks" >&2
  exit 1
}
"${python_bin}" -u "${gf}/fuse_and_filter_masks.py" \
  --pred-a "${output}/raw075" --fusion a --min-mm3 1000 \
  --workers 4 --output "${output}/filtered075" > "${output}/filter075.log" 2>&1
"${python_bin}" -u "${root}/component_confidence_rescue.py" materialize \
  --base "${anchor}" --prompt-support "${biomed}/prompt_or" \
  --high-focus-support "${output}/filtered075" --tumor-npz "${biomed}/tumor_npz" \
  --config "${config}" --output "${output}/predictions" \
  > "${output}/materialize.log" 2>&1
[[ $(find "${output}/predictions" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 confidence-rescue predictions" >&2
  exit 1
}
"${python_bin}" -u "${workspace}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/predictions" \
  --output "${submission}" --expected-count 100 > "${output}/package.log" 2>&1
unzip -t "${submission}" > "${output}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
echo "Confidence-rescue submission ready: ${submission}"

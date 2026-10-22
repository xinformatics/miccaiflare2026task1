#!/usr/bin/env bash
set -euo pipefail

root="/mnt/nvme2-data/shashank/MICCAI26"
code="${root}/external/flare23-aladdin5"
gf="${root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train_proposals="${code}/train_unlabel_roi_tumor_conf_20260829"
healthy_images="${code}/healthy_no_lesion_flat_20260829"
healthy_proposals="${code}/healthy_no_lesion_roi_tumor_conf_20260829"
training_root="${code}/refiner_training_20260829"
patches="${training_root}/patches_flare23_min1000"
runs="${training_root}/runs"
mkdir -p "${healthy_proposals}" "${training_root}" "${runs}"

while [[ ! -f "${train_proposals}/VALIDATION_COMPLETE.json" ]]; do
  sleep 20
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ ! -f "${healthy_proposals}/VALIDATION_COMPLETE.json" ]]; then
  "${python_bin}" "${code}/run_tumor_fine_safe.py" \
  --images "${healthy_images}" \
  --output "${healthy_proposals}" \
  --checkpoint "${root}/flare23model/model_fine.pt" \
  --roi-checkpoint "${root}/flare23model/model_roi.pt" \
  --organ-checkpoint "${root}/flare23model/model_fine2.pt" \
  --omit-organ-inference --gpu 0 --role training \
  --shard-index 0 --shard-count 2 \
  --terminal "${healthy_proposals}/TERMINAL_SHARD0.json" \
  >"${healthy_proposals}/shard0.log" 2>&1 &
  healthy_pid0=$!

  "${python_bin}" "${code}/run_tumor_fine_safe.py" \
  --images "${healthy_images}" \
  --output "${healthy_proposals}" \
  --checkpoint "${root}/flare23model/model_fine.pt" \
  --roi-checkpoint "${root}/flare23model/model_roi.pt" \
  --organ-checkpoint "${root}/flare23model/model_fine2.pt" \
  --omit-organ-inference --gpu 1 --role training \
  --shard-index 1 --shard-count 2 \
  --terminal "${healthy_proposals}/TERMINAL_SHARD1.json" \
  >"${healthy_proposals}/shard1.log" 2>&1 &
  healthy_pid1=$!
  healthy_status=0
  wait "${healthy_pid0}" || healthy_status=$?
  wait "${healthy_pid1}" || healthy_status=$?
  if [[ "${healthy_status}" -ne 0 ]]; then
    exit "${healthy_status}"
  fi

  "${python_bin}" "${code}/validate_train_unlabel_flare23.py" \
  --images "${healthy_images}" \
  --predictions "${healthy_proposals}" \
  --terminal "${healthy_proposals}/TERMINAL_SHARD0.json" \
  --terminal "${healthy_proposals}/TERMINAL_SHARD1.json" \
  --expected 172 \
  --summary "${healthy_proposals}/VALIDATION_COMPLETE.json" \
    >"${healthy_proposals}/validation.log" 2>&1
fi

if [[ ! -f "${training_root}/MANIFEST_COMPLETE.json" ]]; then
  "${python_bin}" "${code}/build_refiner_training_manifest.py" \
    --exact-manifest "${gf}/analysis/fpr_training_mining_707_20260822/manifest.json" \
    --exact-proposals "${code}/exact707_roi_tumor_conf_20260829" \
    --pseudo-manifest "${gf}/public_sweeps/gf_train_unlabel_cbg_direct_20260825/balanced/manifest.json" \
    --pseudo-proposals "${train_proposals}" \
    --pseudo-validation "${train_proposals}/VALIDATION_COMPLETE.json" \
    --healthy-images "${healthy_images}" \
    --healthy-proposals "${healthy_proposals}" \
    --healthy-validation "${healthy_proposals}/VALIDATION_COMPLETE.json" \
    --minimum-mm3 1000 \
    --workers 16 \
    --output "${training_root}" \
    >"${training_root}/manifest.log" 2>&1
fi

if [[ ! -f "${patches}/summary.json" ]]; then
  "${python_bin}" "${gf}/prepare_candidate_refiner_patches.py" \
    --manifest "${training_root}/manifest.json" \
    --proposal "flare23=${training_root}/proposals" \
    --output "${patches}" \
    --shape 64,64,64 \
    --spacing 1.5,1.5,1.5 \
    --min-mm3 1000 \
    --max-mm3 1000000000000 \
    --max-negatives-per-case 16 \
    --min-negatives-per-source 0 \
    --bbox-margin-mm 24 \
    --tile-stride-mm 72 \
    --max-negative-tiles 1 \
    --component-policy source_separate \
    --compact-source-components \
    --workers 4 \
    --resume \
    >"${training_root}/patches.log" 2>&1
fi

train_fold() {
  local gpu="$1"
  local fold="$2"
  local output="${runs}/fold${fold}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${gf}/train_candidate_refiner.py" \
    --index "${patches}/index.csv" \
    --output "${output}" \
    --folds 5 --fold "${fold}" \
    --epochs 30 --batch-size 8 --validation-batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
    --sampling-temperature 1.0 --samples-per-epoch 4096 \
    --prompt-masked-classifier \
    --classification-negative-weight 1.0 \
    --negative-segmentation-weight 0.1 \
    --segmentation-threshold-max 0.90 \
    --validation-interval 5 \
    --seed 20260829 \
    --wandb-project pancancer-ct \
    --wandb-name "flare23-component-pu-fold${fold}" \
    --resume \
    >"${output}/train.log" 2>&1
}

(
  train_fold 0 0
  train_fold 0 2
  train_fold 0 4
) &
train_pid0=$!
(
  train_fold 1 1
  train_fold 1 3
) &
train_pid1=$!
train_status=0
wait "${train_pid0}" || train_status=$?
wait "${train_pid1}" || train_status=$?
if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi

"${python_bin}" "${code}/summarize_refiner_training.py" \
  --runs "${runs}" --folds 5 \
  --manifest "${training_root}/manifest.json" \
  --patch-summary "${patches}/summary.json" \
  --output "${training_root}/TRAINING_COMPLETE.json" \
  >"${training_root}/summary.log" 2>&1

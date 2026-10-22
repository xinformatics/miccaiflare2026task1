#!/usr/bin/env bash
set -euo pipefail

project="/mnt/nvme2-data/shashank/MICCAI26"
root="${project}/external/FLARE25-Task1-winner"
output="${root}/exact707_primary_audit_20260828/mask_router_cv"
mkdir -p "${output}"
exec >"${output}/run.log" 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/m348911/miniconda3/envs/radoncAI/bin/python "${root}/train_exact707_mask_router.py" \
  --manifest "${project}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/manifest.json" \
  --cases "${root}/exact707_primary_audit_20260828/positive_only_comparison/cases.csv" \
  --gf "${project}/external/GF-Screen/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050" \
  --flare "${root}/exact707_primary_audit_20260828/predictions" \
  --policy "${root}/ROUTER_AUDIT_POLICY_20260828.json" \
  --output "${output}" \
  --workers 16

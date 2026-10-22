#!/usr/bin/env bash
set -euo pipefail

# Matched full-volume transfer baseline wrapper. It invokes the same exact-210
# calibration implementation, proposal cache, threshold grid, and deployment
# modes used by union-native OOF.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
transfer="${root}/analysis/candidate_refiner_union_native_training_20260823/transfer_fivefold"
output="${root}/analysis/candidate_refiner_union_transfer_fullvolume_oof_20260823"
export UNION_OOF_ROLE=transfer
export UNION_OOF_TRAIN_ROOT="${root}/analysis/candidate_refiner_union_native_training_20260823"
export UNION_OOF_FIVEFOLD_ROOT="${transfer}"
export UNION_OOF_FIVEFOLD_TERMINAL="${transfer}/UNION_TRANSFER_FIVEFOLD_TERMINAL.json"
export UNION_OOF_SELECTION="${transfer}/checkpoint_selection.json"
export UNION_OOF_PATCH_OOF="${transfer}/five_fold_oof_summary.json"
export UNION_OOF_OUTPUT="${output}"
export UNION_OOF_TERMINAL="${output}/UNION_TRANSFER_OOF_TERMINAL.json"
export UNION_OOF_RUN_PREFIX="candidate_refiner_v2_bbox24_b32_v2det_fold"
export UNION_OOF_VARIANT_PREFIX=tr
export UNION_OOF_LOCK="${root}/runs/run_candidate_refiner_union_transfer_fullvolume_oof_20260823.lock"
exec bash "${root}/run_candidate_refiner_union_native_fullvolume_oof_20260823.sh"

#!/usr/bin/env bash
set -euo pipefail

# Fixed-public wrapper for the matched transfer calibration. It uses the same
# implementation, proposal cache, anchor, and paired gates as union-native.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oof="${root}/analysis/candidate_refiner_union_transfer_fullvolume_oof_20260823"
train="${root}/analysis/candidate_refiner_union_native_training_20260823/transfer_fivefold"
output="${root}/public_sweeps/candidate_refiner_union_transfer_20260823"
export UNION_PUBLIC_ROLE=transfer
export UNION_PUBLIC_OOF_ROOT="${oof}"
export UNION_PUBLIC_OOF_TERMINAL="${oof}/UNION_TRANSFER_OOF_TERMINAL.json"
export UNION_PUBLIC_EXPECTED_OOF_DECISION=BASELINE_READY
export UNION_PUBLIC_CALIBRATION="${oof}/deployment_calibration.json"
export UNION_PUBLIC_TRAIN_ROOT="${train}"
export UNION_PUBLIC_SELECTION="${train}/checkpoint_selection.json"
export UNION_PUBLIC_OUTPUT="${output}"
export UNION_PUBLIC_TERMINAL="${output}/UNION_TRANSFER_PUBLIC_TERMINAL.json"
export UNION_PUBLIC_RUN_PREFIX="candidate_refiner_v2_bbox24_b32_v2det_fold"
export UNION_PUBLIC_VARIANT_PREFIX=tr
export UNION_PUBLIC_LOCK="${root}/runs/run_candidate_refiner_union_transfer_public_20260823.lock"
exec bash "${root}/run_candidate_refiner_union_native_public_20260823.sh"

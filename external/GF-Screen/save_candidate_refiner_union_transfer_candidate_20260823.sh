#!/usr/bin/env bash
set -euo pipefail

# Exact-100 save-only wrapper for a strictly promoted matched-transfer model.
# The generic consumer performs all proposal, inference, ZIP, and hash audits.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
public="${root}/public_sweeps/candidate_refiner_union_transfer_20260823"
train="${root}/analysis/candidate_refiner_union_native_training_20260823/transfer_fivefold"
oof="${root}/analysis/candidate_refiner_union_transfer_fullvolume_oof_20260823"
saved="${root}/public_sweeps/candidate_refiner_union_transfer_saved_20260823"
hidden="${root}/hidden_predictions/candidate_refiner_union_transfer_20260823"
export UNION_SAVE_ROLE=transfer
export UNION_SAVE_PUBLIC_ROOT="${public}"
export UNION_SAVE_PUBLIC_TERMINAL="${public}/UNION_TRANSFER_PUBLIC_TERMINAL.json"
export UNION_SAVE_SELECTION="${train}/checkpoint_selection.json"
export UNION_SAVE_CALIBRATION="${oof}/deployment_calibration.json"
export UNION_SAVE_ROOT="${saved}"
export UNION_SAVE_HIDDEN_ROOT="${hidden}"
export UNION_SAVE_TERMINAL="${saved}/UNION_TRANSFER_SAVE_ONLY_TERMINAL.json"
export UNION_SAVE_RUN_PREFIX="candidate_refiner_v2_bbox24_b32_v2det_fold"
export UNION_SAVE_ARCHIVE_PREFIX=gf_reftr
export UNION_SAVE_FAMILY=candidate_refiner_union_transfer
export UNION_SAVE_MANIFEST_NAME=saved_union_transfer_submission.tsv
export UNION_SAVE_LOCK="${root}/runs/save_candidate_refiner_union_transfer_candidate_20260823.lock"
exec bash "${root}/save_candidate_refiner_union_native_candidate_20260823.sh"

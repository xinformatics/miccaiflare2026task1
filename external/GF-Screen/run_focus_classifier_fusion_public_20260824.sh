#!/usr/bin/env bash
set -euo pipefail

# One fixed 50-case public comparison after exact707 patch and full-volume OOF.
# This controller cannot read hidden data, package a ZIP, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="${1:-}"
case "${variant}" in
  frozen)
    focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
    fusion="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
    focus_oof="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
    output="${root}/public_sweeps/focus_classifier_fusion_frozen_20260824"
    prefix=fff707
    ;;
  discriminative)
    focus_train="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
    fusion="${root}/analysis/focus_classifier_fusion_disc01_20260824/DECISION.json"
    focus_oof="${root}/analysis/focus_classifier_fusion_disc01_fullvolume_oof_20260824"
    output="${root}/public_sweeps/focus_classifier_fusion_disc01_20260824"
    prefix=ffd707
    ;;
  *) echo "usage: $0 frozen|discriminative" >&2; exit 2 ;;
esac

export WEAK707_PUBLIC_QUALIFICATION="${focus_oof}/FOCUS_OOF_TERMINAL.json"
export WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE
export WEAK707_PUBLIC_OOF_ROOT="${focus_oof}"
export WEAK707_PUBLIC_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_PUBLIC_CONTOUR_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_PUBLIC_CLASSIFIER_FUSION="${fusion}"
export WEAK707_PUBLIC_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt
export WEAK707_PUBLIC_OUTPUT="${output}"
export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
export WEAK707_PUBLIC_TAG_PREFIX="${prefix}"
export WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_classifier_fusion_${variant}_public_20260824.lock"

exec bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh"

#!/usr/bin/env bash
set -euo pipefail

# Save-only exact-100 hidden consumer for a strictly promoted classifier fusion.
# It validates and hashes the ZIP but never uploads it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="${1:-}"
case "${variant}" in
  frozen)
    focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
    fusion="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
    focus_oof="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
    public_root="${root}/public_sweeps/focus_classifier_fusion_frozen_20260824"
    saved_root="${root}/public_sweeps/focus_classifier_fusion_frozen_saved_20260824"
    hidden_root="${root}/hidden_predictions/focus_classifier_fusion_frozen_20260824"
    prefix=gf_fff7
    family=focus_classifier_fusion_frozen
    ;;
  discriminative)
    focus_train="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
    fusion="${root}/analysis/focus_classifier_fusion_disc01_20260824/DECISION.json"
    focus_oof="${root}/analysis/focus_classifier_fusion_disc01_fullvolume_oof_20260824"
    public_root="${root}/public_sweeps/focus_classifier_fusion_disc01_20260824"
    saved_root="${root}/public_sweeps/focus_classifier_fusion_disc01_saved_20260824"
    hidden_root="${root}/hidden_predictions/focus_classifier_fusion_disc01_20260824"
    prefix=gf_ffd7
    family=focus_classifier_fusion_discriminative
    ;;
  *) echo "usage: $0 frozen|discriminative" >&2; exit 2 ;;
esac

export WEAK707_SAVE_PUBLIC_ROOT="${public_root}"
export WEAK707_SAVE_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_SAVE_CONTOUR_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_SAVE_CLASSIFIER_FUSION="${fusion}"
export WEAK707_SAVE_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt
export WEAK707_SAVE_OUTPUT="${saved_root}"
export WEAK707_SAVE_HIDDEN_ROOT="${hidden_root}"
export WEAK707_SAVE_PROPOSAL_CACHE_ROOT="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"
export WEAK707_SAVE_TAG_PREFIX="${prefix}"
export WEAK707_SAVE_FAMILY="${family}"
export WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0
export WEAK707_SAVE_LOCK="${root}/runs/save_focus_classifier_fusion_${variant}_candidate_20260824.lock"

exec bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh"

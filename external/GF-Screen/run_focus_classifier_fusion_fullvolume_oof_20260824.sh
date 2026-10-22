#!/usr/bin/env bash
set -euo pipefail

# Exact707 full-volume OOF for a locally qualified B32/Focus confidence stack.
# No public or hidden data are read. The generic controller recomputes frozen
# classification thresholds on held-out full volumes before any public use.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="${1:-}"
b32_train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
b32_oof="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"

case "${variant}" in
  frozen)
    focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
    fusion="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
    output="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
    candidate_name=focus_fusion_frozen_cell707_oof
    ;;
  discriminative)
    focus_train="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
    fusion="${root}/analysis/focus_classifier_fusion_disc01_20260824/DECISION.json"
    output="${root}/analysis/focus_classifier_fusion_disc01_fullvolume_oof_20260824"
    candidate_name=focus_fusion_disc01_cell707_oof
    ;;
  *) echo "usage: $0 frozen|discriminative" >&2; exit 2 ;;
esac

export FOCUS_TRAIN_ROOT="${focus_train}"
export FOCUS_TRAIN_TERMINAL="${focus_train}/FIVEFOLD_TERMINAL.json"
export FOCUS_PATCH_OOF="${focus_train}/focus_patch_oof.json"
export FOCUS_OOF_CLASSIFIER_FUSION="${fusion}"
export FOCUS_OOF_PRIMARY_TRAIN="${b32_train}"
export FOCUS_OOF_BASE_ROOT="${b32_oof}"
export FOCUS_OOF_BASE_COMPLETE="${b32_oof}/FULL_VOLUME_OOF_COMPLETE.json"
export FOCUS_OOF_BASE_CALIBRATION="${b32_oof}/deployment_calibration.json"
export FOCUS_OOF_PRIMARY_PATCH_OOF="${b32_train}/five_fold_oof_summary.json"
export FOCUS_OOF_SELECTION="${b32_train}/checkpoint_selection.json"
export FOCUS_OOF_CLASSIFIER_CHECKPOINT_NAME=checkpoint_best_classification.pt
export FOCUS_OOF_MANIFEST="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
export FOCUS_OOF_OUTPUT="${output}"
export FOCUS_OOF_TERMINAL="${output}/FOCUS_OOF_TERMINAL.json"
export FOCUS_OOF_LOCK="${root}/runs/run_focus_classifier_fusion_${variant}_fullvolume_20260824.lock"
export FOCUS_OOF_CANDIDATE_NAME="${candidate_name}"
export FOCUS_OOF_SMALL_RUN_PREFIX="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold"
export FOCUS_OOF_BASE_WORK_NAME=work_bbox24
export FOCUS_OOF_BASE_PROPOSAL_REL=proposals/epoch105
export FOCUS_OOF_EXPECTED_SOURCES=epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7
export FOCUS_OOF_EXPECTED_CASES=707

exec bash "${root}/run_focus_candidate_refiner_fullvolume_oof_20260823.sh"

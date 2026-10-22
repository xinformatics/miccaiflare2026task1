#!/usr/bin/env bash
set -euo pipefail

# Exact-707 local-only full-volume evaluation of the dual-window Focus contour
# model paired with the matched B32 candidate classifier. This wrapper cannot
# read public/hidden data, package a submission, or upload anything. The generic
# controller first requires the five-fold patch gate to promote Focus and then
# compares it against the completed B32 exact-707 full-volume OOF deployment.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
b32_train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
b32_oof="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
output="${root}/analysis/focus_dualwindow_cell707_fullvolume_oof_20260823"

export FOCUS_TRAIN_ROOT="${focus_train}"
export FOCUS_TRAIN_TERMINAL="${focus_train}/FIVEFOLD_TERMINAL.json"
export FOCUS_PATCH_OOF="${focus_train}/focus_patch_oof.json"
export FOCUS_OOF_PRIMARY_TRAIN="${b32_train}"
export FOCUS_OOF_BASE_ROOT="${b32_oof}"
export FOCUS_OOF_BASE_COMPLETE="${b32_oof}/FULL_VOLUME_OOF_COMPLETE.json"
export FOCUS_OOF_BASE_CALIBRATION="${b32_oof}/deployment_calibration.json"
export FOCUS_OOF_PRIMARY_PATCH_OOF="${b32_train}/five_fold_oof_summary.json"
export FOCUS_OOF_SELECTION="${b32_train}/checkpoint_selection.json"
export FOCUS_OOF_MANIFEST="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
export FOCUS_OOF_OUTPUT="${output}"
export FOCUS_OOF_TERMINAL="${output}/FOCUS_OOF_TERMINAL.json"
export FOCUS_OOF_LOCK="${root}/runs/run_focus_dualwindow_weak707_fullvolume_oof_20260823.lock"
export FOCUS_OOF_CANDIDATE_NAME="focus_dualwindow_cell707_hybrid_oof"
export FOCUS_OOF_SMALL_RUN_PREFIX="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold"
export FOCUS_OOF_BASE_WORK_NAME="work_bbox24"
export FOCUS_OOF_BASE_PROPOSAL_REL="proposals/epoch105"
export FOCUS_OOF_EXPECTED_SOURCES="epoch105,currentgf,autopet_weak,lndb_weak,lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7"
export FOCUS_OOF_EXPECTED_CASES=707

exec bash "${root}/run_focus_candidate_refiner_fullvolume_oof_20260823.sh"

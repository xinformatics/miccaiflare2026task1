#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden consumer for a Focus hybrid that passes exact-707 OOF and
# one fixed public comparison. The delegated controller validates an exact flat
# 100-mask ZIP and SHA-256 but contains no upload operation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
focus_oof="${root}/analysis/focus_dualwindow_cell707_fullvolume_oof_20260823"
public_root="${root}/public_sweeps/focus_dualwindow_cell707_20260823"
saved_root="${root}/public_sweeps/focus_dualwindow_cell707_saved_20260823"
hidden_root="${root}/hidden_predictions/focus_dualwindow_cell707_20260823"

export WEAK707_SAVE_PUBLIC_ROOT="${public_root}"
export WEAK707_SAVE_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_SAVE_CONTOUR_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_SAVE_OUTPUT="${saved_root}"
export WEAK707_SAVE_HIDDEN_ROOT="${hidden_root}"
export WEAK707_SAVE_TAG_PREFIX=gf_refdwc
export WEAK707_SAVE_LOCK="${root}/runs/save_focus_dualwindow_weak707_candidate_20260823.lock"

exec bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh"

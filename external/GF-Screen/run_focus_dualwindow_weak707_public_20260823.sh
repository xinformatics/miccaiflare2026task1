#!/usr/bin/env bash
set -euo pipefail

# One fixed public comparison for a locally promoted exact-707 dual-window
# Focus hybrid. The delegated controller has no hidden/package/upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
focus_oof="${root}/analysis/focus_dualwindow_cell707_fullvolume_oof_20260823"
public_output="${root}/public_sweeps/focus_dualwindow_cell707_20260823"

export WEAK707_PUBLIC_QUALIFICATION="${focus_oof}/FOCUS_OOF_TERMINAL.json"
export WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE
export WEAK707_PUBLIC_OOF_ROOT="${focus_oof}"
export WEAK707_PUBLIC_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_PUBLIC_CONTOUR_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_PUBLIC_OUTPUT="${public_output}"
export WEAK707_PUBLIC_TAG_PREFIX=dwc707
export WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_dualwindow_weak707_public_20260823.lock"

exec bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh"

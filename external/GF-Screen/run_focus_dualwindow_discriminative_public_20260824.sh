#!/usr/bin/env bash
set -euo pipefail

# Fixed-public consumer for a locally promoted discriminative-LR Focus model.
# The delegated controller has no hidden/package/upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_disc01_fivefold_20260824"
focus_oof="${root}/analysis/focus_dualwindow_disc01_fullvolume_oof_20260824"
public_output="${root}/public_sweeps/focus_dualwindow_disc01_20260824"

export WEAK707_PUBLIC_QUALIFICATION="${focus_oof}/FOCUS_OOF_TERMINAL.json"
export WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE
export WEAK707_PUBLIC_OOF_ROOT="${focus_oof}"
export WEAK707_PUBLIC_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_PUBLIC_CONTOUR_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_PUBLIC_OUTPUT="${public_output}"
export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
export WEAK707_PUBLIC_TAG_PREFIX=dlf707
export WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_dualwindow_disc01_public_20260824.lock"

exec bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh"

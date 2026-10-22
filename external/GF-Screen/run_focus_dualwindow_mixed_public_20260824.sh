#!/usr/bin/env bash
set -euo pipefail

# Fixed-public consumer for the mixed B32/Focus source-family contour route.
# It runs only after exact707 full-volume OOF promotion and cannot read hidden
# data, package an archive, or upload anything.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
focus_oof="${root}/analysis/focus_dualwindow_mixed_fullvolume_oof_20260824"
public_output="${root}/public_sweeps/focus_dualwindow_mixed_20260824"

export WEAK707_PUBLIC_QUALIFICATION="${focus_oof}/FOCUS_OOF_TERMINAL.json"
export WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE
export WEAK707_PUBLIC_OOF_ROOT="${focus_oof}"
export WEAK707_PUBLIC_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_PUBLIC_MIXED_FOCUS_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_PUBLIC_MIXED_ROUTE_DECISION="${focus_train}/focus_b32_source_hybrid_decision.json"
export WEAK707_PUBLIC_OUTPUT="${public_output}"
export WEAK707_PUBLIC_TAG_PREFIX=dwm707
export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
export WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_dualwindow_mixed_public_20260824.lock"

exec bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh"

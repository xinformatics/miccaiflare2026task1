#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden consumer for a mixed B32/Focus route that passes exact707
# OOF and fixed-public gates. It validates a flat exact-100 ZIP and never
# uploads it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
focus_oof="${root}/analysis/focus_dualwindow_mixed_fullvolume_oof_20260824"
public_root="${root}/public_sweeps/focus_dualwindow_mixed_20260824"
saved_root="${root}/public_sweeps/focus_dualwindow_mixed_saved_20260824"
hidden_root="${root}/hidden_predictions/focus_dualwindow_mixed_20260824"

export WEAK707_SAVE_PUBLIC_ROOT="${public_root}"
export WEAK707_SAVE_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_SAVE_MIXED_FOCUS_SUMMARY="${focus_train}/focus_patch_oof.json"
export WEAK707_SAVE_MIXED_ROUTE_DECISION="${focus_train}/focus_b32_source_hybrid_decision.json"
export WEAK707_SAVE_OUTPUT="${saved_root}"
export WEAK707_SAVE_HIDDEN_ROOT="${hidden_root}"
export WEAK707_SAVE_PROPOSAL_CACHE_ROOT="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"
export WEAK707_SAVE_TAG_PREFIX=gf_dwm707
export WEAK707_SAVE_FAMILY=focus_dualwindow_mixed
export WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0
export WEAK707_SAVE_LOCK="${root}/runs/save_focus_dualwindow_mixed_candidate_20260824.lock"

exec bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh"

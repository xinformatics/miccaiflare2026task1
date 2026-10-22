#!/usr/bin/env bash
set -euo pipefail

# Fixed-public consumer for epoch-routed Focus. It can run only after both the
# patch-OOF route gate and exact707 routed full-volume OOF promote. No hidden
# inference, packaging, or upload occurs here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
focus_train="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
focus_oof="${root}/analysis/focus_dualwindow_epoch_routed_fullvolume_oof_20260824"
public_output="${root}/public_sweeps/focus_dualwindow_epoch_routed_20260824"

export WEAK707_PUBLIC_QUALIFICATION="${focus_oof}/FOCUS_OOF_TERMINAL.json"
export WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE
export WEAK707_PUBLIC_OOF_ROOT="${focus_oof}"
export WEAK707_PUBLIC_CALIBRATION="${focus_oof}/deployment_calibration.json"
export WEAK707_PUBLIC_FOCUS_ROUTE_SUMMARY="${focus_train}/focus_epoch_routes.json"
export WEAK707_PUBLIC_FOCUS_ROUTE_DECISION="${focus_train}/focus_epoch_routes_decision.json"
export WEAK707_PUBLIC_OUTPUT="${public_output}"
export WEAK707_PUBLIC_TAG_PREFIX=dwr707
export WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_dualwindow_epoch_routed_public_20260824.lock"

exec bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh"

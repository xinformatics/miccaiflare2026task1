#!/usr/bin/env bash
set -euo pipefail

# Exact-210 prompt-Focus OOF wrapper. The generic controller remains the
# authority for fixed threshold/mode calibration and paired promotion.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FOCUS_TRAIN_ROOT="${root}/analysis/focus_prompt_candidate_refiner_fivefold_20260823"
export FOCUS_TRAIN_TERMINAL="${FOCUS_TRAIN_ROOT}/FOCUS_PROMPT_FIVEFOLD_TERMINAL.json"
export FOCUS_PATCH_OOF="${FOCUS_TRAIN_ROOT}/focus_prompt_patch_oof.json"
export FOCUS_OOF_BASE_ROOT="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
export FOCUS_OOF_BASE_COMPLETE="${FOCUS_OOF_BASE_ROOT}/FULL_VOLUME_OOF_COMPLETE.txt"
export FOCUS_OOF_BASE_CALIBRATION="${FOCUS_OOF_BASE_ROOT}/deployment_calibration.json"
export FOCUS_OOF_OUTPUT="${root}/analysis/focus_prompt_candidate_refiner_fullvolume_oof_20260823"
export FOCUS_OOF_TERMINAL="${FOCUS_OOF_OUTPUT}/FOCUS_PROMPT_OOF_TERMINAL.json"
export FOCUS_OOF_LOCK="${root}/runs/run_focus_prompt_candidate_refiner_fullvolume_oof_20260823.lock"
export FOCUS_OOF_CANDIDATE_NAME="focus_prompt_multisource_oof"
exec bash "${root}/run_focus_candidate_refiner_fullvolume_oof_20260823.sh"

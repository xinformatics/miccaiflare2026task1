#!/usr/bin/env bash
set -euo pipefail

# Fixed-public wrapper for the prompt-conditioned Focus candidate. It inherits
# thresholds and deployment mode from exact-210 OOF and performs no public
# search.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FOCUS_PUBLIC_OOF_ROOT="${root}/analysis/focus_prompt_candidate_refiner_fullvolume_oof_20260823"
export FOCUS_PUBLIC_OOF_TERMINAL="${FOCUS_PUBLIC_OOF_ROOT}/FOCUS_PROMPT_OOF_TERMINAL.json"
export FOCUS_PUBLIC_CALIBRATION="${FOCUS_PUBLIC_OOF_ROOT}/deployment_calibration.json"
export FOCUS_PUBLIC_PATCH_OOF="${root}/analysis/focus_prompt_candidate_refiner_fivefold_20260823/focus_prompt_patch_oof.json"
export FOCUS_PUBLIC_PROPOSAL_WORK="${root}/public_sweeps/candidate_refiner_v2_multisource_20260823/work_all_bbox24"
export FOCUS_PUBLIC_PROPOSAL_TERMINAL="${root}/public_sweeps/candidate_refiner_v2_multisource_20260823/PUBLIC_ONLY_COMPLETE.txt"
export FOCUS_PUBLIC_OUTPUT="${root}/public_sweeps/focus_prompt_candidate_refiner_20260823"
export FOCUS_PUBLIC_TERMINAL="${FOCUS_PUBLIC_OUTPUT}/FOCUS_PROMPT_PUBLIC_TERMINAL.json"
export FOCUS_PUBLIC_LOCK="${root}/runs/run_focus_prompt_candidate_refiner_public_20260823.lock"
export FOCUS_PUBLIC_VARIANT_PREFIX="focus_prompt_stand"
exec bash "${root}/run_focus_candidate_refiner_public_20260823.sh"

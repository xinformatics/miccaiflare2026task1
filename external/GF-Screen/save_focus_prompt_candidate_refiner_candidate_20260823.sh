#!/usr/bin/env bash
set -euo pipefail

# Save-only wrapper for a strictly promoted prompt-conditioned Focus candidate.
# This can create and hash an exact-100 archive, but it cannot upload it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FOCUS_SAVE_PUBLIC_ROOT="${root}/public_sweeps/focus_prompt_candidate_refiner_20260823"
export FOCUS_SAVE_PUBLIC_TERMINAL="${FOCUS_SAVE_PUBLIC_ROOT}/FOCUS_PROMPT_PUBLIC_TERMINAL.json"
export FOCUS_SAVE_PATCH_OOF="${root}/analysis/focus_prompt_candidate_refiner_fivefold_20260823/focus_prompt_patch_oof.json"
export FOCUS_SAVE_ROOT="${root}/public_sweeps/focus_prompt_candidate_refiner_saved_20260823"
export FOCUS_SAVE_HIDDEN_ROOT="${root}/hidden_predictions/focus_prompt_candidate_refiner_20260823"
export FOCUS_SAVE_SECOND_PROPOSAL="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
export FOCUS_SAVE_PROPOSAL_DIR="${FOCUS_SAVE_HIDDEN_ROOT}/proposal_union"
export FOCUS_SAVE_PROPOSAL_MARKER="${FOCUS_SAVE_HIDDEN_ROOT}/PROPOSAL_UNION_COMPLETE.json"
export FOCUS_SAVE_PROPOSAL_NAME="union"
export FOCUS_SAVE_TERMINAL="${FOCUS_SAVE_ROOT}/FOCUS_PROMPT_SAVE_ONLY_COMPLETE.txt"
export FOCUS_SAVE_LOCK="${root}/runs/save_focus_prompt_candidate_refiner_candidate_20260823.lock"
export FOCUS_SAVE_ARCHIVE_PREFIX="gf_refpf"
export FOCUS_SAVE_FAMILY="focus_prompt_candidate_refiner"
exec bash "${root}/save_focus_candidate_refiner_candidate_20260823.sh"

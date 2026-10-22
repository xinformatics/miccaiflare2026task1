#!/usr/bin/env bash
set -euo pipefail

# Resume the exact full run after the epoch-100 public/save decision.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_INTERIM_EPOCH=100
export STUNET_PUBLIC_OUTPUT="${root}/public_sweeps/stunetb_e100_ensemble_20260826"
export STUNET_SAVE_OUTPUT="${root}/public_sweeps/stunetb_e100_ensemble_saved_20260826"
export STUNET_ACTIVE_UNITS="miccai-stunetb-training-post-e20-20260825.service miccai-stunetb-training-resume-e100-failure-20260826.service"
export STUNET_RESUME_UNIT=miccai-stunetb-dual-post-e100-20260826
export STUNET_RESUME_LOCK="${root}/runs/resume_training_after_e100_ensemble_20260826.lock"
export STUNET_POST_GATE_SCRIPT="${root}/launch_stunetb_dual_after_e100_20260826.sh"
exec /usr/bin/bash "${root}/resume_training_after_e20_ensemble_20260825.sh"

#!/usr/bin/env bash
set -euo pipefail

# Evaluate the same full STU-Net run immediately at epoch 100. The generic
# controller pauses only after a hash-bound optimizer checkpoint is stable.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export STUNET_INTERIM_EPOCH=100
export STUNET_TRAINING_UNIT=miccai-stunetb-training-post-e20-20260825.service
export STUNET_FAILURE_RESUME_UNIT=miccai-stunetb-training-resume-e100-failure-20260826
export STUNET_INTERIM_OUTPUT="${project_root}/external/STU-Net/pancancer/public_sweeps/stunetb_e100_ensemble_20260826"
exec /usr/bin/bash "${project_root}/external/STU-Net/pancancer/run_interim_e20_public_ensemble_20260825.sh"

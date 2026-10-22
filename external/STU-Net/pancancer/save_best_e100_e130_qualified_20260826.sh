#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_PUBLIC_OUTPUT="${root}/public_sweeps/stunetb_best_e100_e130_20260826"
export STUNET_SAVE_OUTPUT="${root}/public_sweeps/stunetb_matched_e130_saved_20260826"
export STUNET_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_best_e100_e130_20260826"
export STUNET_ALLOWED_EPOCHS=100,130
exec /usr/bin/bash "${root}/save_public_qualified_ensemble_20260825.sh"

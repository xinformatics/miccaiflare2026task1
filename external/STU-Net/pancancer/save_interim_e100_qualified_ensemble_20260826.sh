#!/usr/bin/env bash
set -euo pipefail

# Reuse the closed generic saver for only the epoch-100 decision. It cannot
# access hidden images unless the fixed-public gate has already passed.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_PUBLIC_OUTPUT="${root}/public_sweeps/stunetb_e100_ensemble_20260826"
export STUNET_SAVE_OUTPUT="${root}/public_sweeps/stunetb_e100_ensemble_saved_20260826"
export STUNET_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_e100_ensemble_20260826"
export STUNET_ALLOWED_EPOCHS=100
exec /usr/bin/bash "${root}/save_public_qualified_ensemble_20260825.sh"

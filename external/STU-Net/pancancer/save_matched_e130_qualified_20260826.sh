#!/usr/bin/env bash
set -euo pipefail

# Generate a submission-ready hidden archive only if the matched epoch-130
# public comparison passes both immutable gates. Never upload automatically.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_PUBLIC_OUTPUT="${root}/public_sweeps/stunetb_matched_e130_20260826"
export STUNET_SAVE_OUTPUT="${root}/public_sweeps/stunetb_matched_e130_saved_20260826"
export STUNET_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_matched_e130_20260826"
export STUNET_ALLOWED_EPOCHS=130
exec /usr/bin/bash "${root}/save_public_qualified_ensemble_20260825.sh"

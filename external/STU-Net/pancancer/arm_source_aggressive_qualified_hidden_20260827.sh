#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_EXPECTED_EVALUATION_BRANCH="source_aggressive_e160"
export STUNET_PUBLIC_SELECTION_ROOT="${root}/public_sweeps/stunetb_sourceaggressive_e40_e80_20260827"
export STUNET_EXPORT_CONTRACT_ROOT="${root}/public_sweeps/stunetb_sourceaggressive_export_contract_20260827"
export STUNET_BRANCH_SAVE_OUTPUT="${root}/public_sweeps/stunetb_sourceaggressive_saved_20260827"
export STUNET_BRANCH_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_sourceaggressive_20260827"
export STUNET_BRANCH_MODEL="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceAggressive__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
export STUNET_BRANCH_ARCHIVE_PREFIX="gf_sa"
exec /usr/bin/bash "${root}/arm_complete_hybrid_qualified_hidden_20260826.sh"

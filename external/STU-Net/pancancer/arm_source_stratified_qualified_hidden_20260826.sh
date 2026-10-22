#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_EXPECTED_EVALUATION_BRANCH="source_stratified_e160"
export STUNET_PUBLIC_SELECTION_ROOT="${root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826"
export STUNET_EXPORT_CONTRACT_ROOT="${root}/public_sweeps/stunetb_sourcestrat_export_contract_20260826"
export STUNET_BRANCH_SAVE_OUTPUT="${root}/public_sweeps/stunetb_sourcestrat_saved_20260826"
export STUNET_BRANCH_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_sourcestrat_20260826"
export STUNET_BRANCH_MODEL="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceStratified__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
export STUNET_BRANCH_ARCHIVE_PREFIX="gf_ss"
export STUNET_RUNTIME_GATE_TERMINAL="${project_root}/external/GF-Screen/docker_submission/analysis/runtime_anchor_equivalence_20260827/EQUIVALENCE.json"
exec /usr/bin/bash "${root}/arm_complete_hybrid_qualified_hidden_20260826.sh"

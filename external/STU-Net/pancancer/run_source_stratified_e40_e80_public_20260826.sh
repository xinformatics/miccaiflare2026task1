#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
export STUNET_EVALUATION_BRANCH="source_stratified_e160"
export STUNET_EXPECTED_TRAINER="STUNetTrainerPanCancerComponentBoxE160SourceStratified"
export STUNET_EVALUATION_MODEL="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceStratified__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
export STUNET_TRAINING_TERMINAL="${root}/runs/stunetb_componentbox_e160_sourcesize_f025_cap3_80ep_20260826/TRAINING_TERMINAL.json"
export STUNET_PUBLIC_OUTPUT_ROOT="${root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826"
exec /usr/bin/bash "${root}/run_complete_hybrid_e40_e80_public_20260826.sh"

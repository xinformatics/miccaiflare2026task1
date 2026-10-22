#!/usr/bin/env bash
set -euo pipefail

# Reproducible, CPU-only diagnostic across eight complementary public mask
# families. It is ground-truth informed, writes no NIfTI masks, and has no
# hidden-data or submission path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/analysis/component_proposal_oracle_20260822"
mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "component proposal oracle already active" >&2; exit 1; }
[[ -s "${output}/summary.json" ]] && exit 0

cd "${root}"
"${python_bin}" -u analyze_component_proposal_oracle.py \
  --gt ../../validation/Validation-Public-Labels \
  --candidate champion=public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25 \
  --candidate richfull_l060=public_sweeps/gf_rich16_full_dual_deployed_20260821/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281/deployed_l0p60_mm1000_new2k_s25 \
  --candidate epoch105=public_sweeps/epoch0105/abdomen_g09_l05/predictions \
  --candidate biomed_malignant10k=../BiomedParse/public_sweeps/standalone_malignant_filter_extension_20260821/mm10000 \
  --candidate cross_overlap=public_sweeps/cross_model_component_fusion/overlap_no_rescue \
  --candidate rich_l025=public_sweeps/gf_rich16_comp281_low_thresholds_20260821/l0p25/deployed_mm1000_new2k_s25 \
  --candidate fullcoverage_l075=public_sweeps/full_coverage_focus_20260820/merged/lesion_0.75 \
  --candidate biomed_tumor_e030=../BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e030_nifti \
  --output-dir "${output}" --workers 1 > "${output}/run.log" 2>&1
date -Ins > "${output}/COMPLETE.txt"

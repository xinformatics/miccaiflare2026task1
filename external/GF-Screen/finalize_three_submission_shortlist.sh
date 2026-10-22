#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
"${python_bin}" -u build_three_submission_shortlist.py \
  --project ../.. --output ../../submissions/final_shortlist_20260820 \
  --anchor ../BiomedParse/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv

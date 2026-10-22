#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

python "${SCRIPT_DIR}/prepare_dataset.py"

# The 17,625 native image/mask pairs represent about 3.5 TiB when expanded.
# nnU-Net 2.6.4 writes chunked compressed b2nd arrays, but preprocessing still
# needs generous headroom. Refuse to begin if less than 1 TiB is free.
MIN_FREE_BYTES="${NNUNET_MIN_FREE_BYTES:-1099511627776}"
AVAILABLE_BYTES="$(df --output=avail -B1 "${nnUNet_preprocessed}" | tail -n 1 | tr -d ' ')"
if (( AVAILABLE_BYTES < MIN_FREE_BYTES )); then
  echo "Insufficient free space for this dataset." >&2
  echo "Available: ${AVAILABLE_BYTES} bytes; required safety floor: ${MIN_FREE_BYTES} bytes." >&2
  echo "Override NNUNET_MIN_FREE_BYTES only after estimating the planned output." >&2
  exit 1
fi

# ResEnc L is nnU-Net's recommended default and targets 24 GB VRAM. Eight
# preprocessing workers fit this dataset's unusually high per-case RAM use.
# Override with NNUNET_FINGERPRINT_PROCESSES or NNUNET_PREPROCESS_PROCESSES.
FINGERPRINT_PROCESSES="${NNUNET_FINGERPRINT_PROCESSES:-4}"
PREPROCESS_PROCESSES="${NNUNET_PREPROCESS_PROCESSES:-8}"

nnUNetv2_plan_and_preprocess \
  -d 501 \
  --verify_dataset_integrity \
  -pl nnUNetPlannerResEncL \
  -c 3d_fullres \
  -npfp "${FINGERPRINT_PROCESSES}" \
  -np "${PREPROCESS_PROCESSES}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

DATASET="Dataset501_PanCancer"
PLANS="${nnUNet_preprocessed}/${DATASET}/nnUNetPlans.json"
DATA="${nnUNet_preprocessed}/${DATASET}/nnUNetPlans_3d_fullres_S4D2W32"
LOG_DIR="${NNUNET_PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/nnunet_1epoch.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "One-epoch pipeline started at $(date --iso-8601=seconds)"
echo "Log: ${LOG_FILE}"

if [[ ! -f "${PLANS}" || ! -d "${DATA}" ]]; then
  echo "Transferred ResEnc S6 cache is missing: ${DATA}" >&2
  exit 1
fi

export WANDB_NAME="${WANDB_NAME:-resenc-s6-107m-smoke-1epoch}"
echo "Starting two-GPU fold-all ResEnc S6 training for exactly one epoch."
"${SCRIPT_DIR}/train_2gpu.sh" --epochs 1

echo "One-epoch pipeline completed at $(date --iso-8601=seconds)"

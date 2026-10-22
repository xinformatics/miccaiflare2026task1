#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

# This host runs systemd-oomd and can kill long-lived trainers under sustained
# page-cache pressure even when plenty of reclaimable RAM remains. Single-GPU
# jobs use nnU-Net's supported single-threaded augmentation path (zero worker
# subprocesses) plus one inference/export worker. This prevents a killed main
# trainer from leaving an augmentation child holding a CUDA context.
export nnUNet_n_proc_DA=0
export nnUNet_def_n_proc=1
export NNUNET_EPOCH_PREDICTION_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

: "${NNUNET_CONFIGURATION:?Set NNUNET_CONFIGURATION}"
: "${NNUNET_BATCH_SIZE:?Set NNUNET_BATCH_SIZE}"

# A durable guard prevents background monitors from reviving experiments that
# the user explicitly retired. Remove STOP_RESENC only when an intentional
# ResEnc resume is requested.
if [[ "${NNUNET_CONFIGURATION}" == "3d_fullres_ResEncS6" \
      && -f "${SCRIPT_DIR}/STOP_RESENC" ]]; then
  echo "ResEnc launch blocked by ${SCRIPT_DIR}/STOP_RESENC" >&2
  exit 3
fi

if [[ "${NNUNET_CONFIGURATION}" == "3d_fullres_WinnerW48" \
      && -f "${SCRIPT_DIR}/STOP_WINNER_W48" ]]; then
  echo "WinnerW48 launch blocked by ${SCRIPT_DIR}/STOP_WINNER_W48" >&2
  exit 3
fi

if [[ "${NNUNET_CONFIGURATION}" == "3d_fullres_S4D2W32_ft" \
      && -f "${SCRIPT_DIR}/STOP_FINETUNE" ]]; then
  echo "Fine-tune launch blocked by ${SCRIPT_DIR}/STOP_FINETUNE" >&2
  exit 3
fi

export NNUNET_WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Expose exactly one physical GPU}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one GPU for this launcher." >&2
  exit 1
fi

PLANS="${nnUNet_preprocessed}/${NNUNET_DATASET:-Dataset501_PanCancer}/${NNUNET_PLANS:-nnUNetPlans}.json"
DATA="${nnUNet_preprocessed}/${NNUNET_DATASET:-Dataset501_PanCancer}/nnUNetPlans_3d_fullres_S4D2W32"
if [[ ! -f "${PLANS}" || ! -d "${DATA}" ]]; then
  echo "Required plans or transferred preprocessed cache is missing." >&2
  exit 1
fi

python - "${PLANS}" "${NNUNET_CONFIGURATION}" "${NNUNET_BATCH_SIZE}" <<'PY'
import json
import sys

plans = json.load(open(sys.argv[1]))
configuration = sys.argv[2]
batch_size = int(sys.argv[3])
if configuration not in plans["configurations"]:
    raise SystemExit(f"Configuration not found: {configuration}")
if batch_size < 1:
    raise SystemExit("NNUNET_BATCH_SIZE must be positive")
print(f"Configuration={configuration}; microbatch={batch_size}; single visible GPU")
PY

python "${SCRIPT_DIR}/train_wandb.py" "$@"

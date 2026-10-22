#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

PLANS="${nnUNet_preprocessed}/Dataset501_PanCancer/nnUNetPlans.json"
DATA="${nnUNet_preprocessed}/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32"
if [[ ! -f "${PLANS}" || ! -d "${DATA}" ]]; then
  echo "Preprocessed ResEnc S6 data is missing. Run the S6 resumable preprocessor first." >&2
  exit 1
fi

# This project is intended to train a six-stage residual-encoder model. Refuse
# to silently launch the older auto-planned seven-stage/141M configuration.
python - "${PLANS}" <<'PY'
import json
import sys

plans = json.load(open(sys.argv[1]))
configuration = plans["configurations"]["3d_fullres_ResEncS6"]
architecture = configuration["architecture"]
stages = architecture["arch_kwargs"]["n_stages"]
network = architecture["network_class_name"].rsplit(".", 1)[-1]
features = architecture["arch_kwargs"]["features_per_stage"]
blocks = architecture["arch_kwargs"]["n_blocks_per_stage"]
if (network != "ResidualEncoderUNet" or stages != 6 or
        features != [32, 64, 128, 256, 320, 320] or
        blocks != [1, 3, 4, 6, 6, 6] or
        configuration["spacing"] != [2.4, 1.6, 1.6] or
        configuration["patch_size"] != [128, 160, 160]):
    raise SystemExit(
        "Training launch blocked: plan does not match the transferred exact "
        "ResEnc S6 W32 recipe."
    )
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose two GPUs (default: 0,1)." >&2
  exit 1
fi

# The NVIDIA device nodes are currently transiently absent on this host. Do not
# throw away an unattended launch after hours of preprocessing: wait until both
# requested GPUs are actually visible. Set GPU_WAIT_TIMEOUT_SECONDS to a positive
# number to make an automated job fail after a deadline; zero waits indefinitely.
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
GPU_WAIT_STARTED="$(date +%s)"
while ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 2 else 1)
PY
do
  GPU_WAIT_NOW="$(date +%s)"
  GPU_WAIT_ELAPSED="$((GPU_WAIT_NOW - GPU_WAIT_STARTED))"
  if (( GPU_WAIT_TIMEOUT_SECONDS > 0 && GPU_WAIT_ELAPSED >= GPU_WAIT_TIMEOUT_SECONDS )); then
    echo "Timed out after ${GPU_WAIT_ELAPSED}s waiting for two CUDA GPUs." >&2
    exit 1
  fi
  echo "$(date '+%F %T') waiting for two CUDA GPUs (${GPU_WAIT_ELAPSED}s elapsed)"
  sleep 60
done

python - "${PLANS}" <<'PY'
import json
import sys

import torch

plans = json.load(open(sys.argv[1]))
batch_size = plans["configurations"]["3d_fullres_ResEncS6"]["batch_size"]
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Expected 2 visible CUDA GPUs, found {torch.cuda.device_count()}")
if batch_size < 2:
    raise SystemExit(f"Planned global batch size {batch_size} is too small for 2-GPU DDP")
print(f"Using 2 CUDA GPUs; planned global batch size: {batch_size}")
PY

# The project entry point launches two DDP ranks and adds rank-0 W&B logging
# without changing the stock nnU-Net architecture or checkpoint format.
python "${SCRIPT_DIR}/train_wandb.py" "$@"

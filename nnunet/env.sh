#!/usr/bin/env bash

# Source this file before calling nnU-Net commands directly.
NNUNET_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

export nnUNet_raw="${NNUNET_PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_PROJECT_ROOT}/nnUNet_results"
export PYTHONPATH="${NNUNET_PROJECT_ROOT}:${PYTHONPATH:-}"

# A small subset of the challenge NIfTIs encodes gantry tilt/shear in sform:
# 13 training images and 10 masks. Their paired voxel arrays and affines match,
# but ITK rejects non-orthonormal directions by default. ITK's documented
# permissive mode orthogonalizes the direction on read without rewriting the
# original challenge files. nnU-Net otherwise cannot pass integrity checking.
export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"

# Keep caches and temporary files on the project volume. The default home
# cache is not writable in this environment.
export MPLCONFIGDIR="${NNUNET_PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${NNUNET_PROJECT_ROOT}/.cache"

# W&B defaults to durable local logging. Set WANDB_MODE=online after `wandb
# login` (or provide WANDB_API_KEY) for live dashboard sync.
export WANDB_PROJECT="${WANDB_PROJECT:-pancancer-ct}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${NNUNET_PROJECT_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${NNUNET_PROJECT_ROOT}/.cache/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${NNUNET_PROJECT_ROOT}/.cache/wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${NNUNET_PROJECT_ROOT}/wandb-artifacts}"

# Two augmentation workers per training rank is the conservative setting after
# higher worker counts proved unstable during preprocessing on this host.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Export and officially score the 50 held-out public cases after every epoch.
# The two DDP ranks split the cases and therefore use both GPUs. Predictions,
# per-case metrics, and a compact summary are retained in epoch directories.
export NNUNET_EPOCH_PREDICTIONS="${NNUNET_EPOCH_PREDICTIONS:-1}"
export NNUNET_EPOCH_PREDICTION_EVERY="${NNUNET_EPOCH_PREDICTION_EVERY:-1}"
export NNUNET_EPOCH_PREDICTION_WORKERS="${NNUNET_EPOCH_PREDICTION_WORKERS:-2}"
export NNUNET_PUBLIC_IMAGES="${NNUNET_PUBLIC_IMAGES:-${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Images}"
export NNUNET_PUBLIC_LABELS="${NNUNET_PUBLIC_LABELS:-${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Labels}"
export NNUNET_EPOCH_PREDICTION_DIR="${NNUNET_EPOCH_PREDICTION_DIR:-${NNUNET_PROJECT_ROOT}/predictions/public_by_epoch}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" \
         "${MPLCONFIGDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
         "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}"

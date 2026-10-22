#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

DATASET="Dataset501_PanCancer"
PLANS_NAME="nnUNetPlans"
CONFIGURATION="3d_fullres_ResEncS6"
FOLD="all"
MODEL_DIR="${nnUNet_results}/${DATASET}/nnUNetTrainer__${PLANS_NAME}__${CONFIGURATION}/fold_${FOLD}"
PLANS="${nnUNet_preprocessed}/${DATASET}/${PLANS_NAME}.json"
PUBLIC_IMAGES="${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Images"
PUBLIC_LABELS="${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Labels"
OUTPUT="${NNUNET_PROJECT_ROOT}/predictions/public_s6_fold_all"

if [[ ! -f "${MODEL_DIR}/checkpoint_final.pth" ]]; then
  echo "Trained fold-0 checkpoint is missing: ${MODEL_DIR}/checkpoint_final.pth" >&2
  exit 1
fi
if [[ ! -f "${PLANS}" ]]; then
  echo "Plans file is missing: ${PLANS}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

nnUNetv2_predict \
  -i "${PUBLIC_IMAGES}" \
  -o "${OUTPUT}" \
  -d 501 \
  -p "${PLANS_NAME}" \
  -c "${CONFIGURATION}" \
  -f "${FOLD}" \
  -npp 2 \
  -nps 2 \
  --disable_tta \
  --continue_prediction

nnUNetv2_evaluate_folder \
  "${PUBLIC_LABELS}" \
  "${OUTPUT}" \
  -djfile "${nnUNet_raw}/${DATASET}/dataset.json" \
  -pfile "${PLANS}" \
  -o "${OUTPUT}/summary.json" \
  -np 4

# Also calculate the challenge's actual DSC/NSD decision ladder, including the
# DSC<0.2 NSD cutoff and 2 mm surface tolerance.
python "${NNUNET_PROJECT_ROOT}/handoff_bundle/code/evaluate.py" \
  --pred "${OUTPUT}" \
  --gt "${PUBLIC_LABELS}" \
  --csv "${OUTPUT}/challenge_metrics.csv"

python "${SCRIPT_DIR}/log_public_to_wandb.py" \
  "${OUTPUT}/summary.json" \
  "${MODEL_DIR}/wandb_run_id.txt" \
  --challenge-csv "${OUTPUT}/challenge_metrics.csv"

echo "Public validation metrics: ${OUTPUT}/summary.json"

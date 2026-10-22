#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

DATASET="Dataset501_PanCancer"
PLANS_NAME="nnUNetPlans"
CONFIGURATION="3d_fullres_ResEncS6"
FOLD="${FOLD:-all}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
CONTINUE_PREDICTION="${CONTINUE_PREDICTION:-0}"
MODEL_DIR="${nnUNet_results}/${DATASET}/nnUNetTrainer__${PLANS_NAME}__${CONFIGURATION}/fold_${FOLD}"
PLANS="${nnUNet_preprocessed}/${DATASET}/${PLANS_NAME}.json"
HIDDEN_IMAGES="${NNUNET_PROJECT_ROOT}/validation/Validation-Hidden-Images"
OUTPUT_ROOT="${NNUNET_PROJECT_ROOT}/predictions/hidden_fold${FOLD}"
PART0="${OUTPUT_ROOT}/part0"
PART1="${OUTPUT_ROOT}/part1"
SUBMISSION="${NNUNET_PROJECT_ROOT}/submissions/pancancer_hidden_fold${FOLD}.zip"

package_only=0
if [[ $# -eq 1 && "${1}" == "--package-only" ]]; then
  package_only=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--package-only]" >&2
  exit 2
fi

if (( package_only == 0 )); then
  if [[ ! -f "${MODEL_DIR}/${CHECKPOINT}" ]]; then
    echo "Checkpoint is missing: ${MODEL_DIR}/${CHECKPOINT}" >&2
    exit 1
  fi
  if [[ ! -f "${PLANS}" ]]; then
    echo "Plans file is missing: ${PLANS}" >&2
    exit 1
  fi
  if [[ ! -d "${HIDDEN_IMAGES}" ]]; then
    echo "Hidden-validation images are missing: ${HIDDEN_IMAGES}" >&2
    exit 1
  fi

  mkdir -p "${PART0}" "${PART1}"

  common_args=(
    -i "${HIDDEN_IMAGES}"
    -d 501
    -p "${PLANS_NAME}"
    -c "${CONFIGURATION}"
    -f "${FOLD}"
    -chk "${CHECKPOINT}"
    -npp 2
    -nps 2
    -num_parts 2
    --disable_tta
  )
  if [[ "${CONTINUE_PREDICTION}" == "1" ]]; then
    common_args+=(--continue_prediction)
  fi

  echo "Predicting hidden-validation part 0 on GPU 0..."
  CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    "${common_args[@]}" -part_id 0 -o "${PART0}" &
  pid0=$!

  echo "Predicting hidden-validation part 1 on GPU 1..."
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
    "${common_args[@]}" -part_id 1 -o "${PART1}" &
  pid1=$!

  cleanup_children() {
    kill "${pid0}" "${pid1}" 2>/dev/null || true
  }
  trap cleanup_children INT TERM

  failed=0
  if ! wait "${pid0}"; then
    echo "GPU 0 inference failed." >&2
    failed=1
  fi
  if ! wait "${pid1}"; then
    echo "GPU 1 inference failed." >&2
    failed=1
  fi
  trap - INT TERM
  if (( failed != 0 )); then
    exit 1
  fi
fi

python "${SCRIPT_DIR}/package_codabench.py" \
  --images "${HIDDEN_IMAGES}" \
  --pred-dir "${PART0}" \
  --pred-dir "${PART1}" \
  --output "${SUBMISSION}" \
  --expected-count 100

echo
echo "Upload this file in Codabench -> My Submissions -> Development Phase:"
echo "${SUBMISSION}"

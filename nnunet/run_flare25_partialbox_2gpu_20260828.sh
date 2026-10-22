#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/nvme2-data/shashank/MICCAI26"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
runner="${project_root}/nnunet/run_flare25_partialbox_finetune.py"
checkpoint="${project_root}/external/FLARE25-Task1-winner/nnUNet_results/nnUNet_results/Dataset523_train_all/nnUNetTrainer_Epoch5000_Lr1e3__nnUNetPlans__3d_fullres_S4D2W32/fold_all/checkpoint_final.pth"
run_dir="${project_root}/nnunet/runs/flare25_partialbox_50ep_20260828"
log_file="${run_dir}/training.log"
terminal="${run_dir}/TERMINAL.json"

mkdir -p "${run_dir}"
cd "${project_root}"

export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${run_dir}/matplotlib"
export WANDB_PROJECT="pancancer-ct"
export WANDB_MODE="${WANDB_MODE:-online}"
export nnUNet_compile="false"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-0}"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export PANCANCER_AUGMENTATION_SEED_OFFSET="2026082800"

mkdir -p "${MPLCONFIGDIR}"

if [[ ! -e /dev/nvidia0 || ! -e /dev/nvidia1 ]]; then
  "${python_bin}" - "${terminal}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1])
p.write_text(json.dumps({
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "status":"WAITING_FOR_GPU",
  "decision":"TRAINER_READY_GPU_DEVICES_UNAVAILABLE",
  "required_devices":["/dev/nvidia0","/dev/nvidia1"],
  "training_started":False
},indent=2)+"\n")
PY
  echo "FLARE25 trainer is ready, but /dev/nvidia0 and /dev/nvidia1 are unavailable." >&2
  exit 3
fi

expected_sha="c3f136cf916ea135025dcd43742e8e42ce6fcb6c9aab7802522bd8fb3e3b6941"
observed_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${observed_sha}" != "${expected_sha}" ]]; then
  echo "FLARE25 checkpoint SHA256 mismatch: ${observed_sha}" >&2
  exit 4
fi

"${python_bin}" - "${terminal}" "${checkpoint}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
terminal=Path(sys.argv[1]); checkpoint=Path(sys.argv[2])
terminal.write_text(json.dumps({
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "status":"RUNNING",
  "decision":"FINETUNE_OFFICIAL_FLARE25_ON_DATASET501_PARTIALBOX",
  "launcher_pid":os.getppid(),
  "source_checkpoint":str(checkpoint),
  "gpus":2,
  "training_started":True,
  "public_used_for_training":False,
  "hidden_used_for_training":False
},indent=2)+"\n")
PY

"${python_bin}" -u "${runner}" Dataset501_PanCancer 3d_fullres_FLARE25FT all \
  -tr nnUNetTrainerFLARE25PartialBoxFineTune -p nnUNetPlans \
  -pretrained_weights "${checkpoint}" -num_gpus 2 2>&1 | tee "${log_file}"

"${python_bin}" - "${terminal}" "${checkpoint}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
terminal=Path(sys.argv[1]); checkpoint=Path(sys.argv[2])
terminal.write_text(json.dumps({
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "status":"COMPLETE",
  "decision":"EVALUATE_PREDECLARED_EPOCHS_10_25_50_ON_FIXED_PUBLIC",
  "source_checkpoint":{"path":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
  "public_used_for_training":False,
  "hidden_used_for_training":False,
  "uploaded":False
},indent=2)+"\n")
PY

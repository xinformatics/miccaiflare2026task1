#!/usr/bin/env bash
set -euo pipefail

# Label-free FLARE25 inference over the frozen 707-case primary-lesion audit
# cohort. This controller cannot read labels, public/hidden validation, create
# submissions, upload, or start an optimizer.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${project_root}/external/FLARE25-Task1-winner"
input="${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/staged_images"
manifest="${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/manifest.json"
results="${root}/nnUNet_results/nnUNet_results"
checkpoint="${results}/Dataset523_train_all/nnUNetTrainer_Epoch5000_Lr1e3__nnUNetPlans__3d_fullres_S4D2W32/fold_all/checkpoint_final.pth"
output="${root}/exact707_primary_audit_20260828"
predictions="${output}/predictions"
terminal="${output}/INFERENCE_TERMINAL.json"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predict_bin="/home/m348911/miniconda3/envs/radoncAI/bin/nnUNetv2_predict"

mkdir -p "${output}" "${predictions}" /tmp/h25-707
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "FLARE25 exact-707 controller already active" >&2; exit 1; }
exec >>"${output}/inference.log" 2>&1

[[ $(find "${input}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 707 ]] || {
  echo "exact-707 input inventory is not 707" >&2; exit 2;
}
[[ $(sha256sum "${manifest}" | cut -d' ' -f1) == fdd67a08838ba1103ca27aae45637aaf33743e428f23a4b7da75abe5195de7c1 ]] || {
  echo "exact-707 manifest hash mismatch" >&2; exit 2;
}
[[ $(sha256sum "${checkpoint}" | cut -d' ' -f1) == c3f136cf916ea135025dcd43742e8e42ce6fcb6c9aab7802522bd8fb3e3b6941 ]] || {
  echo "FLARE25 checkpoint hash mismatch" >&2; exit 2;
}
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
  echo "unexpected existing terminal status: ${status}" >&2
  exit 2
fi

used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" -le 1024 ]] || { echo "GPU 1 is not idle: ${used} MiB" >&2; exit 3; }

export CUDA_VISIBLE_DEVICES=1
export nnUNet_compile=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/h25-707
export PYTHONPATH="${root}:${PYTHONPATH:-}"
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${results}"

started=$(date -Ins)
"${predict_bin}" \
  -i "${input}" \
  -o "${predictions}" \
  -c 3d_fullres_S4D2W32 \
  -f all \
  -d 523 \
  -tr nnUNetTrainer_Epoch5000_Lr1e3 \
  -npp 1 \
  -nps 1

count=$(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 707 ]] || { echo "FLARE25 exact-707 output inventory is ${count}, not 707" >&2; exit 4; }
finished=$(date -Ins)
"${python_bin}" - "${terminal}" "${started}" "${finished}" "${manifest}" "${checkpoint}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path

terminal = Path(sys.argv[1])
started, finished = sys.argv[2:4]
manifest, checkpoint = map(Path, sys.argv[4:6])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "cases": 707,
    "started_at": started,
    "finished_at": finished,
    "model": "official FLARE25 Task-1 winner",
    "manifest": {"path": str(manifest.resolve()), "sha256": digest(manifest)},
    "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest(checkpoint)},
    "execution": {"physical_gpu": 1, "npp": 1, "nps": 1, "tta": True},
    "decision_role": "PRIMARY_LESION_EXPERT_DIAGNOSTIC_ONLY",
    "labels_read": False,
    "public_read": False,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
    "optimizer_started": False,
}
temporary = terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, terminal)
PY

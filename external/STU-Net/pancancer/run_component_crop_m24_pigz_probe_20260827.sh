#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="checkpoint_inference_epoch_0160.pth"
reference="${root}/public_sweeps/stunet_component_crop_e160_m24_public_20260827/predictions"
output="${root}/public_sweeps/stunet_component_crop_e160_m24_pigz_probe_20260827"
predictions="${output}/predictions"
predict_terminal="${output}/PREDICT_TERMINAL.json"
terminal="${output}/PROBE.json"
case_name="FLARE23Ts_0029.nii.gz"
gpu="${COMPONENT_CROP_GPU:-0}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-component-crop-pigz
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "pigz component-crop probe already active" >&2; exit 1; }
[[ -s "${reference}/${case_name}" ]]
used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] || exit 3
"${python_bin}" -u "${root}/predict_stunet_component_crops.py" \
  --images "${images}" --anchors "${anchor}" --output "${predictions}" \
  --model-dir "${model}" --checkpoint "${checkpoint}" --gpu "${gpu}" \
  --part-id 14 --num-parts 50 --workers 1 --io-workers 8 --margin-mm 24 \
  --dataset-role public --terminal "${predict_terminal}" >"${output}/predict.log" 2>&1
"${python_bin}" - "${case_name}" "${reference}" "${predictions}" \
  "${predict_terminal}" "${root}/predict_stunet_component_crops.py" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
name=sys.argv[1];reference,predictions,predict,script,output=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=np.asanyarray(nib.load(str(reference/name)).dataobj)
b=np.asanyarray(nib.load(str(predictions/name)).dataobj)
p=json.load(open(predict,encoding="utf-8"));row=p["cases"][0]
equal=bool(np.array_equal(a,b));eligible=bool(equal and row["seconds"]<=20 and p["peak_gpu_reserved_mb"]<=4096)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_PIGZ_CROP_LOADER" if eligible else "REJECT_PIGZ_CROP_LOADER",
 "case":name,"mask_equal_to_prior_m24":equal,"runtime_seconds":row["seconds"],
 "materialization_seconds":row["compressed_input_materialization_seconds"],
 "peak_gpu_reserved_mb":p["peak_gpu_reserved_mb"],"runtime_eligible_20s":row["seconds"]<=20,
 "memory_eligible_4096mb":p["peak_gpu_reserved_mb"]<=4096,
 "predict_terminal":{"path":str(predict.resolve()),"sha256":sha(predict)},
 "predictor":{"path":str(script.resolve()),"sha256":sha(script)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,"deployment_authorized":False}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(tmp,output)
print(json.dumps(payload,indent=2))
PY

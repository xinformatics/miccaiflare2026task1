#!/usr/bin/env bash
set -euo pipefail

# Public-only runtime/quality probe of native pre-preprocessing component crops.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
baseline="${root}/public_sweeps/stunetb_matched_e160_20260826/component_predictions"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
margin="${COMPONENT_CROP_MARGIN_MM:-48}"
[[ "${margin}" =~ ^(24|32|48)$ ]] || { echo "margin must be one of 24, 32, 48 mm" >&2; exit 2; }
output="${root}/public_sweeps/stunet_component_crop_e160_m${margin}_probe_20260827"
predictions="${output}/predictions"
predict_terminal="${output}/PREDICT_TERMINAL.json"
terminal="${output}/PROBE.json"
gpu="${COMPONENT_CROP_GPU:-0}"
part_id=30
case_name="FLARE23Ts_0061.nii.gz"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-component-crop
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "component-crop probe already active" >&2; exit 1; }

[[ "${gpu}" =~ ^[01]$ ]]
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ -s "${checkpoint}" ]]
"${python_bin}" - "${images}" "${case_name}" "${part_id}" <<'PY'
import sys
from pathlib import Path
images=Path(sys.argv[1]); expected=sys.argv[2]; part=int(sys.argv[3])
names=sorted(path.name.replace("_0000.nii.gz",".nii.gz") for path in images.glob("*_0000.nii.gz"))
if len(names)!=50 or names[part]!=expected: raise SystemExit("component-crop probe case drift")
PY
used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] || {
  echo "GPU ${gpu} is occupied (${used} MiB); component-crop probe not started" >&2; exit 3;
}

reuse=0
if [[ -s "${predict_terminal}" && -s "${predictions}/${case_name}" ]]; then
  if "${python_bin}" - "${predict_terminal}" "${checkpoint}" "${case_name}" "${margin}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,checkpoint=map(Path,sys.argv[1:3]);case=sys.argv[3];margin=float(sys.argv[4])
p=json.load(open(terminal,encoding="utf-8"));sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
rows=p.get("cases") or []
raise SystemExit(0 if p.get("status")=="COMPLETE" and p.get("case_count")==1
                 and p.get("margin_mm")==margin and len(rows)==1 and rows[0].get("case")==case
                 and p.get("checkpoint",{}).get("sha256")==sha and not p.get("hidden_read")
                 and not p.get("submission_created") and not p.get("uploaded") else 1)
PY
  then reuse=1; fi
fi
if [[ "${reuse}" -eq 0 ]]; then
  "${python_bin}" -u "${root}/predict_stunet_component_crops.py" \
    --images "${images}" --anchors "${anchor}" --output "${predictions}" \
    --model-dir "${model}" --checkpoint "$(basename "${checkpoint}")" \
    --gpu "${gpu}" --part-id "${part_id}" --num-parts 50 --workers 1 \
    --margin-mm "${margin}" --dataset-role public --terminal "${predict_terminal}" \
    >"${output}/predict.log" 2>&1
fi

"${python_bin}" - "${case_name}" "${anchor}" "${baseline}" "${labels}" \
  "${predictions}" "${predict_terminal}" "${checkpoint}" "${terminal}" "${margin}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
root=Path(sys.argv[2]).parents[3]/"STU-Net"/"pancancer"
gf=root.parents[1]/"GF-Screen"
sys.path[:0]=[str(root),str(gf)]
from analyze_stunet_gf_ensemble import confirmed_anchor_components
from fast_official_lesion_metrics import score_case
name=sys.argv[1];anchor,baseline,labels,predictions,predict_path,checkpoint,output=map(Path,sys.argv[2:9]);margin=float(sys.argv[9])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aimg=nib.load(str(anchor/name));a=np.asanyarray(aimg.dataobj).astype(bool)
with np.load(baseline/f"{name.removesuffix('.nii.gz')}.prob16.npz") as archive:
 probability=archive["lesion_probability_uint16"].astype(np.float32)/65535.0
reference=confirmed_anchor_components(a,probability>0.70,np.zeros_like(a),"stu")
oimg=nib.load(str(predictions/name));observed=np.asanyarray(oimg.dataobj).astype(bool)
truth=np.asanyarray(nib.load(str(labels/name)).dataobj).astype(bool)
spacing=aimg.header.get_zooms()[:3]
reference_score=score_case(truth.astype(np.uint8),reference.astype(np.uint8),spacing)
observed_score=score_case(truth.astype(np.uint8),observed.astype(np.uint8),spacing)
intersection=np.count_nonzero(reference&observed);den=int(reference.sum()+observed.sum())
mask_dice=float(2*intersection/den) if den else 1.0
predict=json.load(open(predict_path,encoding="utf-8"));row=predict["cases"][0]
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"RUN_ALL50_M48" if row["seconds"]<=20 and mask_dice>=0.98 else "REVISE_CROP_ROUTE",
 "case":name,"margin_mm":margin,"mask_equal":bool(np.array_equal(reference,observed)),
 "mask_dice_vs_full_stunet_only":mask_dice,
 "reference":{"Lesion_DSC":reference_score[0],"Lesion_NSD":reference_score[1],"foreground_voxels":int(reference.sum())},
 "cropped":{"Lesion_DSC":observed_score[0],"Lesion_NSD":observed_score[1],"foreground_voxels":int(observed.sum())},
 "delta_DSC":float(observed_score[0]-reference_score[0]),"delta_NSD":float(observed_score[1]-reference_score[1]),
 "runtime_seconds":row["seconds"],"crop_count":row["crop_count"],"crop_voxels":row["crop_voxels"],
 "native_voxels":row["native_voxels"],"peak_gpu_reserved_mb":row["peak_gpu_reserved_mb"],
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "predict_terminal":{"path":str(predict_path.resolve()),"sha256":sha(predict_path)},
 "public_images_read":True,"public_labels_read":True,"hidden_read":False,
 "submission_created":False,"uploaded":False,"deployment_authorized":False}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(tmp,output)
print(json.dumps(payload,indent=2))
PY

#!/usr/bin/env bash
set -euo pipefail

# Public-label-free one-case equivalence/runtime probe for the experimental
# sparse STU p70 confirmer. This script fails before inference when the chosen
# GPU is occupied and cannot access hidden data, package, or upload.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
baseline_root="${root}/public_sweeps/stunetb_matched_e160_20260826"
baseline_stu="${baseline_root}/component_predictions"
baseline_terminal="${baseline_root}/TERMINAL.json"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
output="${root}/public_sweeps/stunet_anchor_sparse_e160_probe_20260827"
predictions="${output}/predictions"
predict_terminal="${output}/PREDICT_TERMINAL.json"
terminal="${output}/EQUIVALENCE.json"
gpu="${SPARSE_PROBE_GPU:-0}"
part_id=14
num_parts=50
case_name="FLARE23Ts_0029.nii.gz"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-sparse
[[ "${gpu}" =~ ^[01]$ ]] || { echo "SPARSE_PROBE_GPU must be 0 or 1" >&2; exit 2; }
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "sparse probe already active" >&2; exit 1; }

[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${anchor}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${freetumor}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ -s "${checkpoint}" && -s "${baseline_terminal}" ]]
"${python_bin}" - "${images}" "${case_name}" "${part_id}" "${baseline_terminal}" \
  "${checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
images=Path(sys.argv[1]);expected=sys.argv[2];part=int(sys.argv[3]);terminal=Path(sys.argv[4]);checkpoint=Path(sys.argv[5])
names=sorted(path.name.replace("_0000.nii.gz",".nii.gz") for path in images.glob("*_0000.nii.gz"))
if len(names)!=50 or names[part]!=expected: raise SystemExit("sparse probe case selection drift")
p=json.load(open(terminal,encoding="utf-8"));sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
if (p.get("status")!="COMPLETE" or p.get("evaluation_epoch")!=160
    or p.get("component_checkpoint",{}).get("sha256")!=sha(checkpoint)
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid immutable component-epoch-160 baseline")
PY

used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] || {
  echo "GPU ${gpu} is occupied (${used} MiB); sparse probe not started" >&2
  exit 3
}

"${python_bin}" -u "${root}/predict_stunet_anchor_sparse.py" \
  --images "${images}" --anchors "${anchor}" --freetumor "${freetumor}" \
  --output "${predictions}" --model-dir "${model}" \
  --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
  --part-id "${part_id}" --num-parts "${num_parts}" --workers 1 \
  --dataset-role public --terminal "${predict_terminal}" \
  >"${output}/predict.log" 2>&1

"${python_bin}" - "${case_name}" "${anchor}" "${freetumor}" "${baseline_stu}" \
  "${predictions}" "${predict_terminal}" "${checkpoint}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
sys.path.insert(0,str(Path(sys.argv[2]).parents[3]/"STU-Net"/"pancancer"))
from analyze_stunet_gf_ensemble import confirmed_anchor_components
name=sys.argv[1];anchor,free,baseline,predictions,predict_terminal,checkpoint,output=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aimg=nib.load(str(anchor/name));fimg=nib.load(str(free/name));simg=nib.load(str(predictions/name))
a=np.asanyarray(aimg.dataobj).astype(bool);f=np.asanyarray(fimg.dataobj).astype(bool)
with np.load(baseline/f"{name.removesuffix('.nii.gz')}.prob16.npz") as archive:
    probability=archive["lesion_probability_uint16"].astype(np.float32)/65535.0
reference=confirmed_anchor_components(a,probability>0.70,f,"either")
sparse=np.asanyarray(simg.dataobj).astype(bool)
predict=json.load(open(predict_terminal,encoding="utf-8"));rows=predict.get("cases") or []
geometry_equal=simg.shape==aimg.shape and np.allclose(simg.affine,aimg.affine,atol=1e-6)
mask_equal=bool(np.array_equal(sparse,reference));row=rows[0] if len(rows)==1 else {}
wall=float(predict.get("wall_seconds",float("inf")))
probe_pass=bool(mask_equal and geometry_equal and wall<=55 and not row.get("stunet_skipped",True)
                and row.get("selected_tiles",0)<row.get("full_tiles",0))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if probe_pass else "FAILED","case":name,"mask_equal":mask_equal,
 "geometry_equal":bool(geometry_equal),"runtime_eligible_55s":wall<=55,
 "wall_seconds":wall,"model_initialization_seconds":predict.get("model_initialization_seconds"),
 "full_tiles":row.get("full_tiles"),"selected_tiles":row.get("selected_tiles"),
 "tile_fraction":row.get("selected_tiles",0)/row.get("full_tiles",1),
 "probe_pass":probe_pass,"deployment_authorized":False,
 "authorization_requirement":"all 50 fixed-public final masks must match before deployment",
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "predict_terminal":{"path":str(predict_terminal.resolve()),"sha256":sha(predict_terminal)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps(payload,indent=2))
if not probe_pass: raise SystemExit("sparse probe failed exactness or runtime gate")
PY


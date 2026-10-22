#!/usr/bin/env bash
set -euo pipefail

# Stress-case, public-label-free exactness/runtime probe for the FreeTumor-free
# existing-support proxy plus sparse STU confirmer.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
gf="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${gf}/analysis/lesion_error_touchcomp_20260825/predictions"
bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
whole="${project_root}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project_root}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
full_stu="${root}/public_sweeps/stunetb_matched_e160_20260826/component_predictions"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
predictor="${root}/predict_stunet_proxy_sparse.py"
biomed_terminal="${gf}/docker_submission/analysis/biomedparse_dual_prompt_public_equivalence_20260827/EQUIVALENCE.json"
output="${root}/public_sweeps/stunet_proxy_sparse_probe_20260827"
reference="${output}/reference"
predictions="${output}/predictions"
predict_terminal="${output}/PREDICT.json"
terminal="${output}/EQUIVALENCE.json"
gpu="${PROXY_SPARSE_PROBE_GPU:-0}"
part_id=28

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-proxy-sparse
export PYTHONPATH="${gf}/docker_submission:${gf}:${root}:${PYTHONPATH:-}"
[[ "${gpu}" =~ ^[01]$ ]] || { echo "PROXY_SPARSE_PROBE_GPU must be 0 or 1" >&2; exit 2; }
mkdir -p "${output}" "${reference}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "proxy-sparse probe already active" >&2; exit 1; }

while [[ ! -s "${biomed_terminal}" ]]; do
  printf '%s waiting for serialized GF/STU/Biomed equivalence chain\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${biomed_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_prompt_outputs")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("BiomedParse equivalence prerequisite is invalid")
PY

for directory in "${images}" "${anchor}" "${bbox}" "${whole}" "${tumor}" "${full_stu}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "proxy-sparse fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
for file in "${checkpoint}" "${predictor}"; do
  [[ -s "${file}" ]] || { echo "missing proxy-sparse input: ${file}" >&2; exit 2; }
done

while :; do
  read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
print(memory["MemAvailable"],avg10)
PY
  )
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used}" <<'PY'
import sys
available,pressure,gpu=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and gpu<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe proxy-sparse probe: available=%s KiB pressure=%s gpu=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used}"
  sleep 30
done

"${python_bin}" - "${images}" "${anchor}" "${bbox}" "${whole}" "${tumor}" \
  "${full_stu}" "${reference}" "${part_id}" <<'PY'
import os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
from runtime_winner_mask_recipe import confirm_with_stunet_p70_or_existing_support_proxy
images,anchor,bbox,whole,tumor,stu,output=map(Path,sys.argv[1:8]);part=int(sys.argv[8])
image_path=sorted(images.glob("*_0000.nii.gz"))[part]
name=image_path.name.replace("_0000.nii.gz",".nii.gz")
reference=nib.load(str(image_path))
def binary(directory):
 image=nib.load(str(directory/name));values=np.asanyarray(image.dataobj)
 if image.shape!=reference.shape or not np.allclose(image.affine,reference.affine,atol=1e-4):raise SystemExit("geometry mismatch")
 return values.astype(bool)
with np.load(stu/f"{name.removesuffix('.nii.gz')}.prob16.npz") as archive:
 probability=archive["lesion_probability_uint16"]
result,_=confirm_with_stunet_p70_or_existing_support_proxy(
 anchor=binary(anchor),prompt_whole_body_lesion=binary(whole),
 prompt_tumor=binary(tumor),bbox_raw=binary(bbox),
 stunet_probability_uint16=probability,
 spacing=tuple(float(x) for x in reference.header.get_zooms()[:3]))
header=reference.header.copy();header.set_data_dtype(np.uint8)
destination=output/name;temporary=destination.with_name(f".{name}.{os.getpid()}.tmp.nii.gz")
nib.save(nib.Nifti1Image(result.astype(np.uint8),reference.affine,header),str(temporary));os.replace(temporary,destination)
print(name)
PY

timeout --signal=TERM 120s "${python_bin}" -u "${predictor}" \
  --images "${images}" --anchors "${anchor}" --bbox-raw "${bbox}" \
  --prompt-whole "${whole}" --prompt-tumor "${tumor}" \
  --output "${predictions}" --model-dir "${model}" \
  --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
  --part-id "${part_id}" --num-parts 50 --workers 1 \
  --dataset-role public --terminal "${predict_terminal}" \
  >"${output}/predict.log" 2>&1

"${python_bin}" - "${images}" "${reference}" "${predictions}" "${predict_terminal}" \
  "${checkpoint}" "${predictor}" "${terminal}" "${part_id}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,reference,predictions,predict,checkpoint,predictor,output=map(Path,sys.argv[1:8]);part=int(sys.argv[8])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))[part]
p=json.load(open(predict,encoding="utf-8"));rows=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1 or len(rows)!=1
    or rows[0].get("case")!=name or p.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid proxy-sparse prediction terminal")
r=nib.load(str(reference/name));o=nib.load(str(predictions/name))
mask_equal=bool(np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)))
geometry_equal=bool(r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6))
runtime=float(p["wall_seconds"]);runtime_eligible=runtime<=20.0
passed=bool(mask_equal and geometry_equal and runtime_eligible)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUEUE_ALL50_PROXY_SPARSE" if passed else "DO_NOT_QUEUE_ALL50_PROXY_SPARSE",
 "probe_pass":passed,"deployment_authorized":False,"case":name,
 "stress_selection":"maximum label-blind unresolved-anchor physical volume among fixed-public cases",
 "mask_equal":mask_equal,"geometry_equal":geometry_equal,"runtime_eligible_20s":runtime_eligible,
 "wall_seconds":runtime,"case_runtime_seconds":float(rows[0]["seconds"]),
 "stunet_skipped":bool(rows[0]["stunet_skipped"]),"full_tiles":int(rows[0]["full_tiles"]),
 "selected_tiles":int(rows[0]["selected_tiles"]),
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "predictor":{"path":str(predictor.resolve()),"sha256":sha(predictor)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

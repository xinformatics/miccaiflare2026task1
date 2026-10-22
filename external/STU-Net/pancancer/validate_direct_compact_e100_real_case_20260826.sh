#!/usr/bin/env bash
set -euo pipefail

# Real-GPU equivalence gate for the opt-in direct compact exporter. This reads
# one fixed-public image and an already generated legacy epoch-100 prediction;
# it cannot access hidden inputs, package a submission, or upload anything.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
legacy_root="${root}/public_sweeps/stunetb_e100_ensemble_20260826"
legacy="${legacy_root}/predictions"
public_terminal="${legacy_root}/TERMINAL.json"
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model_dir}/fold_all/checkpoint_epoch_0100.pth"
output="${root}/public_sweeps/direct_compact_equivalence_e100_20260826"
direct="${output}/predictions"
predict_terminal="${output}/PREDICT_TERMINAL.json"
terminal="${output}/EQUIVALENCE.json"
gpu="${DIRECT_COMPACT_EQUIVALENCE_GPU:-0}"
case_name="FLARE23Ts_0001.nii.gz"

[[ "${gpu}" =~ ^[01]$ ]] || { echo "DIRECT_COMPACT_EQUIVALENCE_GPU must be 0 or 1" >&2; exit 2; }
mkdir -p "${output}" "${direct}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "direct-compact equivalence already active" >&2; exit 1; }

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
  mv --no-clobber "${terminal}" "${output}/EQUIVALENCE.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

[[ -s "${public_terminal}" ]] || { echo "epoch-100 public gate has not completed" >&2; exit 3; }
[[ -s "${legacy}/${case_name}" ]]
[[ -s "${legacy}/${case_name%.nii.gz}.prob16.npz" ]]
[[ -s "${checkpoint}" ]]

# The chosen public case is the first sorted image, so num-parts=50/part-id=0
# isolates exactly this one CT without a temporary image copy or symlink.
"${python_bin}" - "${images}" "${case_name}" <<'PY'
import sys
from pathlib import Path
images=Path(sys.argv[1]);expected=sys.argv[2]
names=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))
if len(names)!=50 or names[0]!=expected: raise SystemExit("equivalence case selection drift")
PY

"${python_bin}" -u "${root}/predict_stunet_safe.py" \
  --images "${images}" --output "${direct}" --model-dir "${model_dir}" \
  --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
  --part-id 0 --num-parts 50 --workers 1 --save-probabilities \
  --direct-compact-probabilities --dataset-role public \
  --terminal "${predict_terminal}" > "${output}/predict.log" 2>&1

"${python_bin}" - "${legacy}" "${direct}" "${checkpoint}" \
  "${predict_terminal}" "${case_name}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
legacy,direct,checkpoint,predict_terminal=map(Path,sys.argv[1:5]);name=sys.argv[5];output=Path(sys.argv[6])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
predict=json.load(open(predict_terminal,encoding="utf-8"))
if (predict.get("status")!="COMPLETE" or predict.get("dataset_role")!="public"
    or predict.get("part_id")!=0 or predict.get("num_parts")!=50
    or predict.get("case_count")!=1 or predict.get("cases")!=[name]
    or not predict.get("save_probabilities") or not predict.get("direct_compact_probabilities")
    or predict.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
    or not predict.get("public_images_read") or predict.get("public_labels_read")
    or predict.get("hidden_read") or predict.get("hidden_labels_read")
    or predict.get("submission_created") or predict.get("uploaded")):
    raise SystemExit("invalid direct-compact predictor terminal")
legacy_image=nib.load(str(legacy/name));direct_image=nib.load(str(direct/name))
legacy_mask=np.asanyarray(legacy_image.dataobj);direct_mask=np.asanyarray(direct_image.dataobj)
probability_name=name.removesuffix(".nii.gz")+".prob16.npz"
with np.load(legacy/probability_name) as archive:
    legacy_probability=archive["lesion_probability_uint16"]
with np.load(direct/probability_name) as archive:
    direct_probability=archive["lesion_probability_uint16"]
mask_equal=bool(np.array_equal(legacy_mask,direct_mask))
probability_equal=bool(np.array_equal(legacy_probability,direct_probability))
geometry_equal=(legacy_image.shape==direct_image.shape and np.allclose(legacy_image.affine,direct_image.affine,atol=1e-6))
maximum_probability_code_difference=int(np.max(np.abs(legacy_probability.astype(np.int32)-direct_probability.astype(np.int32))))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if mask_equal and probability_equal and geometry_equal else "FAILED",
 "case":name,"mask_equal":mask_equal,"probability_uint16_equal":probability_equal,
 "geometry_equal":geometry_equal,"maximum_probability_code_difference":maximum_probability_code_difference,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "legacy_mask":{"path":str((legacy/name).resolve()),"sha256":sha(legacy/name)},
 "legacy_probability":{"path":str((legacy/probability_name).resolve()),"sha256":sha(legacy/probability_name)},
 "direct_mask":{"path":str((direct/name).resolve()),"sha256":sha(direct/name)},
 "direct_probability":{"path":str((direct/probability_name).resolve()),"sha256":sha(direct/probability_name)},
 "predict_terminal":{"path":str(predict_terminal.resolve()),"sha256":sha(predict_terminal)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
 "activation_authorized":bool(mask_equal and probability_equal and geometry_equal)}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
if payload["status"]!="COMPLETE": raise SystemExit("direct compact exporter differs from legacy")
print(json.dumps(payload,indent=2))
PY

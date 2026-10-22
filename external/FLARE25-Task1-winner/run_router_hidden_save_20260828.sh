#!/usr/bin/env bash
set -euo pipefail

project="/mnt/nvme2-data/shashank/MICCAI26"
root="${project}/external/FLARE25-Task1-winner"
public_terminal="${root}/router_public_replay_20260828/TERMINAL.json"
router_report="${root}/exact707_primary_audit_20260828/mask_router_cv/report.json"
force_router_hidden="${FORCE_ROUTER_HIDDEN:-0}"
if [[ "${force_router_hidden}" == 1 ]]; then
  hidden_root="${root}/router_hidden_forced_20260828"
  archive="${project}/submissions/gf_flrtr_force_0828.zip"
else
  hidden_root="${root}/router_hidden_20260828"
  archive="${project}/submissions/gf_flrtr_0828.zip"
fi
gf_hidden="${hidden_root}/gf_e160_champion"
flare_hidden="${hidden_root}/flare25"
routed="${hidden_root}/routed"
docker_tmp="${hidden_root}/docker_tmp"
terminal="${hidden_root}/TERMINAL.json"
champion="${project}/submissions/gf_stu_e160_gfanchorconfirmeitherp70_0825.zip"
champion_sha="6f84ec34f53c8a97ddf62fbae7148d34507b61be3b838f3d263ae408de25456a"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${hidden_root}" "${gf_hidden}" "${flare_hidden}" "${routed}" "${docker_tmp}"
exec >"${hidden_root}/run.log" 2>&1
while [[ ! -s "${public_terminal}" ]]; do sleep 30; done
if ! "${python_bin}" - "${public_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("status")!="COMPLETE" or not p.get("public_gate_pass") or p.get("decision")!="AUTHORIZE_HIDDEN_SAVE_ONLY":
    raise SystemExit("fixed-public router gate failed; hidden inference forbidden")
PY
then
  if [[ "${force_router_hidden}" == 1 ]]; then
    echo "continuing under explicit user override despite failed public gate"
  else
  "${python_bin}" - "${public_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
public,terminal=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_CHAMPION","reason":"router public gate did not pass",
 "public_terminal":{"path":str(public.resolve()),"sha256":sha(public)},
 "hidden_images_read":False,"hidden_labels_read":False,"submission_created":False,"uploaded":False}
tmp=terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,terminal)
PY
    exit 0
  fi
fi
[[ $(sha256sum "${champion}" | cut -d' ' -f1) == "${champion_sha}" ]] || {
  echo "champion archive hash mismatch" >&2; exit 2;
}
"${python_bin}" - "${champion}" "${gf_hidden}" <<'PY'
import sys,zipfile
from pathlib import Path
archive,output=map(Path,sys.argv[1:])
with zipfile.ZipFile(archive) as z:
    members=[i for i in z.infolist() if not i.is_dir()]
    if len(members)!=100 or z.testzip() is not None: raise SystemExit("invalid champion ZIP")
    for item in members:
        if "/" in item.filename or not item.filename.endswith(".nii.gz"): raise SystemExit("non-flat champion ZIP")
        target=output/item.filename
        if not target.is_file() or target.stat().st_size!=item.file_size:
            target.write_bytes(z.read(item.filename))
PY

while true; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" -le 1024 ]] && break
  sleep 30
done

if [[ $(find "${flare_hidden}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  docker run --rm --name hias25-router-hidden-20260828 \
    --memory 48g --memory-swap 48g \
    --device /dev/nvidiactl --device /dev/nvidia1 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -v "${project}/validation/Validation-Hidden-Images:/workspace/inputs:ro" \
    -v "${flare_hidden}:/workspace/outputs" -v "${docker_tmp}:/tmp" \
    -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
    -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
    hias25:latest /bin/bash -lc \
    'export nnUNet_results=./nnUNet_results; nnUNetv2_predict -i ./inputs -o ./outputs -c 3d_fullres_S4D2W32 -f all -d 523 -tr nnUNetTrainer_Epoch5000_Lr1e3 -npp 0 -nps 0 --continue_prediction'
fi
[[ $(find "${flare_hidden}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "FLARE25 hidden output inventory is incomplete" >&2; exit 3;
}
"${python_bin}" "${root}/apply_exact707_mask_router.py" \
  --gf "${gf_hidden}" --flare "${flare_hidden}" --report "${router_report}" \
  --output "${routed}" --workers 4 --expected 100
"${python_bin}" - "${project}/validation/Validation-Hidden-Images" "${routed}" "${archive}" \
  "${public_terminal}" "${router_report}" "${terminal}" "${force_router_hidden}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
import nibabel as nib
import numpy as np
images,routed,archive,public,router,terminal=map(Path,sys.argv[1:7]); forced=sys.argv[7]=="1"
expected={p.name.removesuffix("_0000.nii.gz")+".nii.gz":p for p in images.glob("*_0000.nii.gz")}
observed={p.name:p for p in routed.glob("*.nii.gz")}
if len(expected)!=100 or set(observed)!=set(expected): raise SystemExit("hidden routed inventory mismatch")
for name,path in observed.items():
    image=nib.load(str(expected[name])); mask=nib.load(str(path)); values=np.unique(np.asanyarray(mask.dataobj))
    if image.shape!=mask.shape or not np.allclose(image.affine,mask.affine,atol=1e-4): raise SystemExit(f"geometry mismatch {name}")
    if not np.isin(values,[0,1]).all(): raise SystemExit(f"nonbinary {name}")
if archive.exists(): raise SystemExit(f"refusing to overwrite existing archive: {archive}")
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in sorted(observed): z.write(observed[name],arcname=name)
with zipfile.ZipFile(archive) as z:
    members=[i for i in z.infolist() if not i.is_dir()]
    if len(members)!=100 or len({i.filename for i in members})!=100 or z.testzip() is not None: raise SystemExit("ZIP audit failed")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"ARCHIVE_READY_FOR_MANUAL_UPLOAD",
 "submission":{"path":str(archive.resolve()),"sha256":sha(archive),"members":100},
 "public_terminal":{"path":str(public.resolve()),"sha256":sha(public)},
 "router_report":{"path":str(router.resolve()),"sha256":sha(router)},
 "user_explicit_override_of_failed_public_gate":forced,
 "hidden_images_read":True,"hidden_labels_read":False,"submission_created":True,"uploaded":False}
tmp=terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,terminal)
print(json.dumps(payload,indent=2))
PY

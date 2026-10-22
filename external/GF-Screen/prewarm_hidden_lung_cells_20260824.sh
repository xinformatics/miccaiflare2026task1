#!/usr/bin/env bash
set -euo pipefail

# Image-only CPU prewarm for the exact hidden lung-cell proposal cache used by
# all gated exact707 save consumers. No labels, checkpoints, GPUs, packaging,
# or uploads are accessible from this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Hidden-Images"
hidden_root="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"
manifest="${hidden_root}/hidden_manifest.json"
cells="${hidden_root}/cells48"
terminal="${cells}/PREWARM_AUDIT.json"

mkdir -p "${hidden_root}" "${cells}"
exec 9>"${root}/runs/prewarm_hidden_lung_cells_20260824.lock"
flock -n 9 || { echo "hidden lung-cell prewarm already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" - "${images}" "${manifest}" <<'PY'
import json,os,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz",
                  "image":str(image.resolve()),"complete_label":False})
if len(items)!=100 or len({item["name"] for item in items})!=100:
    raise SystemExit("expected exact 100 hidden images")
payload={"description":"label-blind hidden cell manifest","items":items}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

if [[ ! -s "${cells}/PROPOSALS_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/generate_lung_cell_proposals.py" \
    --manifest "${manifest}" --all-cases --output "${cells}" \
    --target-spacing-mm 2.5 --cell-stride-mm 48 --expected-count 100 \
    --workers 2 --resume
fi

"${python_bin}" - "${manifest}" "${cells}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
manifest,cells,output=map(Path,sys.argv[1:])
complete=cells/"PROPOSALS_COMPLETE.json"
payload=json.loads(complete.read_text(encoding="utf-8"))
expected={item["prediction_name"] for item in json.loads(manifest.read_text())["items"]}
if payload.get("cases")!=100 or payload.get("sources")!=8:
    raise SystemExit("hidden cell completion count mismatch")
if payload.get("labels_read") is not False:
    raise SystemExit("hidden cell generator reports label access")
observed={}
for colour in range(8):
    names={path.name for path in (cells/f"colour{colour}").glob("*.nii.gz")}
    if names!=expected: raise SystemExit(f"hidden cell coverage mismatch in colour {colour}")
    observed[str(colour)]=len(names)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit={
 "schema_version":1,"status":"PASS","cases":100,"sources":8,
 "files":sum(observed.values()),"files_by_colour":observed,
 "manifest":str(manifest.resolve()),"manifest_sha256":digest(manifest),
 "generator_terminal":str(complete.resolve()),
 "generator_terminal_sha256":digest(complete),
 "labels_read":False,"checkpoints_read":False,"gpu_used":False,
 "public_labels_read":False,"hidden_labels_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(audit,indent=2))
PY

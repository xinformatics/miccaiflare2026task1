#!/usr/bin/env bash
set -euo pipefail

root=/mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen
project=/mnt/nvme2-data/shashank/MICCAI26
python_bin=/home/m348911/miniconda3/envs/radoncAI/bin/python
qualification="${root}/analysis/subtractive_gate_trace2k_cv_707_20260823/report.json"
fit="${root}/analysis/subtractive_gate_trace2k_final_20260823"
raw="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
anchor="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/subtractive_gate_trace2k_20260823"
prediction="${output}/predictions"
metrics="${output}/metrics.csv"
robust="${output}/robust.json"
terminal="${output}/PUBLIC_COMPLETE.json"

mkdir -p "${output}" "${prediction}"
[[ -s "${terminal}" ]] && exit 0
for path in "${qualification}" "${fit}/FIT_COMPLETE.json" "${fit}/gate.joblib" "${anchor_metrics}"; do
  [[ -s "${path}" ]] || { echo "missing required traceable-gate asset: ${path}" >&2; exit 2; }
done
for directory in "${raw}" "${anchor}" "${images}" "${labels}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done

if [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 || ! -s "${prediction}/GATE_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/apply_subtractive_component_gate.py" \
    --raw-gf "${raw}" --anchor "${anchor}" --images "${images}" \
    --model "${fit}/gate.joblib" --fit-marker "${fit}/FIT_COMPLETE.json" \
    --output "${prediction}" --expected-count 50 --workers 3 \
    > "${output}/apply.log" 2>&1
fi
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
  > "${output}/score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name gf_champ_new2k_s25 --candidate-name gf_gate_trace2k \
  --output "${robust}" > "${output}/robust.log" 2>&1

"${python_bin}" - "${qualification}" "${fit}/FIT_COMPLETE.json" \
  "${prediction}/GATE_COMPLETE.json" "${metrics}" "${robust}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
qualification,fit,apply,metrics,robust,output=map(Path,sys.argv[1:])
q=json.loads(qualification.read_text()); f=json.loads(fit.read_text()); a=json.loads(apply.read_text()); r=json.loads(robust.read_text())
with metrics.open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
def mean(key): return sum(float(row[key]) for row in rows)/len(rows)
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":r["decision"],"cases":len(rows),
 "DSC":mean("Lesion_DSC"),"NSD":mean("Lesion_NSD"),
 "qualification":str(qualification.resolve()),"fit":str(fit.resolve()),
 "application":str(apply.resolve()),"metrics":str(metrics.resolve()),
 "robust":str(robust.resolve()),"robust_payload":r,
 "minimum_deletion_mm3_exclusive":a["minimum_deletion_mm3_exclusive"],
 "labels_read":True,"labels_role":"fixed paired public promotion gate only",
 "hidden_read":False,"submission_created":False,
}
if q.get("decision")!="QUALIFY_GF_BASE_GATE_FOR_COMPOSITION" or not q.get("passed"):
 raise SystemExit("qualification changed before public evaluation")
if f.get("qualification")!=str(qualification.resolve()) or a.get("model")!=f.get("model"):
 raise SystemExit("fit/application provenance mismatch")
temporary=output.with_name(f'.{output.name}.{os.getpid()}.tmp')
temporary.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

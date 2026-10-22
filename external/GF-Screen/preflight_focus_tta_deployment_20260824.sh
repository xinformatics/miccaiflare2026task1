#!/usr/bin/env bash
set -euo pipefail

# CPU-only proof that the no-TTA deployment recipe reproduces the immutable
# champion within locked numerical-equivalence bounds. No GPU, hidden case,
# ZIP, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/public_sweeps/gf_champion_focus_tta_preflight_20260824"
base="${root}/public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000"
support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
labels="${project}/validation/Validation-Public-Labels"
anchor="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
terminal="${output}/PREFLIGHT_COMPLETE.json"

mkdir -p "${output}/prompted" "${output}/deployed" "${root}/runs"
exec 9>"${root}/runs/preflight_focus_tta_deployment_20260824.lock"
flock -n 9 || { echo "TTA deployment preflight already active" >&2; exit 1; }
if [[ -s "${terminal}" ]] && "${python_bin}" -c \
  'import json,sys;raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="PASS" else 1)' \
  "${terminal}"; then
  exit 0
fi
for directory in "${base}" "${support}" "${candidate}" "${labels}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2; exit 2;
  }
done
for required in "${config}" "${anchor}"; do
  [[ -s "${required}" ]] || { echo "missing preflight asset: ${required}" >&2; exit 2; }
done

if [[ $(find "${output}/prompted" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
  "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
    --base "${base}" --support "${support}" --config "${config}" \
    --output "${output}/prompted" > "${output}/prompted.log" 2>&1
fi
[[ $(find "${output}/prompted" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

if [[ $(find "${output}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
  "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
    --base "${output}/prompted" --candidate "${candidate}" --support "${support}" \
    --output "${output}/deployed" --min-mm3 0 --max-mm3 2000 \
    --min-distance-mm 0 --min-support-fraction 0.25 --min-support-mm3 100 \
    --support-dilation-mm inf --workers 4 > "${output}/deployed.log" 2>&1
fi
[[ $(find "${output}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${output}/deployed" --gt "${labels}" --csv "${output}/metrics.csv" \
  --workers 4 > "${output}/score.log" 2>&1
comparison="${output}/deployment_equivalence.json"
"${python_bin}" -u "${root}/compare_focus_deployment_metrics.py" \
  --anchor "${anchor}" --candidate "${output}/metrics.csv" --output "${comparison}"
"${python_bin}" - "${anchor}" "${output}/metrics.csv" "${comparison}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
anchor,candidate,comparison,output=map(Path,sys.argv[1:])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
details=json.loads(comparison.read_text())
payload={"schema_version":2,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"PASS" if details["deployment_equivalent"] else "FAIL",**details,
 "anchor":str(anchor.resolve()),"anchor_sha256":sha(anchor),
 "candidate":str(candidate.resolve()),"candidate_sha256":sha(candidate),
 "comparison":str(comparison.resolve()),"comparison_sha256":sha(comparison),
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
if not details["deployment_equivalent"]:raise SystemExit("no-TTA deployment exceeded locked equivalence bounds")
PY

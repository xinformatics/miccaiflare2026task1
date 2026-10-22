#!/usr/bin/env bash
set -euo pipefail

# CPU-only, label-free prewarming for the two public refiner arms. This removes
# avoidable latency after OOF calibration. No model inference or scoring occurs.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
epoch105="${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
primary="${root}/public_sweeps/candidate_refiner_v2_bbox24_20260822"
primary_work="${primary}/work_all_bbox24"
multi="${root}/public_sweeps/candidate_refiner_v2_multisource_20260823"
multi_union="${multi}/proposal_union"
multi_work="${multi}/work_all_bbox24"

mkdir -p "${primary}" "${multi}"
exec 9>"${root}/runs/prewarm_candidate_refiner_v2_public_caches_20260823.lock"
flock -n 9 || { echo "public refiner cache prewarmer already active" >&2; exit 1; }
[[ -s "${multi}/PREWARM_COMPLETE.txt" ]] && exit 0
for directory in "${images}" "${epoch105}" "${currentgf}" "${base}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected exact 50 public files: ${directory}" >&2; exit 2;
  }
done

if [[ ! -s "${primary_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${images}" --proposal "epoch105=${epoch105}" --base "${base}" \
    --work "${primary_work}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${primary}/prewarm_prepare.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${primary_work}" --output "${primary_work}/AUDIT.json" --workers 2 \
  > "${primary}/prewarm_audit.log" 2>&1
date -Ins > "${primary}/PREWARM_COMPLETE.txt"

# Avoid adding disk contention while the larger 210-case union caches build.
while :; do
  ready=1
  for fold in 0 1 2 3 4; do
    audit="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/fold${fold}/work_bbox24/AUDIT.json"
    [[ -s "${audit}" ]] || ready=0
  done
  [[ "${ready}" -eq 1 ]] && break
  printf '%s waiting for all multisource OOF cache audits\n' "$(date -Ins)"
  sleep 30
done

manifest="${multi}/public_manifest.json"
"${python_bin}" - "${images}" "${manifest}" <<'PY'
import json,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz","image":str(image.resolve()),"complete_label":False})
if len(items)!=50 or len({item["name"] for item in items})!=50: raise SystemExit("expected exact 50 public images")
output.write_text(json.dumps({"description":"unlabeled public geometry manifest","items":items},indent=2)+"\n")
PY
if [[ ! -s "${multi}/PROPOSAL_UNION_COMPLETE.json" ]]; then
  mkdir -p "${multi_union}"
  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
    --pred-a "${epoch105}" --pred-b "${currentgf}" --fusion union --min-mm3 0 \
    --output "${multi_union}" --workers 2 > "${multi}/prewarm_fuse.log" 2>&1
  "${python_bin}" -u "${root}/audit_proposal_union.py" \
    --manifest "${manifest}" --source-a "${epoch105}" --source-b "${currentgf}" \
    --union "${multi_union}" --output "${multi}/PROPOSAL_UNION_COMPLETE.json" \
    --workers 2 --allow-unlabeled > "${multi}/prewarm_union_audit.log" 2>&1
fi
if [[ ! -s "${multi_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${images}" --proposal "union=${multi_union}" --base "${base}" \
    --work "${multi_work}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${multi}/prewarm_prepare.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${multi_work}" --output "${multi_work}/AUDIT.json" --workers 2 \
  > "${multi}/prewarm_cache_audit.log" 2>&1
date -Ins > "${multi}/PREWARM_COMPLETE.txt"

#!/usr/bin/env bash
set -euo pipefail

# CPU-only preparation of an exact-210 union-native training index. This waits
# behind primary submission preservation and cannot train, infer public/hidden,
# package, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
plan_root="${root}/analysis/candidate_refiner_union_native_training_20260823"
plan="${plan_root}/PLAN.json"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
proposal_root="${root}/analysis/candidate_refiner_v2_multisource_20260823"
proposal="${proposal_root}/proposal_union"
proposal_audit="${proposal_root}/PROPOSAL_UNION_COMPLETE.json"
patches="${plan_root}/patches_bbox24_union_v1"
audit="${patches}/AUDIT.json"
primary_saved="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
terminal="${plan_root}/PREPARATION_TERMINAL.json"
expected_manifest_sha="e91fd393ee401bb512148dfa0fc2a084dcb434f238d07bed19dc8aa73e248fbb"
expected_proposal_audit_sha="971822b4ff64927ce2b1ea7d88f66d30078d1909fcec78960c2f186c9e6809f1"
expected_union_inventory="1d36e9803a04b69f646fceb661cbf1f496d4232be1fe536896a345bbb569836b"

mkdir -p "${plan_root}"
exec 9>"${root}/runs/prepare_candidate_refiner_union_native_20260823.lock"
flock -n 9 || { echo "union-native patch preparation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","exit_status":int(sys.argv[2]),"failed_line":sys.argv[3],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${primary_saved}" ]]; do
  printf '%s waiting for primary save-only terminal\n' "$(date -Ins)"
  sleep 30
done

[[ -s "${plan}" && -s "${manifest}" && -s "${proposal_audit}" ]]
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${expected_manifest_sha}" ]]
[[ "$(sha256sum "${proposal_audit}" | awk '{print $1}')" == "${expected_proposal_audit_sha}" ]]
"${python_bin}" - "${proposal_audit}" "${proposal}" "${expected_union_inventory}" <<'PY'
import json,sys
from pathlib import Path
audit=Path(sys.argv[1]); proposal=Path(sys.argv[2]); expected=sys.argv[3]
payload=json.loads(audit.read_text(encoding="utf-8")); required=(
 "exact_file_coverage","all_masks_finite_binary",
 "all_geometries_match_source_images","all_unions_exact_voxelwise",
)
union=payload.get("inventories",{}).get("union",{})
valid=(payload.get("cases")==210 and payload.get("labels_read") is False
       and all(payload.get(key) is True for key in required)
       and Path(union.get("directory","")).resolve()==proposal.resolve()
       and union.get("files")==210 and union.get("sha256_inventory")==expected)
raise SystemExit(0 if valid else "proposal-union audit mismatch")
PY

export MPLCONFIGDIR="${plan_root}/matplotlib"
export TMPDIR="/tmp/gf_union_native_prepare_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
if [[ ! -s "${patches}/COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/prepare_candidate_refiner_patches.py" \
    --manifest "${manifest}" --proposal "union=${proposal}" --output "${patches}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 \
    --max-negatives-per-case 64 --max-negative-tiles 1 --workers 6 \
    > "${plan_root}/prepare.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_patches.py" \
  --index "${patches}/index.csv" --summary "${patches}/summary.json" \
  --output "${audit}" --workers 4 > "${plan_root}/audit.log" 2>&1
"${python_bin}" - "${plan}" "${patches}/summary.json" "${audit}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
plan,summary,audit,output=map(Path,sys.argv[1:]); p=json.loads(plan.read_text())
s=json.loads(summary.read_text()); a=json.loads(audit.read_text())
required=("exact_file_coverage","all_prompts_nonempty","all_values_finite","all_targets_binary_and_label_consistent")
if s.get("cases")!=p["cases"] or a.get("patches")!=s.get("patches") or not all(a.get(key) is True for key in required):
    raise SystemExit("union-native patch audit mismatch")
index=Path(a["index"])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","cases":s["cases"],"patches":a["patches"],
 "positive":a["positive"],"negative":a["negative"],
 "index":str(index.resolve()),"index_sha256":hashlib.sha256(index.read_bytes()).hexdigest(),
 "summary":str(summary.resolve()),"audit":str(audit.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

#!/usr/bin/env bash
set -euo pipefail

# Qualified-route handoff only. This watcher cannot upload: it verifies the
# fixed-public terminal, invokes the exact composite hidden consumer, and
# writes an exact-100 archive identity for manual Codabench upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${root}/public_sweeps/focus_encoder_notta_direct_20260825"
public_terminal="${public}/TERMINAL.json"
route="${public}/ROUTE.json"
calibration="${public}/PREDECLARED_CALIBRATION.json"
saved="${root}/public_sweeps/focus_encoder_notta_saved_20260825"
hidden="${root}/hidden_predictions/focus_encoder_notta_20260825"
archive_prefix="gf_fen25"
save_complete="${saved}/SAVE_ONLY_COMPLETE.txt"
terminal="${saved}/TERMINAL.json"
lock="${root}/runs/save_focus_encoder_notta_after_public_20260825.lock"

mkdir -p "${saved}" "${hidden}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "Focus encoder no-TTA saver already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" \
    "${public_terminal}" "${route}" "${calibration}" "${saved}/saved_weak707_submission.tsv" \
    "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason=sys.argv[1:4]
public,route,calibration,saved,output=map(Path,sys.argv[4:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path):
    return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
archive=None
if saved.is_file():
    rows=list(csv.DictReader(open(saved,encoding="utf-8"),delimiter="\t"))
    if len(rows)!=1: raise SystemExit("expected exactly one saved submission row")
    path=Path(rows[0]["submission"])
    if not path.is_file() or rows[0]["sha256"]!=sha(path):
        raise SystemExit("saved archive identity mismatch")
    with zipfile.ZipFile(path) as handle:
        members=[item for item in handle.infolist() if not item.is_dir()]
        if len(members)!=100 or len({item.filename for item in members})!=100:
            raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members):
            raise SystemExit("archive must contain 100 nonempty flat NIfTI members")
        if handle.testzip() is not None: raise SystemExit("archive CRC failure")
    archive={"path":str(path.resolve()),"sha256":sha(path),"members":100}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "public_terminal":asset(public),"route":asset(route),"calibration":asset(calibration),
 "saved_manifest":asset(saved),"submission":archive,
 "hidden_labels_read":False,"uploaded":False,
}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(tmp,output)
PY
}
failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED NO_ARCHIVE \
    "qualified Focus encoder saver failed at line ${line}, status ${status}"
  exit "${status}"
}
trap failure EXIT

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for direct Focus fixed-public terminal\n' "$(date -Ins)"
  sleep 30
done
decision=$("${python_bin}" - "${public_terminal}" "${route}" "${calibration}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal_path,route,calibration=map(Path,sys.argv[1:])
p=json.load(open(terminal_path,encoding="utf-8"));sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
if p.get("status")!="COMPLETE": raise SystemExit("public terminal is incomplete")
if p.get("decision") not in {"QUALIFY_SUBMISSION_PREPARATION","KEEP_TOUCHCOMP"}:
    raise SystemExit("unexpected public decision")
for key,path in (("route",route),("calibration",calibration)):
    identity=p.get(key,{})
    if Path(identity.get("path","")).resolve()!=path.resolve() or identity.get("sha256")!=sha(path):
        raise SystemExit(f"{key} identity mismatch")
verified={}
for key in ("fixed_public_anchor","paired_gate"):
    identity=p.get(key,{})
    path=Path(identity.get("path",""))
    if not path.is_file() or identity.get("sha256")!=sha(path):
        raise SystemExit(f"{key} identity mismatch")
    verified[key]=path
gate=json.load(open(verified["paired_gate"],encoding="utf-8"))
if gate.get("decision")!=p["paired_gate"].get("decision"):
    raise SystemExit("paired-gate terminal decision mismatch")
if (
    Path(gate.get("anchor_metrics","")).resolve()!=verified["fixed_public_anchor"].resolve()
    or gate.get("anchor_metrics_sha256")!=sha(verified["fixed_public_anchor"])
    or int(gate.get("cases",-1))!=50
):
    raise SystemExit("paired-gate anchor mismatch")
fixed=p.get("fixed_public",{});dsc=float(fixed.get("DSC",-1));nsd=float(fixed.get("NSD",-1))
qualified=(
    dsc>=0.6154316379500749
    and nsd>=0.5104225107769019
    and gate.get("decision")=="PROMOTE"
)
if qualified != (p["decision"]=="QUALIFY_SUBMISSION_PREPARATION"):
    raise SystemExit("public decision/gate mismatch")
if p.get("hidden_read") is not False or p.get("submission_created") is not False or p.get("uploaded") is not False:
    raise SystemExit("public-only provenance mismatch")
print(p["decision"])
PY
)
if [[ "${decision}" == KEEP_TOUCHCOMP ]]; then
  write_terminal SKIPPED KEEP_TOUCHCOMP \
    "direct Focus route missed the fixed-public DSC-first/NSD-nonregression gate"
  trap - EXIT
  exit 0
fi

WEAK707_SAVE_PUBLIC_ROOT="${public}" \
WEAK707_SAVE_CALIBRATION="${calibration}" \
WEAK707_SAVE_COMPOSITE_ROUTE_DECISION="${route}" \
WEAK707_SAVE_OUTPUT="${saved}" \
WEAK707_SAVE_HIDDEN_ROOT="${hidden}" \
WEAK707_SAVE_PROPOSAL_CACHE_ROOT="${hidden}" \
WEAK707_SAVE_TAG_PREFIX="${archive_prefix}" \
WEAK707_SAVE_FAMILY="focus_encoder_notta" \
WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0 \
WEAK707_SAVE_LOCK="${root}/runs/save_focus_encoder_notta_inner_20260825.lock" \
bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh" \
  >> "${saved}/controller.log" 2>&1

[[ -s "${save_complete}" && -s "${saved}/saved_weak707_submission.tsv" ]]
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "qualified composite route reproduced on 100 hidden images and packaged without upload"
trap - EXIT
cat "${saved}/saved_weak707_submission.tsv"

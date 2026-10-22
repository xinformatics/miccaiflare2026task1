#!/usr/bin/env bash
set -euo pipefail

# Public-gated, save-only deployment of the exact-707 prune-composition arm.
# The OOF controller must promote first. A paired fixed-public promotion against
# the deployed champion is then required before any label-blind exact-100 ZIP
# is created. This controller contains no upload operation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
oof="${root}/analysis/candidate_refiner_v4_prune_composition_oof_20260824"
oof_terminal="${oof}/FULL_VOLUME_OOF_TERMINAL.json"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
public="${root}/public_sweeps/candidate_refiner_v4_prune_composition_20260824"
saved="${root}/public_sweeps/candidate_refiner_v4_prune_composition_saved_20260824"
hidden="${root}/hidden_predictions/candidate_refiner_v4_prune_composition_20260824"
public_base="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
hidden_base="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
saved_manifest="${saved}/saved_weak707_submission.tsv"
terminal="${saved}/PRUNE_CHAIN_TERMINAL.json"

mkdir -p "${public}" "${saved}"
exec 9>"${root}/runs/run_candidate_refiner_v4_prune_public_save_20260824.lock"
flock -n 9 || { echo "prune public/save controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3" submission="${4:-}" digest="${5:-}"
  "${python_bin}" - "${oof_terminal}" "${public}" "${saved_manifest}" \
    "${status}" "${decision}" "${reason}" "${submission}" "${digest}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
oof,public,manifest=map(Path,sys.argv[1:4]); status,decision,reason,submission,digest=sys.argv[4:9]; output=Path(sys.argv[9])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"variant":"prune_composition","decision":decision,"reason":reason,
 "oof_terminal":str(oof.resolve()),"public_root":str(public.resolve()),
 "public_read":(public/"PUBLIC_ONLY_COMPLETE.txt").is_file(),"hidden_labels_read":False,
 "submission_created":bool(submission),"submission":submission or None,"sha256":digest or None,
 "saved_manifest":str(manifest.resolve()) if manifest.is_file() else None,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED NO_SUBMISSION "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for prune-composition exact-707 OOF terminal\n' "$(date -Ins)"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != QUALIFY_FOR_FIXED_PUBLIC ]]; then
  write_terminal SKIPPED_OOF KEEP_PRIMARY "exact-707 prune decision was ${oof_decision}"
  trap - EXIT
  exit 0
fi

export WEAK707_PUBLIC_QUALIFICATION="${oof_terminal}"
export WEAK707_PUBLIC_REQUIRED_DECISION=QUALIFY_FOR_FIXED_PUBLIC
export WEAK707_PUBLIC_OOF_ROOT="${oof}"
export WEAK707_PUBLIC_CALIBRATION="${oof}/deployment_calibration.json"
export WEAK707_PUBLIC_TRAIN_ROOT="${train}"
export WEAK707_PUBLIC_SELECTION="${train}/checkpoint_selection.json"
export WEAK707_PUBLIC_RUN_PREFIX="candidate_refiner_v4_cell707_b32_t05_fold"
export WEAK707_PUBLIC_OUTPUT="${public}"
export WEAK707_PUBLIC_TAG_PREFIX=pc707
export WEAK707_PUBLIC_BASE="${public_base}"
export WEAK707_PUBLIC_ALLOW_PRUNE_BASE=1
export WEAK707_PUBLIC_LOCK="${root}/runs/run_candidate_refiner_v4_prune_public_20260824.lock"
bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh" \
  >> "${public}/prune_public_controller.log" 2>&1
[[ -s "${public}/PUBLIC_ONLY_COMPLETE.txt" && -s "${public}/summary.tsv" ]] || {
  echo "prune public evaluation did not produce an audited summary" >&2; exit 3;
}
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" \
  --summary "${public}/summary.tsv" --expected-cases 50)
if [[ -z "${selected}" ]]; then
  write_terminal COMPLETE KEEP_PRIMARY "paired fixed-public gate did not promote"
  trap - EXIT
  exit 0
fi

export WEAK707_SAVE_PUBLIC_ROOT="${public}"
export WEAK707_SAVE_CALIBRATION="${oof}/deployment_calibration.json"
export WEAK707_SAVE_TRAIN_ROOT="${train}"
export WEAK707_SAVE_SELECTION="${train}/checkpoint_selection.json"
export WEAK707_SAVE_RUN_PREFIX="candidate_refiner_v4_cell707_b32_t05_fold"
export WEAK707_SAVE_OUTPUT="${saved}"
export WEAK707_SAVE_HIDDEN_ROOT="${hidden}"
export WEAK707_SAVE_TAG_PREFIX=gf_pc707
export WEAK707_SAVE_FAMILY=candidate_refiner_v4_prune_composition
export WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0
export WEAK707_SAVE_BASE="${hidden_base}"
export WEAK707_SAVE_ALLOW_PRUNE_BASE=1
export WEAK707_SAVE_LOCK="${root}/runs/save_candidate_refiner_v4_prune_20260824.lock"
bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh" \
  >> "${saved}/prune_save_controller.log" 2>&1
[[ -s "${saved}/SAVE_ONLY_COMPLETE.txt" && -s "${saved_manifest}" ]] || {
  echo "promoted prune composition did not produce a saved manifest" >&2; exit 3;
}

read -r submission digest < <("${python_bin}" - "${saved_manifest}" <<'PY'
import csv,hashlib,sys,zipfile
from pathlib import Path
with open(sys.argv[1],newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream,delimiter="\t"))
if len(rows)!=1: raise SystemExit(f"expected one saved candidate, found {len(rows)}")
path=Path(rows[0]["submission"]); expected=rows[0]["sha256"]; actual=hashlib.sha256(path.read_bytes()).hexdigest()
if not path.is_file() or len(path.name)>=64 or actual!=expected: raise SystemExit("archive identity validation failed")
with zipfile.ZipFile(path) as archive:
    members=[item for item in archive.infolist() if not item.is_dir()]
    if len(members)!=100 or len({item.filename for item in members})!=100: raise SystemExit("archive coverage mismatch")
    if any("/" in item.filename or "\\" in item.filename or not item.filename.endswith(".nii.gz") for item in members):
        raise SystemExit("archive is not flat NIfTI")
    if archive.testzip() is not None: raise SystemExit("archive CRC failure")
print(path.resolve(),actual)
PY
)
write_terminal COMPLETE CANDIDATE_SAVED "passed exact-707 OOF and paired fixed-public gates" "${submission}" "${digest}"
trap - EXIT
printf '%s saved %s (%s)\n' "$(date -Ins)" "${submission}" "${digest}"

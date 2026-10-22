#!/usr/bin/env bash
set -euo pipefail

# Controlled public-to-hidden promotion for one exact-707 B32 ablation. The
# variant must first beat the primary on case-held-out full-volume OOF, then
# pass the same paired 50-case public gate against the deployed champion. Only
# a strict public PROMOTE can trigger label-blind hidden inference and creation
# of one exact-100 ZIP. This controller never uploads a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
variant="${1:-}"
precedence_terminal=""
source_routed=0
discovery_run_prefix=""
discovery_run_suffix="20260823"
public_proposal_cache=""
hidden_proposal_cache=""
classifier_route_decision=""

case "${variant}" in
  prompt_pool)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_fivefold_20260824"
    oof="${root}/analysis/candidate_refiner_v4_cell707_prompt_pool_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_prompt_pool_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_prompt_pool_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_prompt_pool_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_promptpool_fold"
    public_prefix="pp707"
    archive_prefix="gf_pp707"
    family="candidate_refiner_v4_cell707_prompt_masked_classifier"
    ;;
  primary_decoupled)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/decoupled_oof"
    oof="${root}/analysis/candidate_refiner_v4_cell707_decoupled_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_decoupled_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_decoupled_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_decoupled_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    public_prefix="dc707"
    archive_prefix="gf_dc707"
    family="candidate_refiner_v4_cell707_decoupled"
    public_proposal_cache="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
    hidden_proposal_cache="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"
    ;;
  classifier_routed)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/source_classifier_routed_oof"
    oof="${root}/analysis/candidate_refiner_v4_cell707_classifier_routed_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_classifier_routed_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_classifier_routed_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_classifier_routed_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    public_prefix="cr707"
    archive_prefix="gf_cr707"
    family="candidate_refiner_v4_cell707_source_routed_classifier"
    classifier_route_decision="${fivefold}/DECISION.json"
    public_proposal_cache="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
    hidden_proposal_cache="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823"
    ;;
  source_condition)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_fivefold_20260824"
    oof="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_source_conditioning_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_source_conditioning_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_source_conditioning_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_srccond_fold"
    public_prefix="sc707"
    archive_prefix="gf_sc707"
    family="candidate_refiner_v4_cell707_source_conditioning"
    ;;
  small_lesion)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_small_loss_fivefold_20260824"
    oof="${root}/analysis/candidate_refiner_v4_cell707_small_loss_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_small_loss_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_small_loss_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_small_loss_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_smallloss30_fold"
    public_prefix="sl707"
    archive_prefix="gf_sl707"
    family="candidate_refiner_v4_cell707_small_lesion_loss"
    precedence_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_source_conditioning_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
    ;;
  uncertain_negative)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_fivefold_20260824"
    oof="${root}/analysis/candidate_refiner_v4_cell707_uncertain_negative_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_uncertain_negative_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_uncertain_negative_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_uncertain_negative_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_uneg025_fold"
    public_prefix="un707"
    archive_prefix="gf_un707"
    family="candidate_refiner_v4_cell707_uncertain_negative_025"
    precedence_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_small_loss_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
    ;;
  source_routed)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/source_routed_oof"
    oof="${root}/analysis/candidate_refiner_v4_cell707_source_routed_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_source_routed_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_source_routed_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_source_routed_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    public_prefix="rt707"
    archive_prefix="gf_rt707"
    family="candidate_refiner_v4_cell707_source_routed_contours"
    source_routed=1
    ;;
  discovery_specialist)
    fivefold="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_fivefold_20260824"
    oof="${root}/analysis/candidate_refiner_v4_cell707_discovery_specialist_fullvolume_oof_20260824"
    public="${root}/public_sweeps/candidate_refiner_v4_cell707_discovery_specialist_20260824"
    saved="${root}/public_sweeps/candidate_refiner_v4_cell707_discovery_specialist_saved_20260824"
    hidden="${root}/hidden_predictions/candidate_refiner_v4_cell707_discovery_specialist_20260824"
    run_prefix="candidate_refiner_v4_cell707_b32_t05_fold"
    discovery_run_prefix="candidate_refiner_v4_cell707_b32_t05_discovery_fold"
    discovery_run_suffix="20260824"
    public_prefix="ds707"
    archive_prefix="gf_ds707"
    family="candidate_refiner_v4_cell707_discovery_specialist"
    source_routed=1
    ;;
  *) echo "usage: $0 prompt_pool|primary_decoupled|classifier_routed|source_condition|small_lesion|uncertain_negative|source_routed|discovery_specialist" >&2; exit 2 ;;
esac

deploy_selection="${fivefold}/checkpoint_selection.json"
[[ "${source_routed}" -eq 1 ]] && deploy_selection="${oof}/checkpoint_selection.json"
[[ "${variant}" == primary_decoupled ]] && deploy_selection="${oof}/checkpoint_selection.json"

oof_terminal="${oof}/FULL_VOLUME_OOF_TERMINAL.json"
public_complete="${public}/PUBLIC_ONLY_COMPLETE.txt"
save_complete="${saved}/SAVE_ONLY_COMPLETE.txt"
saved_manifest="${saved}/saved_weak707_submission.tsv"
terminal="${saved}/VARIANT_CHAIN_TERMINAL.json"
lock="${root}/runs/run_candidate_refiner_v4_${variant}_public_save_20260824.lock"
mkdir -p "${public}" "${saved}"
exec 9>"${lock}"
flock -n 9 || { echo "${variant} public/save controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3" submission="${4:-}" digest="${5:-}"
  "${python_bin}" - "${variant}" "${oof_terminal}" "${public}" "${saved_manifest}" \
    "${status}" "${decision}" "${reason}" "${submission}" "${digest}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
variant=sys.argv[1]; oof,public,manifest=map(Path,sys.argv[2:5])
status,decision,reason,submission,digest=sys.argv[5:10]; output=Path(sys.argv[10])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"variant":variant,"decision":decision,"reason":reason,
 "oof_terminal":str(oof.resolve()),"public_root":str(public.resolve()),
 "public_read":(public/"PUBLIC_ONLY_COMPLETE.txt").is_file(),"hidden_labels_read":False,
 "submission_created":bool(submission),"submission":submission or None,
 "sha256":digest or None,"saved_manifest":str(manifest.resolve()) if manifest.is_file() else None,
 "uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
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

if [[ -n "${precedence_terminal}" ]]; then
  while [[ ! -s "${precedence_terminal}" ]]; do
    printf '%s waiting for higher-priority variant chain terminal: %s\n' \
      "$(date -Ins)" "${precedence_terminal}"
    sleep 30
  done
fi

while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for %s exact-707 full-volume OOF terminal\n' "$(date -Ins)" "${variant}"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != QUALIFY_FOR_FIXED_PUBLIC ]]; then
  write_terminal SKIPPED_OOF KEEP_PRIMARY "exact-707 full-volume OOF decision was ${oof_decision}"
  trap - EXIT
  exit 0
fi

export WEAK707_PUBLIC_QUALIFICATION="${oof_terminal}"
export WEAK707_PUBLIC_REQUIRED_DECISION=QUALIFY_FOR_FIXED_PUBLIC
export WEAK707_PUBLIC_OOF_ROOT="${oof}"
export WEAK707_PUBLIC_CALIBRATION="${oof}/deployment_calibration.json"
export WEAK707_PUBLIC_TRAIN_ROOT="${fivefold}"
export WEAK707_PUBLIC_SELECTION="${deploy_selection}"
export WEAK707_PUBLIC_RUN_PREFIX="${run_prefix}"
export WEAK707_PUBLIC_DISCOVERY_RUN_PREFIX="${discovery_run_prefix:-${run_prefix}}"
export WEAK707_PUBLIC_DISCOVERY_RUN_SUFFIX="${discovery_run_suffix}"
export WEAK707_PUBLIC_SOURCE_ROUTED="${source_routed}"
[[ -z "${classifier_route_decision}" ]] || \
  export WEAK707_PUBLIC_CLASSIFIER_ROUTE_DECISION="${classifier_route_decision}"
export WEAK707_PUBLIC_OUTPUT="${public}"
[[ -z "${public_proposal_cache}" ]] || \
  export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${public_proposal_cache}"
export WEAK707_PUBLIC_TAG_PREFIX="${public_prefix}"
export WEAK707_PUBLIC_LOCK="${root}/runs/run_candidate_refiner_v4_${variant}_public_20260824.lock"
bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh" \
  >> "${public}/variant_public_controller.log" 2>&1
[[ -s "${public_complete}" && -s "${public}/summary.tsv" ]] || {
  echo "${variant} public evaluation did not produce an audited summary" >&2; exit 3;
}

# Strict means strict: unlike the primary exploratory chain, an anchor fallback
# is not eligible for hidden inference in this controlled ablation.
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" \
  --summary "${public}/summary.tsv" --expected-cases 50)
if [[ -z "${selected}" ]]; then
  write_terminal COMPLETE KEEP_PRIMARY "paired 50-case public gate did not promote"
  trap - EXIT
  exit 0
fi

export WEAK707_SAVE_PUBLIC_ROOT="${public}"
export WEAK707_SAVE_CALIBRATION="${oof}/deployment_calibration.json"
export WEAK707_SAVE_TRAIN_ROOT="${fivefold}"
export WEAK707_SAVE_SELECTION="${deploy_selection}"
export WEAK707_SAVE_RUN_PREFIX="${run_prefix}"
export WEAK707_SAVE_DISCOVERY_RUN_PREFIX="${discovery_run_prefix:-${run_prefix}}"
export WEAK707_SAVE_DISCOVERY_RUN_SUFFIX="${discovery_run_suffix}"
export WEAK707_SAVE_SOURCE_ROUTED="${source_routed}"
[[ -z "${classifier_route_decision}" ]] || \
  export WEAK707_SAVE_CLASSIFIER_ROUTE_DECISION="${classifier_route_decision}"
export WEAK707_SAVE_OUTPUT="${saved}"
export WEAK707_SAVE_HIDDEN_ROOT="${hidden}"
[[ -z "${hidden_proposal_cache}" ]] || \
  export WEAK707_SAVE_PROPOSAL_CACHE_ROOT="${hidden_proposal_cache}"
export WEAK707_SAVE_TAG_PREFIX="${archive_prefix}"
export WEAK707_SAVE_FAMILY="${family}"
export WEAK707_SAVE_ALLOW_ANCHOR_FALLBACK=0
export WEAK707_SAVE_LOCK="${root}/runs/save_candidate_refiner_v4_${variant}_20260824.lock"
bash "${root}/save_candidate_refiner_v3_weak707_candidate_20260823.sh" \
  >> "${saved}/variant_save_controller.log" 2>&1
[[ -s "${save_complete}" && -s "${saved_manifest}" ]] || {
  echo "strictly promoted ${variant} did not produce a saved manifest" >&2; exit 3;
}

read -r submission digest < <("${python_bin}" - "${saved_manifest}" <<'PY'
import csv,hashlib,sys,zipfile
from pathlib import Path
with open(sys.argv[1],newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream,delimiter="\t"))
if len(rows)!=1: raise SystemExit(f"expected exactly one saved candidate, found {len(rows)}")
path=Path(rows[0]["submission"]); expected=rows[0]["sha256"]
if not path.is_file() or len(path.name)>=64: raise SystemExit("missing or overlong submission archive")
actual=hashlib.sha256(path.read_bytes()).hexdigest()
if actual!=expected: raise SystemExit("submission checksum mismatch")
with zipfile.ZipFile(path) as archive:
    members=[item for item in archive.infolist() if not item.is_dir()]
    if len(members)!=100 or len({item.filename for item in members})!=100: raise SystemExit("submission coverage mismatch")
    if any("/" in item.filename or "\\" in item.filename or not item.filename.endswith(".nii.gz") for item in members):
        raise SystemExit("submission archive is not a flat NIfTI package")
    if archive.testzip() is not None: raise SystemExit("submission CRC failure")
print(path.resolve(),actual)
PY
)
write_terminal COMPLETE CANDIDATE_SAVED "passed exact-707 OOF and paired 50-case public gates" "${submission}" "${digest}"
trap - EXIT
printf '%s saved %s (%s)\n' "$(date -Ins)" "${submission}" "${digest}"

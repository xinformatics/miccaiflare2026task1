#!/usr/bin/env bash
set -euo pipefail

# Direct challenge-contract-correct positive-unlabeled continuation from the
# CBG epoch-283 Glance checkpoint. Train-Unlabel teacher-empty scans remain
# unknown and never become hard-negative masks. Public data are evaluation
# only; hidden inputs, packaging, and upload are unreachable from this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
dsc_leader="${root}/analysis/DEPLOYMENT_DSC_LEADER.json"
work="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
deployed="${root}/public_sweeps/gf_train_unlabel_cbg_deployed_20260825"
recall_output="${root}/public_sweeps/gf_train_unlabel_cbg_recall_20260825"
balanced="gf_ulcbg_bal_e283_2ep_0825"
conservative="gf_ulcbg_con_e283_2ep_0825"
authorization="${work}/START_AUTHORIZATION.json"
terminal="${work}/TERMINAL.json"
lock="${root}/runs/run_cbg_train_unlabel_direct_20260825.lock"

mkdir -p "${work}" "${deployed}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "direct Train-Unlabel CBG controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${checkpoint}" \
    "${authorization}" "${work}/train_unlabel_audit.json" \
    "${work}/balanced/audit.json" "${work}/conservative/audit.json" \
    "${deployed}/PUBLIC_ONLY_COMPLETE.txt" "${recall_output}/PUBLIC_ONLY_COMPLETE.txt" \
    "${recall_output}/shortlist_ranked.tsv" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]; assets=list(map(Path,sys.argv[4:-1])); output=Path(sys.argv[-1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
keys=("initial_checkpoint","authorization","train_unlabel_audit","balanced_pseudo_audit","conservative_pseudo_audit","public_training_screen_terminal","public_recall_screen_terminal","public_recall_shortlist")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"status":status,"decision":decision,"reason":reason,**{k:asset(p) for k,p in zip(keys,assets)},"train_unlabel_used":asset(assets[2]) is not None,"public_used_for_training":False,"hidden_read":False,"submission_created":False,"uploaded":False}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,output)
PY
}
failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_TOUCHCOMP \
    "Train-Unlabel controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap failure EXIT

for required in "${checkpoint}" "${dsc_leader}" \
    "${root}/configs/challenge_train_labeled_all_partial_20260822.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
"${python_bin}" - "${checkpoint}" "${dsc_leader}" "${authorization}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
checkpoint,leader,output=map(Path,sys.argv[1:]); p=json.load(open(leader,encoding="utf-8"))
if p.get("tag")!="gf_touchcomp_0825" or p.get("archive_sha256")!="d2521c5493169d09719c3c0ecc1b998051dfe5737e2223f85e10d97f39777d11": raise SystemExit("DSC leader identity drift")
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"reason":"direct Train-Unlabel adaptation after invalidating the complete-label interpretation of the 707-case diagnostic set","checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha(checkpoint),"dsc_leader":str(leader.resolve()),"dsc_leader_sha256":sha(leader),"train_unlabel_expected_cases":2856,"teacher_empty_policy":"unknown_not_negative","public_used_for_training":False,"hidden_read":False}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,output)
PY

UNLABELED_CHECKPOINT="${checkpoint}" \
UNLABELED_UPSTREAM="${authorization}" \
UNLABELED_WORK="${work}" \
UNLABELED_ARM_BALANCED="${balanced}" \
UNLABELED_ARM_CONSERVATIVE="${conservative}" \
UNLABELED_DEPLOYED="${deployed}" \
UNLABELED_START_EPOCH=283 \
UNLABELED_MAX_EPOCHS=285 \
UNLABELED_EVAL_EPOCHS="284 285" \
UNLABELED_SKIP_LEGACY_SAVE=1 \
EVALUATE_ALL_EPOCHS=1 \
UNLABELED_GPU0_MAX_MIB=4096 \
UNLABELED_GPU1_MAX_MIB=1024 \
UNLABELED_TEACHER_WORKERS=1 \
UNLABELED_TEACHER_FOCUS_BATCH_SIZE=8 \
UNLABELED_PSEUDO_WORKERS=4 \
UNLABELED_GENERATE_ONLY=1 \
bash "${root}/run_dual_unlabeled_glance_after_allpartial_20260822.sh" \
  >> "${work}/controller.log" 2>&1

if [[ -s "${work}/PSEUDO_LABELS_COMPLETE.json" ]]; then
  write_terminal COMPLETE PSEUDO_LABELS_READY_FOR_LONG_TRAINING \
    "audited Train-Unlabel pseudo labels generated; legacy two-epoch Glance arms intentionally not started"
  trap - EXIT
  exit 0
fi

[[ -s "${deployed}/PUBLIC_ONLY_COMPLETE.txt" ]]
CBG_UL_DEPLOYED="${deployed}" CBG_UL_OUTPUT="${recall_output}" \
CBG_UL_BALANCED="${balanced}" CBG_UL_CONSERVATIVE="${conservative}" \
bash "${root}/evaluate_cbg_unlabeled_recall_additions_public_20260823.sh" \
  >> "${work}/controller.log" 2>&1
[[ -s "${recall_output}/PUBLIC_ONLY_COMPLETE.txt" && -s "${recall_output}/shortlist_ranked.tsv" ]]

decision=$("${python_bin}" - "${recall_output}/shortlist_ranked.tsv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8"),delimiter="\t"))
print("QUALIFY_FOR_TOUCHCOMP_COMPOSITION" if any(r.get("decision")=="PROMOTE" for r in rows) else "KEEP_TOUCHCOMP")
PY
)
write_terminal COMPLETE "${decision}" \
  "positive-unlabeled CBG continuation and untouched-public recall screen complete"
trap - EXIT
cat "${recall_output}/shortlist_ranked.tsv"

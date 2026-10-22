#!/usr/bin/env bash
set -euo pipefail

project="/mnt/nvme2-data/shashank/MICCAI26"
root="${project}/external/FLARE25-Task1-winner"
router_root="${root}/exact707_primary_audit_20260828/mask_router_cv"
report="${router_root}/report.json"
policy="${root}/ROUTER_AUDIT_POLICY_20260828.json"
public_root="${root}/router_public_replay_20260828"
gf_public="${public_root}/gf_e160_champion"
routed="${public_root}/routed"
metrics="${public_root}/routed.metrics.csv"
terminal="${public_root}/TERMINAL.json"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${public_root}"
exec >"${public_root}/run.log" 2>&1
while [[ ! -s "${report}" ]]; do sleep 30; done
if ! "${python_bin}" - "${report}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("status")!="COMPLETE" or not p.get("router_audit_gate_pass"):
    raise SystemExit("router audit gate failed; fixed-public replay forbidden")
PY
then
  "${python_bin}" - "${report}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
report,terminal=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"REJECT_ROUTER_OOF_GATE",
 "reason":"exact-707 nested-CV router audit did not pass its frozen gate",
 "router_report_sha256":sha(report),"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
tmp=terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,terminal)
PY
  exit 0
fi

if [[ ! -d "${gf_public}" ]] || [[ $(find "${gf_public}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 50 ]]; then
  "${python_bin}" "${project}/external/STU-Net/pancancer/materialize_stunet_gf_ensemble.py" \
    --anchor "${project}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions" \
    --stunet "${project}/external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions" \
    --freetumor "${project}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions" \
    --configuration gf_anchor_confirm_either_p70 --output "${gf_public}" --workers 4
fi
"${python_bin}" "${root}/apply_exact707_mask_router.py" \
  --gf "${gf_public}" --flare "${root}/public_audit_20260828/predictions" \
  --report "${report}" --output "${routed}" --workers 4 --expected 50
"${python_bin}" "${project}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${routed}" --gt "${project}/validation/Validation-Public-Labels" \
  --csv "${metrics}" --workers 4
"${python_bin}" - "${metrics}" "${policy}" "${report}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
metrics,policy_path,report_path,terminal=map(Path,sys.argv[1:])
rows=list(csv.DictReader(metrics.open())); policy=json.load(policy_path.open()); report=json.load(report_path.open())
means={key:sum(float(row[key]) for row in rows)/len(rows) for key in ("Lesion_DSC","Lesion_NSD")}
gate=policy["fixed_public_gate"]
checks={"minimum_router_dsc":means["Lesion_DSC"]>=gate["minimum_router_dsc"],
        "minimum_router_nsd":means["Lesion_NSD"]>=gate["minimum_router_nsd"]}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"AUTHORIZE_HIDDEN_SAVE_ONLY" if all(checks.values()) else "REJECT_KEEP_CHAMPION",
 "metrics":means,"gain":{"Lesion_DSC":means["Lesion_DSC"]-gate["current_champion_dsc"],
 "Lesion_NSD":means["Lesion_NSD"]-gate["current_champion_nsd"]},"gate_checks":checks,
 "public_gate_pass":all(checks.values()),"router_report_sha256":sha(report_path),
 "metrics_sha256":sha(metrics),"hidden_read":False,"submission_created":False,"uploaded":False}
tmp=terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,terminal)
print(json.dumps(payload,indent=2))
PY

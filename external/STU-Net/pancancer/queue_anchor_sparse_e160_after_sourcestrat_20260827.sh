#!/usr/bin/env bash
set -euo pipefail

# Queue the public-label-free sparse epoch-160 runtime proof behind both the
# active source-stratified training and its full epoch-40/80 public evaluator.
# Even though the two jobs target different GPUs, concurrent full-volume
# inference previously triggered systemd-oomd through host memory pressure.
# This controller therefore serializes on the evaluator lock and never
# packages or uploads output.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training_terminal="${root}/runs/stunetb_componentbox_e160_sourcesize_f025_cap3_80ep_20260826/TRAINING_TERMINAL.json"
source_public_root="${root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826"
source_public_terminal="${source_public_root}/TERMINAL.json"
source_public_lock="${source_public_root}/controller.lock"
probe="${root}/run_anchor_sparse_e160_probe_20260827.sh"
probe_terminal="${root}/public_sweeps/stunet_anchor_sparse_e160_probe_20260827/EQUIVALENCE.json"
all50="${root}/run_anchor_sparse_e160_public_equivalence_20260827.sh"
all50_terminal="${root}/public_sweeps/stunet_anchor_sparse_e160_all50_20260827/EQUIVALENCE.json"
work="${root}/runs/stunet_anchor_sparse_after_sourcestrat_20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "sparse runtime queue already active" >&2; exit 1; }

for controller in "${probe}" "${all50}"; do
  [[ -s "${controller}" ]] || { echo "missing sparse controller: ${controller}" >&2; exit 2; }
done

while [[ ! -s "${training_terminal}" ]]; do
  printf '%s waiting for source-stratified training terminal\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${training_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
r=p.get("recipe",{})
if (p.get("status")!="COMPLETE" or r.get("epochs")!=80
    or r.get("physical_gpus")!=[0,1] or p.get("public_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid source-stratified training terminal")
PY

# Do not overlap the two endpoint predictors on GPU 1 with sparse full-volume
# inference on GPU 0. Wait for the public terminal, then acquire the controller
# lock so terminal publication and process teardown are both complete.
while [[ ! -s "${source_public_terminal}" ]]; do
  printf '%s waiting for source-stratified fixed-public evaluation terminal\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${source_public_lock}"
flock 8
"${python_bin}" - "${source_public_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("predeclared_epochs")!=[40,80]
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid source-stratified fixed-public terminal")
PY

while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && break
  printf '%s waiting for idle GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done

probe_ok=false
if [[ -s "${probe_terminal}" ]]; then
  probe_ok=$("${python_bin}" - "${probe_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(str(bool(p.get("status")=="COMPLETE" and p.get("probe_pass")
               and p.get("mask_equal") and p.get("geometry_equal")
               and p.get("runtime_eligible_55s")
               and not p.get("deployment_authorized")
               and not p.get("public_labels_read") and not p.get("hidden_read")
               and not p.get("submission_created") and not p.get("uploaded"))).lower())
PY
  )
fi
if [[ "${probe_ok}" != true ]]; then
  SPARSE_PROBE_GPU=0 /usr/bin/bash "${probe}" >"${work}/probe.log" 2>&1
fi

all50_ok=false
if [[ -s "${all50_terminal}" ]]; then
  all50_ok=$("${python_bin}" - "${all50_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(str(bool(p.get("status")=="COMPLETE" and p.get("deployment_authorized")
               and p.get("case_count")==50 and p.get("exact_final_mask_equivalence")
               and p.get("runtime_eligible_55s") and not p.get("public_labels_read")
               and not p.get("hidden_read") and not p.get("submission_created")
               and not p.get("uploaded"))).lower())
PY
  )
fi
if [[ "${all50_ok}" != true ]]; then
  SPARSE_EQUIVALENCE_GPU=0 /usr/bin/bash "${all50}" >"${work}/all50.log" 2>&1
fi

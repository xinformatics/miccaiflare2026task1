#!/usr/bin/env bash
set -euo pipefail

# Public-only DSC-first recovery over the immutable gf_conf champion. The two
# exact-707 model families must independently agree on a disconnected whole
# component. One predeclared large-lesion configuration is evaluated in both
# candidate/support directions. No hidden images, packaging, or upload path is
# present.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
champion="${root}/analysis/DEPLOYMENT_CHAMPION.json"
primary="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823"
focus="${root}/public_sweeps/focus_dualwindow_cell707_20260823"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/gfconf_dsc_first_recovery_20260824"
terminal="${output}/TERMINAL.json"
summary="${output}/summary.tsv"
lock="${root}/runs/run_gfconf_dsc_first_recovery_public_20260824.lock"

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "gf_conf DSC-first recovery watcher already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${champion}" "${summary}" "${status}" "${decision}" \
    "${reason}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
champion,summary=map(Path,sys.argv[1:3]); status,decision,reason=sys.argv[3:6]; output=Path(sys.argv[6])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "champion_manifest":str(champion.resolve()),"champion_manifest_sha256":sha(champion),
 "summary":str(summary.resolve()) if summary.is_file() else None,
 "summary_sha256":sha(summary) if summary.is_file() else None,
 "public_read":summary.is_file(),"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_GFCONF \
    "controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap write_failure EXIT

for required in "${champion}" "${labels}"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
read -r base anchor_metrics expected_public <<< "$(
  "${python_bin}" - "${champion}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"))
base=Path(p["public_predictions"]); metrics=Path(p["public_metrics"])
if not base.is_dir() or not metrics.is_file(): raise SystemExit("champion public assets missing")
if hashlib.sha256(metrics.read_bytes()).hexdigest()!=p["public_metrics_sha256"]:
 raise SystemExit("champion public metrics identity mismatch")
print(base,metrics,int(p["public_cases"]))
PY
)"
[[ "${expected_public}" -eq 50 ]] || { echo "champion public count is not 50" >&2; exit 2; }
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || exit 2
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || exit 2

while [[ ! -s "${primary}/PUBLIC_ONLY_COMPLETE.txt" || \
         ! -s "${focus}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for primary and Focus exact-707 public evaluations\n' "$(date -Ins)"
  sleep 30
done
if [[ ! -s "${primary}/summary.tsv" || ! -s "${focus}/summary.tsv" ]]; then
  write_terminal COMPLETE KEEP_GFCONF \
    "one exact-707 model family did not qualify for fixed-public evaluation"
  trap - EXIT
  exit 0
fi

select_prediction() {
  "${python_bin}" - "$1" <<'PY'
import csv,sys
from pathlib import Path
summary=Path(sys.argv[1]); rows=list(csv.DictReader(summary.open(newline="",encoding="utf-8"),delimiter="\t"))
if not rows: raise SystemExit("empty public summary")
row=max(rows,key=lambda value:(float(value["DSC"]),float(value["NSD"]),value["variant"]))
variant=row["variant"]
if not variant or "/" in variant or "\\" in variant: raise SystemExit("unsafe variant")
prediction=summary.parent/variant
if not prediction.is_dir() or len(list(prediction.glob("*.nii.gz")))!=50:
 raise SystemExit(f"prediction coverage mismatch: {prediction}")
print(f"{prediction.resolve()}\t{variant}")
PY
}
IFS=$'\t' read -r primary_prediction primary_tag <<< "$(select_prediction "${primary}/summary.tsv")"
IFS=$'\t' read -r focus_prediction focus_tag <<< "$(select_prediction "${focus}/summary.tsv")"
[[ "${primary_prediction}" != "${focus_prediction}" ]] || exit 2

# Fixed before either public prediction exists. The >=1000-mm3 emphasis follows
# the internal/public headroom audit: large missed lesions are the only
# available direction that prices DSC and NSD approximately equally.
printf 'direction\tcandidate\tsupport\tDSC\tNSD\tmetrics\trobust\tdecision\n' > "${summary}"
run_direction() {
  local direction="$1" candidate="$2" candidate_tag="$3" support="$4" support_tag="$5"
  local prediction="${output}/${direction}" audit="${output}/${direction}.audit.json"
  local metrics="${output}/${direction}.metrics.csv" robust="${output}/${direction}.robust.json"
  local score_log="${output}/${direction}.score.log" dsc nsd decision
  if [[ ! -s "${audit}" ]]; then
    "${python_bin}" -u "${root}/materialize_immutable_anchor_additions.py" \
      --base "${base}" --candidate "${candidate}" --support "${support}" \
      --output "${prediction}" --audit "${audit}" --expected-count 50 \
      --min-mm3 1000 --max-mm3 inf --min-support-fraction 0.10 \
      --min-support-mm3 100 --min-support-models 1 --workers 2 \
      > "${output}/${direction}.materialize.log" 2>&1
  fi
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_conf_cbg41_e283 --candidate-name "${direction}" \
    --min-dsc-gain 0.003 --min-dsc-win-rate 0.52 --min-nsd-gain 0.0 \
    --min-nsd-uncertainty-bound -0.005 --bootstrap 20000 \
    --output "${robust}" > "${output}/${direction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${direction}" "${candidate_tag}" "${support_tag}" "${dsc}" "${nsd}" \
    "${metrics}" "${robust}" "${decision}" >> "${summary}"
}
run_direction b32_by_focus "${primary_prediction}" "${primary_tag}" \
  "${focus_prediction}" "${focus_tag}"
run_direction focus_by_b32 "${focus_prediction}" "${focus_tag}" \
  "${primary_prediction}" "${primary_tag}"

promoted=$(awk -F '\t' 'NR>1 && $8=="PROMOTE" {n++} END {print n+0}' "${summary}")
if [[ "${promoted}" -gt 0 ]]; then
  write_terminal COMPLETE QUALIFY_FOR_LABEL_BLIND_HIDDEN \
    "at least one immutable consensus-addition direction passed the DSC-first public gate"
else
  write_terminal COMPLETE KEEP_GFCONF \
    "neither immutable consensus-addition direction passed the DSC-first public gate"
fi
trap - EXIT
cat "${summary}"

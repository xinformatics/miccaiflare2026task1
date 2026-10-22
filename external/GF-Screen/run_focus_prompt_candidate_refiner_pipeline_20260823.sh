#!/usr/bin/env bash
set -euo pipefail

# Gate-driven end-to-end controller for prompt-conditioned Focus. It waits for
# primary exact-100 preservation, then runs training-only gates, exact-210 OOF,
# fixed-public qualification, and save-only hidden packaging. No upload command
# exists in this controller or any invoked stage.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
primary_saved="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
pipeline_root="${root}/analysis/focus_prompt_candidate_refiner_pipeline_20260823"
terminal="${pipeline_root}/PIPELINE_TERMINAL.json"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${pipeline_root}"
exec 9>"${root}/runs/run_focus_prompt_candidate_refiner_pipeline_20260823.lock"
flock -n 9 || { echo "prompt Focus pipeline already active" >&2; exit 1; }
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
 "upload_attempted":False,
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

bash "${root}/run_focus_prompt_candidate_refiner_pilot_20260823.sh"
bash "${root}/run_focus_prompt_candidate_refiner_fold1_20260823.sh"
bash "${root}/run_focus_prompt_candidate_refiner_fivefold_20260823.sh"
bash "${root}/run_focus_prompt_candidate_refiner_fullvolume_oof_20260823.sh"
bash "${root}/run_focus_prompt_candidate_refiner_public_20260823.sh"
bash "${root}/save_focus_prompt_candidate_refiner_candidate_20260823.sh"

"${python_bin}" - "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","upload_attempted":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

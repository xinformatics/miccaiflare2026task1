#!/usr/bin/env bash
set -euo pipefail

# Matched one-fold source-conditioning ablation. This script deliberately has
# no public/hidden inference, packaging, submission, or upload. It waits for
# the authoritative primary pipeline, then uses the caller-selected free GPU.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
patches="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48"
source_audit="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/SOURCE_DOMAIN_AUDIT.json"
primary_complete="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/TRAINING_COMPLETE.json"
primary_pipeline_complete="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
discovery_specialist_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_discovery_specialist_saved_20260824/VARIANT_CHAIN_TERMINAL.json"
baseline="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold0_20260823"
output="${root}/runs/candidate_refiner_v4_cell707_b32_t05_srccond_fold0_20260823"
decision_dir="${root}/analysis/candidate_refiner_v4_cell707_source_conditioning_20260823"
decision="${decision_dir}/FOLD0_DECISION.json"
terminal="${decision_dir}/FOLD0_TERMINAL.json"
gpu="${GPU:-0}"

[[ "${gpu}" == 0 || "${gpu}" == 1 ]] || {
  echo "GPU must be 0 or 1" >&2
  exit 2
}
mkdir -p "${output}/matplotlib" "${decision_dir}"
exec 9>"${root}/runs/run_candidate_refiner_v4_source_condition_fold0_20260823.lock"
flock -n 9 || { echo "source-conditioning fold-0 ablation already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${primary_complete}" || ! -s "${primary_pipeline_complete}" \
    || ! -s "${discovery_specialist_terminal}" ]]; do
  printf '%s waiting for exact-707 primary/save terminal and complete higher-priority discovery chain\n' \
    "$(date -Ins)"
  sleep 30
done

for required in \
  "${patches}/COMPLETE.txt" "${patches}/index.csv" "${patches}/AUDIT.json" \
  "${source_audit}" "${baseline}/COMPLETE.txt" "${baseline}/history.json"; do
  [[ -s "${required}" ]] || {
    echo "prerequisite is not complete: ${required}" >&2
    exit 3
  }
done
"${python_bin}" - "${patches}/index.csv" "${source_audit}" <<'PY'
import hashlib,json,sys
from pathlib import Path
index,audit=map(Path,sys.argv[1:]); payload=json.loads(audit.read_text(encoding="utf-8"))
if payload.get("index_sha256") != hashlib.sha256(index.read_bytes()).hexdigest():
    raise SystemExit("source audit/index identity mismatch")
if payload.get("recommendations",{}).get("test_source_conditioning") is not True:
    raise SystemExit("source audit does not recommend source conditioning")
PY

while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
  "${python_bin}" -u "${root}/train_candidate_refiner.py" \
  --index "${patches}/index.csv" --output "${output}" \
  --folds 5 --fold 0 --epochs 30 --batch-size 8 --workers 2 \
  --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
  --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260823 \
  --segmentation-threshold-max 0.95 --validation-interval 3 \
  --validation-batch-size 32 \
  --source-conditioning \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_v4_cell707_b32_t05_srccond_fold0 \
  --resume > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_source_conditioning.py" \
  --baseline-history "${baseline}/history.json" \
  --candidate-history "${output}/history.json" \
  --minimum-segmentation-gain 0.020 \
  --minimum-composite-gain 0.005 \
  --maximum-classification-drop 0.010 \
  --output "${decision}" > "${decision_dir}/select.log" 2>&1

"${python_bin}" - "${patches}/index.csv" "${source_audit}" \
  "${baseline}/history.json" "${output}/history.json" "${decision}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
index,audit,baseline,candidate,decision,output=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,
 "created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE",
 "experiment":"matched exact-707 fold-0 proposal-source conditioning",
 "index":str(index.resolve()),"index_sha256":sha(index),
 "source_audit":str(audit.resolve()),"source_audit_sha256":sha(audit),
 "baseline_history":str(baseline.resolve()),"baseline_history_sha256":sha(baseline),
 "candidate_history":str(candidate.resolve()),"candidate_history_sha256":sha(candidate),
 "decision":str(decision.resolve()),"decision_payload":json.loads(decision.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

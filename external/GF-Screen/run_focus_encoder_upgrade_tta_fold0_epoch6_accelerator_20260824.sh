#!/usr/bin/env bash
set -euo pipefail

# Reusable diagnostic for the current fold-0 lung optimum. It writes into the
# eventual five-fold TTA cache, but cannot promote a route by itself; the final
# pooled selector revalidates all hashes and may ignore this epoch entirely.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
run="${root}/runs/focus_dualwindow_encoder01b8_upgrade_fold0_20260824"
output="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_routed_tta_fivefold_20260824"
directory="${output}/predictions/fold0_epoch0006"
prediction="${directory}/prediction.npz"
summary="${directory}/summary.json"
audit="${directory}/source_audit.json"
terminal="${directory}/EARLY_ACCELERATOR_COMPLETE.json"
checkpoint="${run}/checkpoint_epoch_0006.pt"

mkdir -p "${directory}" "${output}/matplotlib"
exec 9>"${root}/runs/run_focus_encoder_upgrade_tta_fold0_epoch6_accelerator_20260824.lock"
flock -n 9 || { echo "fold-0 encoder TTA accelerator already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
for required in "${index}" "${checkpoint}" \
  "${run}/validation_predictions_epoch_0006.npz"; do
  [[ -s "${required}" ]] || { echo "missing accelerator input: ${required}" >&2; exit 2; }
done
while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s fold-0 encoder TTA waiting for GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done
if [[ ! -s "${summary}" || ! -s "${prediction}" ]]; then
  CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR="${output}/matplotlib" \
    TMPDIR="/tmp/gf_focus_encoder_tta_fold0_epoch6_20260824" \
    "${python_bin}" -u "${root}/evaluate_focus_checkpoint_tta.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --prediction "${prediction}" --summary "${summary}" \
    --folds 5 --fold 0 --seed 20260823 --tta-axes xy \
    --batch-size 8 --workers 2 --segmentation-threshold-max 0.95 \
    > "${directory}/evaluate.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_focus_fold_epoch_sources.py" \
  --index "${index}" --checkpoint "${checkpoint}" --prediction "${prediction}" \
  --folds 5 --fold 0 --seed 20260823 --epoch 6 --output "${audit}" \
  > "${directory}/source_audit.log" 2>&1
"${python_bin}" - "${summary}" "${prediction}" "${audit}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
summary,prediction,audit,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","role":"reusable_diagnostic_not_a_promotion_gate",
 "fold":0,"epoch":6,"summary":str(summary.resolve()),"summary_sha256":sha(summary),
 "prediction":str(prediction.resolve()),"prediction_sha256":sha(prediction),
 "source_audit":str(audit.resolve()),"source_audit_sha256":sha(audit),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

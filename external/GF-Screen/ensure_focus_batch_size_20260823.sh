#!/usr/bin/env bash
set -euo pipefail

# Must be called with exactly one visible, idle CUDA device.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
work="${root}/analysis/focus_candidate_refiner_pilot_20260823"
selection="${work}/BATCH_SELECTION.json"

mkdir -p "${work}"
exec 9>"${work}/BATCH_SELECTION.lock"
flock 9
if [[ ! -s "${selection}" ]]; then
  for batch_size in 8 4; do
    set +e
    "${python_bin}" -u "${root}/probe_focus_candidate_refiner_batch.py" \
      --index "${index}" --pretrained "${pretrained}" --output "${selection}" \
      --batch-size "${batch_size}" --folds 5 --fold 0 --seed 20260822 \
      > "${work}/batch${batch_size}_probe.log" 2>&1
    status=$?
    set -e
    [[ "${status}" -eq 0 ]] && break
    [[ "${status}" -eq 42 ]] || exit "${status}"
  done
fi
[[ -s "${selection}" ]] || { echo "Focus batch sizes 8 and 4 both failed" >&2; exit 2; }
"${python_bin}" - "${selection}" "${index}" "${pretrained}" <<'PY'
import hashlib,json,sys
from pathlib import Path
selection,index,pretrained=map(Path,sys.argv[1:])
p=json.loads(selection.read_text(encoding="utf-8"))
if p.get("status")!="FIT" or int(p.get("batch_size",0)) not in (4,8):
    raise SystemExit("invalid Focus batch selection")
if p.get("index_sha256")!=hashlib.sha256(index.read_bytes()).hexdigest():
    raise SystemExit("Focus batch selection index mismatch")
if p.get("pretrained_sha256")!=hashlib.sha256(pretrained.read_bytes()).hexdigest():
    raise SystemExit("Focus batch selection checkpoint mismatch")
print(int(p["batch_size"]))
PY
